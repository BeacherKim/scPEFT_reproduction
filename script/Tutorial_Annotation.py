import copy
import gc
import argparse
import json
import os
import  random
from pathlib import Path
import shutil
import sys
sys.path.append("/mnt/Data5/23zxy/scPEFT-main - A6000/tutorials")
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
import loralib as lora
import pickle
import torch
from anndata import AnnData
import scanpy as sc
import scvi
import seaborn as sns
import numpy as np
import wandb
from numpy import mean
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from sklearn.metrics import confusion_matrix
# from scgpt.kfloddata import KFold
sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model.model_prompt import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor, TFPreprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='ms_scGPT',help='ms/zheng68k/COVID/NSCLC/MergedMonkey/MergedHuman/mouse_115746/elegans')
parser.add_argument("--data_path", type=str, default='/mnt/Data5/23zxy/scPEFT-main - A6000/newdata/', help='Path of data for predicting.')
parser.add_argument("--prompt_type", type=str, default='encoder-prompt',help='encoder-prompt/prefix-prompt/head-prompt/condition-prompt/finetune/LoRA')
parser.add_argument("--space_conf", type=str, default=[1,1,1,1,1,1,0,0,0,0,0,0],help='encoder space adapter list')
parser.add_argument("--mlp_conf", type=str, default=[0,0,0,0,0,0,0,0,0,0,0,0],help='encoder mlp adapter list')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--use_prompt", type=bool, default=True, help='whether use prompt or not.')
parser.add_argument("--gpu_num", type=int, default=2, help='cuda')
args = parser.parse_args()
hyperparameter_defaults = dict(
    seed=0,
    dataset_name=args.data_name,
    do_train=True,
    load_model="/mnt/Data5/23zxy/scPEFT-main - A6000/scGPT_human",
    mask_ratio=0.0,
    epochs=args.epoch,
    n_bins=51,
    MVC=False, # Masked value prediction for cell embedding
    ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=1e-4,
    batch_size=40,
    layer_size=128,
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=4,  # number of heads in nn.MultiheadAttention
    dropout=0.2,  # dropout probability
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer= False,  #是否使用flash attention
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    include_zero_gene = False,
    freeze = False, #freeze
    DSBN = False,  # Domain-spec batchnorm
    data_path=args.data_path,
    use_prompt=args.use_prompt,
    prompt_type=args.prompt_type,  # prefix_prompt/Gene_encoder_prompt/Gene_token_prompt/LoRA
    num_tokens=64,
    n_layers_conf=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # token
    mlp_adapter_conf=args.mlp_conf,
    space_adapter_conf=args.space_conf,
)
#%%
config = argparse.Namespace(**hyperparameter_defaults)
print(config)

set_seed(config.seed)

# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"  # for masked values, now it should always be auto

include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
max_seq_len = 2001
n_bins = config.n_bins

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = config.MVC  # Masked value prediction for cell embedding
ECS = config.ecs_thres > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

# settings for optimizer
lr = config.lr  # TODO: test learning rate ratio between two tasks
lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1

# settings for the model
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability

data_path = config.data_path
use_prompt = config.use_prompt
prompt_type = config.prompt_type
num_tokens = config.num_tokens
n_layers_conf = config.n_layers_conf
mlp_adapter_conf = config.mlp_adapter_conf
space_adapter_conf = config.space_adapter_conf

# logging
log_interval = 100  # iterations
save_eval_interval = config.save_eval_interval  # epochs
do_eval_scib_metrics = True

# %% validate settings
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False

dataset_name = config.dataset_name
save_dir = Path(f"./save/{dataset_name}_disease_train_healthy_test/{prompt_type}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

data_dir = Path(data_path + dataset_name)

if dataset_name=="MergedHuman":
    n_splits = 4
elif dataset_name=="immune":
    n_splits = 1
else:
    n_splits = 5
accuracys=[]
precisions=[]
recalls=[]
macro_f1s=[]
mic_precisions=[]
mic_recalls=[]
micro_f1s=[]

for i in range(n_splits):
    if dataset_name == "ms_scGPT":

        adata = sc.read(data_dir / f"{i}/ms_train{i}.h5ad")
        adata_val = sc.read(data_dir / f"{i}/ms_val{i}.h5ad")
        adata_test = sc.read(data_dir / f"{i}/ms_test{i}.h5ad")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"
        adata = adata.concatenate((adata_test, adata_val), batch_key="str_batch")
        adata_val = adata[adata.obs["Sample Characteristic[individual]"].isin(["C1", "C2", "C3"])]
        adata_test = adata[
            adata.obs["Sample Characteristic[individual]"].isin(["C4", "C5", "C6", "C7", "C8", "C9"])]
        adata = adata[adata.obs["Sample Characteristic[individual]"].isin(
            ["MS1", "MS2", "MS3", "MS4", "MS5", "MS6", "MS7", "MS8", "MS9", "MS10", "MS11", "MS12"])]

        # adata_test=adata_test[(adata_test.obs["Sample Characteristic[individual]"].isin(["C2","C3"]))]
        # adata_test=adata_test[adata_test.obs["celltype"]=="cortical layer 2-3 excitatory neuron A"]
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        adata_val.obs["celltype"] = adata_val.obs["celltype"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["celltype"].astype("category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_val.var.set_index(adata_val.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata_test.var["gene_name"], inplace=True)
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate((adata_test, adata_val), batch_key="str_batch")

        cell_types = adata.obs["celltype"]

        # 计算每种细胞类型的频次
        cell_type_counts = cell_types.value_counts()

        # 计算比例
        cell_type_percentages = cell_type_counts / len(cell_types) * 100
        cell_type_percentages_formatted = cell_type_percentages.apply(lambda x: f"{x:.2f}%")
        # 输出结果
        print(cell_type_percentages_formatted)
        print(adata[adata.obs['Sample Characteristic[disease]'] == "multiple sclerosis"].obs[
                  'Sample Characteristic[individual]'].unique().__len__())
        print(adata[adata.obs['Sample Characteristic[disease]'] == "normal"].obs[
                  'Sample Characteristic[individual]'].unique().__len__())
    elif dataset_name == "zheng68k":
        adata = sc.read(data_dir / "zheng68k_train.h5ad")
        adata_val = sc.read(data_dir / "zheng68k_val.h5ad")
        adata_test = sc.read(data_dir / "zheng68k_test.h5ad")
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["celltype"].astype("category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"
        adata.var.set_index(adata.var.index, inplace=True)
        adata_val.var.set_index(adata_val.var.index, inplace=True)
        adata_test.var.set_index(adata_test.var.index, inplace=True)
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate((adata_test, adata_val), batch_key="str_batch")
    elif dataset_name=="immune":
        barcodes_file = "/mnt/Data5/23zxy/scPEFT-main - A6000/case_contrl/CD34_idents.csv"
        barcodes_df = pd.read_csv(barcodes_file, header=None, names=["barcode", "celltype"])
        barcode = barcodes_df["barcode"].values.tolist()
        barcode_celltype_dict = dict(zip(barcodes_df["barcode"], barcodes_df["celltype"]))

        # data_dir = "/path/to/data"

        CD34_D2T1_adata = sc.read(f"/mnt/Data5/23zxy/scPEFT-main - A6000/case_contrl/data/immune/GSM4138874_scRNA_CD34_D2T1.h5ad")
        CD34_D2T1_adata.obs["str_batch"] = "CD34_D2T1"
        CD34_D2T1_adata = CD34_D2T1_adata[CD34_D2T1_adata.obs.index.isin(barcode)]
        CD34_D2T1_adata.obs["celltype"] = CD34_D2T1_adata.obs.index.map(barcode_celltype_dict)

        CD34_D3T1_adata = sc.read(f"/mnt/Data5/23zxy/scPEFT-main - A6000/case_contrl/data/immune/GSM4138875_scRNA_CD34_D3T1.h5ad")
        CD34_D3T1_adata.obs["str_batch"] = "CD34_D3T1"
        CD34_D3T1_adata = CD34_D3T1_adata[CD34_D3T1_adata.obs.index.isin(barcode)]
        CD34_D3T1_adata.obs["celltype"] = CD34_D3T1_adata.obs.index.map(barcode_celltype_dict)
        adata_new = CD34_D3T1_adata.concatenate(CD34_D2T1_adata)
        # Shuffle the data
        adata_new = adata_new[adata_new.obs.index.to_series().sample(frac=1).index]

        # Split the data into train and test sets
        adata, adata_test = train_test_split(adata_new, test_size=0.2, random_state=42)

        # Extract 10% of the training data for validation
        adata, adata_val = train_test_split(adata, test_size=0.1, random_state=42)
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["celltype"].astype("category")
        adata_val.obs["celltype"] = adata_val.obs["celltype"].astype("category")
        adata = adata.concatenate((adata_test, adata_val), batch_key="str_batch")
        data_is_raw = True
    elif dataset_name == "COVID-19":
        adata = sc.read(data_dir / f"{i}/COVID-19_train{i}.h5ad")
        adata_val = sc.read(data_dir / f"{i}/COVID-19_val{i}.h5ad")
        adata_test = sc.read(data_dir / f"{i}/COVID-19_test{i}.h5ad")
        adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
        adata_val.obs["celltype"] = adata_val.obs["cell_type"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"
        adata.var.set_index(adata.var.index, inplace=True)
        adata_val.var.set_index(adata_val.var.index, inplace=True)
        adata_test.var.set_index(adata_test.var.index, inplace=True)
        data_is_raw = True
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate((adata_test, adata_val), batch_key="str_batch")
    elif dataset_name == "NSCLC_noHvg":
        adata = sc.read(data_dir / f"{i}/NSCLC_train{i}.h5ad")
        adata_val = sc.read(data_dir / f"{i}/NSCLC_val{i}.h5ad")
        adata_test = sc.read(data_dir / f"{i}/NSCLC_test{i}.h5ad")

        adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
        adata_val.obs["celltype"] = adata_val.obs["cell_type"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"
        adata.var.set_index(adata.var.index, inplace=True)
        adata_val.var.set_index(adata_val.var.index, inplace=True)
        adata_test.var.set_index(adata_test.var.index, inplace=True)
        data_is_raw = True
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate((adata_test, adata_val), batch_key="str_batch")
        max_seq_len = adata.X.shape[1] + 1
    elif dataset_name == "MergedMonkey":
        adata = sc.read(data_dir / f"{i}/MergedMonkey_train{i}.h5ad")
        adata_val = sc.read(data_dir / f"{i}/MergedMonkey_val{i}.h5ad")
        adata_test = sc.read(data_dir / f"{i}/MergedMonkey_test{i}.h5ad")
        adata.obs["celltype"] = adata.obs["CellType"].astype("category")
        adata_val.obs["celltype"] = adata_val.obs["CellType"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["CellType"].astype("category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"
        adata.var.set_index(adata.var.gene_name, inplace=True)
        adata_val.var.set_index(adata_val.var.gene_name, inplace=True)
        adata_test.var.set_index(adata_test.var.gene_name, inplace=True)
        data_is_raw = True
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate((adata_test, adata_val), batch_key="str_batch")
    elif dataset_name == "MergedHuman":
        adata = sc.read(data_dir / f"{i}/MergedHuman_train{i}.h5ad")
        adata_val = sc.read(data_dir / f"{i}/MergedHuman_val{i}.h5ad")
        adata_test = sc.read(data_dir / f"{i}/MergedHuman_test{i}.h5ad")
        adata.obs["celltype"] = adata.obs["CellType"].astype("category")
        adata_val.obs["celltype"] = adata_val.obs["CellType"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["CellType"].astype("category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"
        adata.var.set_index(adata.var.gene_name, inplace=True)
        adata_val.var.set_index(adata.var.gene_name, inplace=True)
        adata_test.var.set_index(adata.var.gene_name, inplace=True)
        data_is_raw = True
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate((adata_test, adata_val), batch_key="str_batch")
    elif dataset_name == "NSCLC":
        adata = sc.read(data_dir / f"{i}/NSCLC_train{i}.h5ad")
        adata_val = sc.read(data_dir / f"{i}/NSCLC_val{i}.h5ad")
        adata_test = sc.read(data_dir / f"{i}/NSCLC_test{i}.h5ad")
        adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
        adata_val.obs["celltype"] = adata_val.obs["cell_type"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"
        adata.var.set_index(adata.var.gene_name, inplace=True)
        adata_val.var.set_index(adata_val.var.gene_name, inplace=True)
        adata_test.var.set_index(adata_test.var.gene_name, inplace=True)
        data_is_raw = True
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate((adata_test, adata_val), batch_key="str_batch")
    elif dataset_name == "mouse_115746_filtered":
        adata = sc.read(data_dir / f"{i}/mouse_115746_train{i}.h5ad")
        adata_val = sc.read(data_dir / f"{i}/mouse_115746_val{i}.h5ad")
        adata_test = sc.read(data_dir / f"{i}/mouse_115746_test{i}.h5ad")
        adata.obs["celltype"] = adata.obs["cell_subclass"].astype("category")
        adata_val.obs["celltype"] = adata_val.obs["cell_subclass"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["cell_subclass"].astype("category")
        # adata = adata[~(adata.obs["celltype"].isin(['Low Quality', 'No Class', 'Non-Neuronal']))]
        # adata_val = adata_val[~(adata_val.obs["celltype"].isin(['Low Quality', 'No Class', 'Non-Neuronal']))]
        # adata_test = adata_test[~(adata_test.obs["celltype"].isin(['Low Quality', 'No Class', 'Non-Neuronal']))]
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_val.var.set_index(adata.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata.var["gene_name"], inplace=True)
        data_is_raw = True
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate((adata_test, adata_val), batch_key="str_batch")
    elif dataset_name == "elegans":
        adata = sc.read(data_dir / f"{i}/elegans_train{i}.h5ad")
        adata_val = sc.read(data_dir / f"{i}/elegans_val{i}.h5ad")
        adata_test = sc.read(data_dir / f"{i}/elegans_test{i}.h5ad")
        adata.obs["celltype"] = adata.obs["tissue_name"].astype("category")
        adata_val.obs["celltype"] = adata_val.obs["tissue_name"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["tissue_name"].astype("category")
        adata = adata[~(adata.obs["celltype"]=="NA")]
        adata_val = adata_val[~(adata_val.obs["celltype"]=="NA")]
        adata_test = adata_test[~(adata_test.obs["celltype"]=="NA")]
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"
        adata.var["gene_name"] = adata.var["orthogene"]
        adata_val.var["gene_name"] = adata_val.var["orthogene"]
        adata_test.var["gene_name"] = adata_test.var["orthogene"]
        data_is_raw = True
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate((adata_test, adata_val), batch_key="str_batch")

        # make the batch category column
    elif dataset_name == "mouse_10x":
        adata = sc.read(f"/mnt/Data5/23zxy/scPEFT-main - A6000/mouse/mouse_unified_5fold/{i}/mouse_115746_train{i}.h5ad")
        adata_test = sc.read(f"/mnt/Data5/23zxy/scPEFT-main - A6000/mouse/mouse_unified_10x/mouse_test{i}.h5ad")
        vocab = adata_test.var["gene_name"].values
        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        # Shuffle the data
        adata = adata[adata.obs.index.to_series().sample(frac=1).index]
        # Split the data into train and test sets
        adata, adata_val = train_test_split(adata, test_size=0.2, random_state=42)

        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
        adata_val.obs["batch_id"] = adata_val.obs["str_batch"] = "2"
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["celltype"].astype("category")
        adata_val.obs["celltype"] = adata_val.obs["celltype"].astype("category")
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_val.var.set_index(adata_val.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata_test.var["gene_name"], inplace=True)
        adata = adata.concatenate((adata_test, adata_val),batch_key="str_batch")

        filter_gene_by_counts = False
        data_is_raw = True


    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    class_num = np.unique(celltype_id_labels, return_counts=True)[1].tolist()
    class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
    celltypes = adata.obs["celltype"].unique()
    num_types = len(np.unique(celltype_id_labels))
    print(f'num_types:{num_types}')
    celltype_dict = dict(zip(celltypes.codes, celltypes.T))
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
    adata.obs["celltype_id"] = celltype_id_labels
    if dataset_name == "elegans":
        pass
    else:
        adata.var["gene_name"] = adata.var.index.tolist()

    if config.load_model is not None:
        model_dir = Path(config.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        shutil.copy(vocab_file, save_dir / "vocab.json")
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]

    # set up the preprocessor, use the args to config the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key=None)
    adata_test = adata[adata.obs["str_batch"] == "1"]
    adata = adata[adata.obs["str_batch"] != "1"]

    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[input_style]
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    genes = adata.var["gene_name"].tolist()
    train_celltype_labels = adata[adata.obs["str_batch"] == "0"].obs["celltype_id"].tolist()  # make sure count from 0
    train_celltype_labels = np.array(train_celltype_labels)

    valid_celltype_labels = adata[adata.obs["str_batch"] == "2"].obs["celltype_id"].tolist()  # make sure count from 0
    valid_celltype_labels = np.array(valid_celltype_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))

    train_batch_labels = np.array(adata[adata.obs["str_batch"] == "0"].obs["batch_id"].tolist())
    valid_batch_labels = np.array(adata[adata.obs["str_batch"] == "2"].obs["batch_id"].tolist())

    adata_val = adata[adata.obs["str_batch"] == "2"]
    adata = adata[adata.obs["str_batch"] == "0"]
    train_data = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    valid_data = (
        adata_val.layers[input_layer_key].A
        if issparse(adata_val.layers[input_layer_key])
        else adata_val.layers[input_layer_key]
    )

    if config.load_model is None:
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )



    def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
        masked_values_train = random_mask_value(
            tokenized_train["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        masked_values_valid = random_mask_value(
            tokenized_valid["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        # print(
        #     f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        #     f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        # )

        input_gene_ids_train, input_gene_ids_valid = (
            tokenized_train["genes"],
            tokenized_valid["genes"],
        )
        input_values_train, input_values_valid = masked_values_train, masked_values_valid
        target_values_train, target_values_valid = (
            tokenized_train["values"],
            tokenized_valid["values"],
        )

        tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
        tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

        tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
        tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

        if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
            train_sort_ids = np.argsort(train_batch_labels)
            input_gene_ids_train = input_gene_ids_train[train_sort_ids]
            input_values_train = input_values_train[train_sort_ids]
            target_values_train = target_values_train[train_sort_ids]
            tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
            tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

            valid_sort_ids = np.argsort(valid_batch_labels)
            input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
            input_values_valid = input_values_valid[valid_sort_ids]
            target_values_valid = target_values_valid[valid_sort_ids]
            tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
            tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

        train_data_pt = {
            "gene_ids": input_gene_ids_train,
            "values": input_values_train,
            "target_values": target_values_train,
            "batch_labels": tensor_batch_labels_train,
            "celltype_labels": tensor_celltype_labels_train,
        }
        valid_data_pt = {
            "gene_ids": input_gene_ids_valid,
            "values": input_values_valid,
            "target_values": target_values_valid,
            "batch_labels": tensor_batch_labels_valid,
            "celltype_labels": tensor_celltype_labels_valid,
        }

        return train_data_pt, valid_data_pt


    # dataset
    class SeqDataset(Dataset):
        def __init__(self, data: Dict[str, torch.Tensor]):
            self.data = data

        def __len__(self):
            return self.data["gene_ids"].shape[0]

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}


    # data_loader
    def prepare_dataloader(
            data_pt: Dict[str, torch.Tensor],
            batch_size: int,
            shuffle: bool = False,
            intra_domain_shuffle: bool = False,
            drop_last: bool = False,
            num_workers: int = 0,
            sampler:str=None,
    ) -> DataLoader:
        if num_workers == 0:
            num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

        dataset = SeqDataset(data_pt)

        if per_seq_batch_sample:
            # find the indices of samples in each seq batch
            subsets = []
            batch_labels_array = data_pt["batch_labels"].numpy()
            for batch_label in np.unique(batch_labels_array):
                batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
                subsets.append(batch_indices)
            data_loader = DataLoader(
                dataset=dataset,
                batch_sampler=SubsetsBatchSampler(
                    subsets,
                    batch_size,
                    intra_subset_shuffle=intra_domain_shuffle,
                    inter_subset_shuffle=shuffle,
                    drop_last=drop_last,
                ),
                num_workers=num_workers,
                pin_memory=True,
            )
            return data_loader

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
        )
        return data_loader

    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    prompt_settings = {
        "use_prompt": config.use_prompt,
        "num_tokens": config.num_tokens,
        "prompt_type": config.prompt_type,
        "n_layers_conf": config.n_layers_conf,
        "mlp_adapter_conf": config.mlp_adapter_conf,
        "space_adapter_conf": config.space_adapter_conf
    }

    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=3,
        n_cls=num_types if CLS else 1,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=MVC,
        do_dab=DAB,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=config.DSBN,
        input_emb_style=input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        ecs_threshold=ecs_threshold,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=config.pre_norm,
        **prompt_settings
    )
    if config.load_model is not None:
        try:
            model.load_state_dict(torch.load(model_file,map_location=device))
            logger.info(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            use_flash_attn = getattr(model, "use_fast_transformer", True)
            pretrained_dict = torch.load(model_file,map_location=device)
            if not use_flash_attn and prompt_type != "LoRA":
                pretrained_dict = {
                    k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_dict.items()
                }

            model_dict = model.state_dict()
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() ).values())

    for name, para in model.named_parameters():
        para.requires_grad = False
    for name, para in model.named_parameters():
        if 'lora_' in name:
            para.requires_grad = True
        if 'cls' in name:
            para.requires_grad = True
        if 'prompt_embeddings' in name:
            para.requires_grad = True
        if 'Adapter' in name:
            para.requires_grad = True
    post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad ).values())
    if prompt_type == "LoRA":
        def lora_bias_trainable(model: nn.Module, bias: str = 'none') -> None:
            if bias == 'none':
                return
            elif bias == 'all':
                for n, p in model.named_parameters():
                    if 'bias' in n:
                        p.requires_grad = True
            elif bias == 'lora_only':
                for m in model.modules():
                    if isinstance(m, lora.LoRALayer) and \
                            hasattr(m, 'bias') and \
                            m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError

        lora_bias_trainable(model, bias='lora_only')
    post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad ).values())
    learnable_params = {k: v for k, v in model.named_parameters() if v.requires_grad == True}
    for k, v in learnable_params.items():
        logger.info(f"Learnable params {k} with shape {v.shape}")
    logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
    logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")
    logger.info(f"learnable params / total params: {(post_freeze_param_count / pre_freeze_param_count):.4f}")
    model.to(device)

    if ADV:
        discriminator = AdversarialDiscriminator(
            d_model=embsize,
            n_cls=num_batch_types,
        ).to(device)

    criterion = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss(weight=class_weight.to(device))
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, schedule_interval, gamma=config.schedule_ratio
    )
    if DAB_separate_optim:
        optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler_dab = torch.optim.lr_scheduler.StepLR(
            optimizer_dab, schedule_interval, gamma=config.schedule_ratio
        )
    if ADV:
        criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
        optimizer_E = torch.optim.Adam(model.parameters(), lr=lr_ADV)
        scheduler_E = torch.optim.lr_scheduler.StepLR(
            optimizer_E, schedule_interval, gamma=config.schedule_ratio
        )
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_ADV)
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            optimizer_D, schedule_interval, gamma=config.schedule_ratio
        )

    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    #%%
    def train(model: nn.Module, loader: DataLoader) -> None:
        """
        Train the model for one epoch.
        """
        model.train()
        (
            total_loss,
            total_mse,
            total_cls,
            total_cce,
            total_mvc,
            total_ecs,
            total_dab,
            total_adv_E,
            total_adv_D,
            total_zero_log_prob,
            total_mvc_zero_log_prob,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        total_error = 0.0
        start_time = time.time()

        num_batches = len(loader)
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            if use_prompt and prompt_type == 'prefix_prompt':
                bool_tensor = torch.empty((input_gene_ids.shape[0],num_tokens), dtype=torch.bool).to(device)
                bool_tensor.fill_(src_key_padding_mask[0][0].item()).to(device)
                src_key_padding_mask = torch.cat((bool_tensor,src_key_padding_mask),dim=1)
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=do_sample_in_train,
                    #generative_training=False
                )

                masked_positions = input_values.eq(mask_value)  # the postions to predict
                loss = 0.0
                metrics_to_log = {}
                if MLM:
                    loss_mse = criterion(
                        output_dict["mlm_output"], target_values, masked_positions
                    )
                    loss = loss + loss_mse
                    metrics_to_log = {"train/mse": loss_mse.item()}
                if explicit_zero_prob:
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                if CLS:
                    loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                    loss = loss + loss_cls
                    metrics_to_log.update({"train/cls": loss_cls.item()})

                    error_rate = 1 - (
                        (output_dict["cls_output"].argmax(1) == celltype_labels)
                        .sum()
                        .item()
                    ) / celltype_labels.size(0)
                if CCE:
                    loss_cce = 10 * output_dict["loss_cce"]
                    loss = loss + loss_cce
                    metrics_to_log.update({"train/cce": loss_cce.item()})
                if MVC:
                    loss_mvc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    loss = loss + loss_mvc
                    metrics_to_log.update({"train/mvc": loss_mvc.item()})
                if MVC and explicit_zero_prob:
                    loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_mvc_zero_log_prob
                    metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
                if ECS:
                    loss_ecs = 10 * output_dict["loss_ecs"]
                    loss = loss + loss_ecs
                    metrics_to_log.update({"train/ecs": loss_ecs.item()})
                if DAB:
                    # try weighting and separate optimizer
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                    loss = loss + dab_weight * loss_dab
                    metrics_to_log.update({"train/dab": loss_dab.item()})

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            if ADV:
                # rerun the model for adversarial training
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=do_sample_in_train,
                    #generative_training=False
                )

                # TRAINING DISCRIMINATOR
                loss_adv_D = criterion_adv(
                    discriminator(output_dict["cell_emb"].detach()), batch_labels
                )
                if epoch > adv_D_delay_epochs:
                    discriminator.zero_grad()
                    loss_adv_D.backward()
                    optimizer_D.step()

                # TRAINING ENCODER
                loss_adv_E = -criterion_adv(
                    discriminator(output_dict["cell_emb"]), batch_labels
                )
                # NOTE: the loss is negative here because we want to maximize
                # the cross_entropy_loss, in other words, disguise against the discriminator
                if epoch > adv_E_delay_epochs:
                    model.zero_grad()
                    discriminator.zero_grad()
                    loss_adv_E.backward()
                    optimizer_E.step()

            total_loss += loss.item()
            total_mse += loss_mse.item() if MLM else 0.0
            total_cls += loss_cls.item() if CLS else 0.0
            total_cce += loss_cce.item() if CCE else 0.0
            total_mvc += loss_mvc.item() if MVC else 0.0
            total_ecs += loss_ecs.item() if ECS else 0.0
            total_dab += loss_dab.item() if DAB else 0.0
            total_adv_E += loss_adv_E.item() if ADV else 0.0
            total_adv_D += loss_adv_D.item() if ADV else 0.0
            total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
            total_mvc_zero_log_prob += (
                loss_mvc_zero_log_prob.item() if MVC and explicit_zero_prob else 0.0
            )
            total_error += error_rate
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_cls = total_cls / log_interval if CLS else 0.0
                cur_cce = total_cce / log_interval if CCE else 0.0
                cur_mvc = total_mvc / log_interval if MVC else 0.0
                cur_ecs = total_ecs / log_interval if ECS else 0.0
                cur_dab = total_dab / log_interval if DAB else 0.0
                cur_adv_E = total_adv_E / log_interval if ADV else 0.0
                cur_adv_D = total_adv_D / log_interval if ADV else 0.0
                cur_zero_log_prob = (
                    total_zero_log_prob / log_interval if explicit_zero_prob else 0.0
                )
                cur_mvc_zero_log_prob = (
                    total_mvc_zero_log_prob / log_interval
                    if MVC and explicit_zero_prob
                    else 0.0
                )
                cur_error = total_error / log_interval
                # ppl = math.exp(cur_loss)
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | "
                    + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if MLM else "")
                    + (f"cls {cur_cls:5.2f} | " if CLS else "")
                    + (f"err {cur_error:5.2f} | " if CLS else "")
                    + (f"cce {cur_cce:5.2f} |" if CCE else "")
                    + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
                    + (f"ecs {cur_ecs:5.2f} |" if ECS else "")
                    + (f"dab {cur_dab:5.2f} |" if DAB else "")
                    + (f"adv_E {cur_adv_E:5.2f} |" if ADV else "")
                    + (f"adv_D {cur_adv_D:5.2f} |" if ADV else "")
                    + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "")
                    + (
                        f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                        if MVC and explicit_zero_prob
                        else ""
                    )
                )
                total_loss = 0
                total_mse = 0
                total_cls = 0
                total_cce = 0
                total_mvc = 0
                total_ecs = 0
                total_dab = 0
                total_adv_E = 0
                total_adv_D = 0
                total_zero_log_prob = 0
                total_mvc_zero_log_prob = 0
                total_error = 0
                start_time = time.time()


    def define_wandb_metrcis():
        wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
        wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
        wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
        wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
        wandb.define_metric("test/avg_bio", summary="max")


    def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
        """
        Evaluate the model on the evaluation data.
        """
        model.eval()
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0
        predictions = []
        with torch.no_grad():
            for batch_data in loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                target_values = batch_data["target_values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)
                celltype_labels = batch_data["celltype_labels"].to(device)

                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                # if use_prompt and prompt_type == 'prefix_prompt':
                #     bool_tensor = torch.empty((input_gene_ids.shape[0], num_tokens), dtype=torch.bool).to(device)
                #     bool_tensor.fill_(src_key_padding_mask[0][0].item()).to(device)
                #     src_key_padding_mask = torch.cat((bool_tensor, src_key_padding_mask), dim=1)
                with torch.cuda.amp.autocast(enabled=config.amp):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                        CLS=CLS,  # evaluation does not need CLS or CCE
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=do_sample_in_train,
                        #generative_training = False,
                    )
                    output_values = output_dict["cls_output"]
                    loss = criterion_cls(output_values, celltype_labels)

                    if DAB:
                        loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

                total_loss += loss.item() * len(input_gene_ids)
                accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
                total_num += len(input_gene_ids)
                preds = output_values.argmax(1).cpu().numpy()
                predictions.append(preds)


        if return_raw:
            return np.concatenate(predictions, axis=0)

        return total_loss / total_num, total_error / total_num


    class EarlyStopping():
        def __init__(self, patience=5, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
        def __call__(self, val_loss):
            if self.best_loss == None:
                self.best_loss = val_loss
            elif self.best_loss - val_loss > self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            elif self.best_loss - val_loss < self.min_delta:
                self.counter += 1
                logger.info(f"INFO: Early stopping counter {self.counter} of {self.patience}")
                if self.counter >= self.patience:
                    logger.info('INFO: Early stopping')
                    self.early_stop = True


    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None

    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)

    train_num = train_data.shape[0]
    train_class_num = np.unique(train_data_pt['celltype_labels'],return_counts=True)[1]
    class_weights = 1.0 / train_class_num[train_data_pt['celltype_labels']]
    sample_weights = class_weights / np.sum(class_weights)

    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, train_num, replacement=True)
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
        sampler=train_sampler,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=eval_batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

    early_stopping = EarlyStopping()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        if config.do_train:
            train(
                model,
                loader=train_loader,
            )
        val_loss, val_err = evaluate(
            model,
            loader=valid_loader,
        )

        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")

        scheduler.step()
        if DAB_separate_optim:
            scheduler_dab.step()
        if ADV:
            scheduler_D.step()
            scheduler_E.step()

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    torch.save(best_model.state_dict(), save_dir / f"model_fold{i}.pt")

    logger.info('Best model saved successfully!')

    def test(model: nn.Module, adata: DataLoader) -> float:
        all_counts = (
            adata.layers[input_layer_key].A
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )

        celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
        celltypes_labels = np.array(celltypes_labels)

        batch_ids = adata.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)

        tokenized_test = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=include_zero_gene,
        )

        input_values_test = random_mask_value(
            tokenized_test["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )

        test_data_pt = {
            "gene_ids": tokenized_test["genes"],
            "values": input_values_test,
            "target_values": tokenized_test["values"],
            "batch_labels": torch.from_numpy(batch_ids).long(),
            "celltype_labels": torch.from_numpy(celltypes_labels).long(),
        }

        test_loader = DataLoader(
            dataset=SeqDataset(test_data_pt),
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
            pin_memory=True,
        )

        model.eval()
        predictions = evaluate(
            model,
            loader=test_loader,
            return_raw=True,
        )


        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

        balanced_accuracy = balanced_accuracy_score(celltypes_labels, predictions)
        f1 = f1_score(celltypes_labels, predictions, average="macro")
        precision = precision_score(celltypes_labels, predictions, average="macro")
        recall = recall_score(celltypes_labels, predictions, average="macro")

        logger.info(f"fold{i}:"
                    f"macro Accuracy: {balanced_accuracy:.3f}, macro Precision: {precision:.3f},macro Recall: {recall:.3f}, "
                    f"macro F1: {f1:.3f}"
                    )
        micro_f1 = f1_score(celltypes_labels, predictions, average="micro")
        micro_precision = precision_score(celltypes_labels, predictions, average="micro")
        micro_recall = recall_score(celltypes_labels, predictions, average="micro")
        logger.info(f"fold{i}:"
                    f"micro Accuracy: {balanced_accuracy:.3f}, micro Precision: {micro_precision:.3f}, micro Recall: {micro_recall:.3f}, "
                    f"micro F1: {micro_f1:.3f}")
        results = {
            "test/accuracy": balanced_accuracy,
            "test/mac_precision": precision,
            "test/mac_recall": recall,
            "test/mac_f1": f1,
            "test/mic_precision": micro_precision,
            "test/mic_recall": micro_recall,
            "test/mic_f1": micro_f1,
        }

        return predictions, celltypes_labels, results

    predictions, labels, results = test(best_model, adata_test)
    accuracys.append(results["test/accuracy"])
    precisions.append(results["test/mac_precision"])
    recalls.append(results["test/mac_recall"])
    macro_f1s.append(results["test/mac_f1"])
    mic_precisions.append(results["test/mic_precision"])
    mic_recalls.append(results["test/mic_recall"])
    micro_f1s.append(results["test/mic_f1"])
    save_dict = {
        "predictions": predictions,
        "labels": labels,
        "results": results,
        "id_maps": id2type
    }
    with open(save_dir / f"results_fold{i}.pkl", "wb") as f:
        pickle.dump(save_dict, f)

logger.info(" aver outcome:"
    f"macro Accuracy: {mean(accuracys):.3f}, macro Precision: {mean(precisions):.3f},macro Recall: {mean(recalls):.3f}, "
    f"macro F1: {mean(macro_f1s):.3f}"
)
logger.info(" variance outcome:"
    f"macro Accuracy: {np.sqrt(np.var(accuracys)):.3f}, macro Precision: {np.sqrt(np.var(precisions)):.3f},macro Recall: {np.sqrt(np.var(recalls)):.3f}, "
    f"macro F1: {np.sqrt(np.var(macro_f1s)):.3f}"
)
logger.info(" aver outcome:"
            f"micro Accuracy: {mean(accuracys):.3f}, micro Precision: {mean(mic_precisions):.3f},micro Recall: {mean(mic_recalls):.3f}, "
            f"micro F1: {mean(micro_f1s):.3f}"
            )
logger.info(" variance outcome:"
            f"micro Accuracy: {np.sqrt(np.var(accuracys)):.3f}, micro Precision: {np.sqrt(np.var(mic_precisions)):.3f},micro Recall: {np.sqrt(np.var(mic_recalls)):.3f}, "
            f"micro F1: {np.sqrt(np.var(micro_f1s)):.3f}"
            )
