import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings

from einops import rearrange
from gears import PertData, GEARS
import torch
from anndata import AnnData
import scanpy as sc
import scvi
import numpy as np
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
import argparse
import scib
from tqdm import tqdm

from scgpt.trainer import SeqDataset

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model.model_prompt import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor, TFPreprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, load_pretrained

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='COVID-19',help='ms/zheng68k/COVID/NSCLC/MergedMonkey/MergedHuman/mouse_115746/elegans')
parser.add_argument("--data_path", type=str, default='/media/fei/Data/zxy/DATA/', help='Path of data for predicting.')
parser.add_argument("--prompt_type", type=str, default='encoder-prompt',help='encoder-prompt/prefix-prompt/head-prompt/condition-prompt/finetune/LoRA')
parser.add_argument("--space_conf", type=str, default=[1,1,1,1,1,1,0,0,0,0,0,0],help='encoder space adapter list')
parser.add_argument("--mlp_conf", type=str, default=[1,1,1,1,1,1,0,0,0,0,0,0],help='encoder mlp adapter list')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--use_prompt", type=bool, default=True, help='whether use prompt or not.')
parser.add_argument("--gpu_num", type=int, default=1, help='cuda')
args = parser.parse_args()
hyperparameter_defaults = dict(
    seed=0,
    dataset_name=args.data_name,
    do_train=True,
    load_model="/mnt/Data6/scGPT/celltype_annotation/COVID-19/COVID-19-Gene_encoder_prompt/COVID-19_0",
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
    include_zero_gene = True,
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
config = argparse.Namespace(**hyperparameter_defaults)
print(config)

set_seed(config.seed)

# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
# mask_ratio = config.mask_ratio
mask_value = -1
pad_value = -2
n_input_bins = config.n_bins

n_hvg = 2000  # number of highly variable genes
max_seq_len = n_hvg + 1
per_seq_batch_sample = False
DSBN = False  # Domain-spec batchnorm
explicit_zero_prob = config.include_zero_gene  # whether explicit bernoulli for zeros
include_zero_gene = config.include_zero_gene

dataset_name = config.dataset_name
logger = scg.logger
data_dir = Path(config.data_path)

if dataset_name == "COVID-19":
    adata = sc.read("/mnt/Data5/23zxy/scPEFT-main - A6000/newdata/COVID-19/0/COVID-19_test0.h5ad")
    adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
    adata.obs["str_batch"] = 0
    data_is_raw = True
if dataset_name == "NSCLC":
    adata = sc.read("/mnt/Data5/23zxy/scPEFT-main - A6000/newdata/NSCLC/0/NSCLC_test0.h5ad")
    adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
    adata.obs["str_batch"] = 0
    data_is_raw = True
if dataset_name == "ms":
    adata = sc.read("/mnt/Data5/23zxy/scPEFT-main - A6000/newdata/ms_scGPT/4/ms_test4.h5ad")
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    adata.obs["str_batch"] = 0
    data_is_raw = True
    adata.var.set_index(adata.var["gene_name"], inplace=True)
adata.var["gene_name"] = adata.var.index.tolist()

if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "model_fold0.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
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
        f"Resume model from {model_file}, the model args will be overriden by the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=False,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key=None)
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()


if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)
tokenized_test = tokenize_and_pad_batch(
    all_counts,
    gene_ids,
    max_len=len(genes) + 1,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=True,
)

input_values_test = random_mask_value(
    tokenized_test["values"],
    mask_ratio=0.0,
    mask_value=mask_value,
    pad_value=pad_value,
)

test_data_pt = {
    "gene_ids": tokenized_test["genes"],
    "values": input_values_test,
    "target_values": tokenized_test["values"],
    # "celltype_labels": torch.from_numpy(celltypes_labels).long(),
}

test_loader = DataLoader(
    dataset=SeqDataset(test_data_pt),
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), 1 // 2),
    pin_memory=True,
)
all_gene_ids, all_values = tokenized_test["genes"], tokenized_test["values"]
src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=False,
    do_dab=False,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style="continuous",
    n_input_bins=n_input_bins,
    cell_emb_style="cls",
    mvc_decoder_style="inner product",
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=False,
    fast_transformer_backend="flash",
    pre_norm=config.pre_norm,
    **prompt_settings
)
device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file,map_location=device))
        logger.info(f"Loading all model params from {model_file}")
    except:
        use_flash_attn = getattr(model, "use_fast_transformer", True)
        pretrained_dict = torch.load(model_file, map_location=device)
        if not use_flash_attn:
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

model.to(device)
num_attn_layers=11
cos_cls_genes = []
pcc_cls_genes = []
input_gene_ids_prompt = []
cell_embeddings=[]
for batch_data in tqdm(test_loader):
    M = batch_data["gene_ids"].size(1)
    input_gene_id = batch_data["gene_ids"].to(device)
    input_values = batch_data["values"].to(device)
    src_key_padding_mask = input_gene_id.eq(vocab[pad_token])
    src_embs = model.encoder(torch.tensor(input_gene_id, dtype=torch.long).to(device))
    val_embs = model.value_encoder(torch.tensor(input_values, dtype=torch.float).to(device))
    total_embs = src_embs + val_embs
    for layer in model.transformer_encoder.layers[:num_attn_layers]:
        total_embs = layer(total_embs, src_key_padding_mask=src_key_padding_mask.to(device))
    attn_weight = model.transformer_encoder.layers[num_attn_layers].self_attn.in_proj_weight
    attn_bias = model.transformer_encoder.layers[num_attn_layers].self_attn.in_proj_bias
    qkv = F.linear(total_embs, attn_weight, attn_bias)
    # Retrieve q, k, and v from flast-attn wrapper
    qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=8)
    q = qkv[:, :, 0, :, :]
    k = qkv[:, :, 1, :, :]
    v = qkv[:, :, 2, :, :]
    attn_scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
    # index=np.where(input_gene_id.detach().cpu().numpy()==60694)[1].min()
    # attn_scores[0,:,:,index:]=0
    # attn_scores[0, :, index:, :] = 0
    #
    attn_scores = attn_scores.reshape((-1, M))
    order = torch.argsort(attn_scores, dim=1)
    rank = torch.argsort(order, dim=1)
    attn_scores = rank.reshape((-1, 8, M, M)) / M
    # Rank normalization by column
    attn_scores = attn_scores.permute(0, 1, 3, 2).reshape((-1, M))
    order = torch.argsort(attn_scores, dim=1)
    rank = torch.argsort(order, dim=1)
    attn_scores = (rank.reshape((-1, 8, M, M)) / M).permute(0, 1, 3, 2)

    attn_scores = attn_scores.mean(1)
    outputs = attn_scores
    # 按行取 八个头
    # for i in range(len(input_values)):
    #     pcc =outputs[0, :, 0, 1:].detach().cpu().numpy()
    #     pcc_cls_genes.append(pcc)
    # 按列取
    # for i in range(len(input_values)):
    #     pcc = outputs[0, 1:, 0].detach().cpu().numpy()
    #     pcc_cls_genes.append(pcc)
    #按行取
    for i in range(len(input_values)):
        pcc = outputs[0, 0, 1:].detach().cpu().numpy()
        pcc_cls_genes.append(pcc)
    input_gene_ids_prompt.extend(input_gene_id[:, 1:])
    #按列 相加
    # for i in range(len(input_values)):
    #     pcc = outputs[0].detach().cpu().numpy().sum(axis=0)[1:]/1843
    #     pcc_cls_genes.append(pcc)
# X = np.zeros((len(adata), gene_ids.size))
# for i in range(len(adata)):
#     index = np.where(np.isin(gene_ids, input_gene_ids_prompt[i].detach().cpu().numpy()))[0]
#     for j, idx in enumerate(index):
#         X[i, idx] = pcc_cls_genes[i][j]
# adata.X = X
# adata.X=np.array(pcc_cls_genes)

genes=[]


# X=torch.stack(pcc_cls_genes)
# X = np.zeros((len(adata), gene_ids.size))
# for i in tqdm(range(len(adata))):
#     gene_name=[]
#     index = np.where(np.isin(gene_ids, input_gene_ids_prompt[i].detach().cpu().numpy()))[0]
#     for j, idx in enumerate(index):
#         # X[i, idx] = pcc_cls_genes[i][j]
#         gene_name.append(adata[i].var_names[idx])
#     # 计算需要补齐的长度差
#     pad_length = len(input_gene_ids_prompt[i]) - len(gene_name)
#     # 如果长度不一样，用 'pad' 填充
#     if pad_length > 0:
#         gene_name.extend(['pad'] * pad_length)
#     genes.append(gene_name)
# adata.X = X
# import numpy as np
# from tqdm import tqdm
#
# # 将 input_gene_ids_prompt 提前转换为 numpy 数组
# input_gene_ids_numpy = [x.detach().cpu().numpy() for x in input_gene_ids_prompt]
#
# genes = []
#
# for i in tqdm(range(len(adata))):
#     gene_name = []
#
#     # 使用 numpy 来加速 isin 操作
#     index = np.where(np.isin(gene_ids, input_gene_ids_numpy[i]))[0]
#
#     # 使用索引直接操作
#     # X[i, index] = pcc_cls_genes[i][:len(index)]
#
#     # 添加基因名称
#     gene_name = adata[i].var_names[index].tolist()
#
#     # 直接计算需要的填充长度并扩展
#     pad_length = len(input_gene_ids_numpy[i]) - len(gene_name)
#     gene_name.extend(['pad'] * pad_length)
#
#     genes.append(gene_name)
# #
# adata.obsm["gene_name"]=np.array(genes)
adata.obsm["value"]=np.array(pcc_cls_genes)



sc.write(f'/mnt/Data6/scGPT/celltype_annotation/COVID-19/923/{dataset_name}_{args.prompt_type}.h5ad', adata)




































