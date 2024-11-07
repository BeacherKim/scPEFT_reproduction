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
import psutil
import threading
import shutil
from contextlib import nullcontext

import torch
from anndata import AnnData
import scanpy as sc
import scvi
import numpy as np
import wandb
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
import pickle
import argparse

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model.dsbn import DomainSpecificBatchNorm1d
from scgpt.model.model_prompt import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
import scib
from scgpt.preprocess import Preprocessor, TFPreprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, load_pretrained

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

hyperparameter_defaults = dict(
    seed=42,
    dataset_name="immune",  # Dataset name
    do_train=True,  # Flag to indicate whether to do update model parameters during training
    load_model="/mnt/Data3/22frx/scGPT_pretrain/scGPT_human",  # Path to pre-trained model
    GEPC=True,  # Gene expression modelling for cell objective
    ecs_thres=0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0,  # DAR objective weight for batch correction
    mask_ratio=0.4,  # Default mask ratio
    mlm_probability=[0.15, 0.25, 0.4],
    epochs=25,  # Default number of epochs for fine-tuning
    n_bins=51,  # Default number of bins for value binning in data pre-processing
    lr=1e-4,  # Default learning rate for fine-tuning
    batch_size=20,  # Default batch size for fine-tuning
    layer_size=128,
    nlayers=4,
    nhead=4,  # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.2,  # Default dropout rate during model fine-tuning
    schedule_ratio=0.9,  # Default rate for learning rate decay
    save_eval_interval=1,  # Default model evaluation interval
    log_interval=100,  # Default log interval
    fast_transformer=False,  # Default setting
    pre_norm=False,  # Default setting
    amp=True,  # # Default setting: Automatic Mixed Precision
    data_path='/mnt/Data5/23zxy/scPEFT-main - A6000/case_contrl/data/immune',
    use_prompt=True,
    prompt_type='LoRA',
    # encoder-prompt/prefix-prompt/head-prompt/condition-prompt/finetune/LoRA
    num_tokens=20,
    n_layers_conf=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # token
    mlp_adapter_conf=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    space_adapter_conf=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    include_zero_gene=False,
    do_batch_classification=False
)

config = argparse.Namespace(**hyperparameter_defaults)
print(config)

set_seed(config.seed)

# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = -1
pad_value = -2
n_input_bins = config.n_bins

n_hvg = 2000  # number of highly variable genes
max_seq_len = n_hvg + 1
per_seq_batch_sample = False
DSBN = False  # Domain-spec batchnorm
explicit_zero_prob = config.include_zero_gene
include_zero_gene = config.include_zero_gene
do_batch_classification = config.do_batch_classification

dataset_name = config.dataset_name
save_dir = Path(f"./save/Unsupervise/Unsupervise_CaseControl_{dataset_name}_{config.prompt_type}_with_TFs/")
save_dir.mkdir(parents=True, exist_ok=True)
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")
data_dir = Path(config.data_path)

if dataset_name == 'immune':
    adata=sc.read("/mnt/Data5/23zxy/scPEFT-main - A6000/case_contrl/data/immune/immune_mix.h5ad")
    adata.obs["str_batch"] = 0
    adata.obs["celltype"] = adata.obs["annot"]
    data_is_raw = True
elif dataset_name == 'tonsil_B_PC':
    data_dir = Path("/mnt/Data3/22frx/scGPT_temporary/CaseControl/data/tonsil")
    file_name = "PC.h5ad"
    adata = sc.read(data_dir / file_name)

    adata.obs["celltype"] = adata.obs["annotation_20230508"]
    adata.obs["str_batch"] = 0
    data_is_raw = False
elif dataset_name == 'tonsil_B_NBC':
    data_dir = Path("/mnt/Data3/22frx/scGPT_temporary/CaseControl/data/tonsil")
    file_name = "NBC.h5ad"
    adata = sc.read(data_dir / file_name)

    adata.obs["celltype"] = adata.obs["annotation_20230508"]
    adata.obs["str_batch"] = adata.obs["assay"]
    data_is_raw = False
elif dataset_name == 'tonsil_B_MBC':
    data_dir = Path("/mnt/Data3/22frx/scGPT_temporary/CaseControl/data/tonsil")
    file_name = "MBC.h5ad"
    adata = sc.read(data_dir / file_name)
    adata = adata[adata.obs["assay"].isin(["3P"])]

    adata.obs["celltype"] = adata.obs["annotation_20230508"]
    adata.obs["str_batch"] = adata.obs["assay"]

    PriorMarkers_Early_MBC = ["CD27"]
    PriorMarkers_GC_commited_NBC = [" FCER2", "TCLA1", "IGHM", "IGHD"]
    PriorMarkers_MBC_FCRL5 = ["CD27", "TNFRSF13B"]
    PriorMarkers_NBC = ["FCER2", "TCLA1", "IGHM", "IGHD"]
    PriorMarkers_NBC_early_activation = ["FCER2", "TCLA1", "IGHM", "IGHD"]
    PriorMarkers_csMBC = ["CD27", "TNFRSF13B","IGHA1","IGHG1"]
    PriorMarkers_FCRL45=["CD27", "TNFRSF13B","IGHA", "IGHG", "GSN","SOX5","FGR","HCK"]
    PriorMarkers_ncsMBCFCRL4=["CD27", "TNFRSF13B","IGHM", "IGHD","FCRL4", "FCRL5"]
    PriorMarkers_ncsMBC=["CD27", "TNFRSF13B","IGHD","IGHM"]
    all_markers_list = [
        "CD27",
        "FCER2", "TCLA1", "IGHM", "IGHD",
        "CD27", "TNFRSF13B",
        "FCER2", "TCLA1", "IGHM", "IGHD",
        "FCER2", "TCLA1", "IGHM", "IGHD",
        "CD27", "TNFRSF13B","IGHA1","IGHG1",
        "CD27", "TNFRSF13B", "IGHA", "IGHG", "GSN", "SOX5", "FGR", "HCK",
        "CD27", "TNFRSF13B", "IGHM", "IGHD", "FCRL4", "FCRL5",
        "CD27", "TNFRSF13B", "IGHD", "IGHM"
    ]

    # 将列表转换为集合，去除重复元素
    unique_markers_set = set(all_markers_list)

    # 将集合转换回列表，如果需要以列表形式输出
    unique_markers_list = list(unique_markers_set)
    PriorMarkers=unique_markers_list
    data_is_raw = False
elif dataset_name == 'tonsil_B_GCBC':
    data_dir = Path("/mnt/Data3/22frx/scGPT_temporary/CaseControl/data/tonsil")
    file_name = "GCBC.h5ad"
    adata = sc.read(data_dir / file_name)
    adata = adata[adata.obs["assay"].isin(["3P"])]
    print(f"{file_name} have assay {set(adata.obs['assay'])}")

    adata.obs["celltype"] = adata.obs["annotation_20230508"]
    adata.obs["str_batch"] = adata.obs["assay"]
    data_is_raw = False

# make the batch category column
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
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
        f"Resume model from {model_file}, the model args will be overriden by the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
# PriorMarkers = [PriorMarker for PriorMarker in PriorMarkers if
#                 PriorMarker in adata.var.index.tolist()] if PriorMarkers else None
# for load TF genes
TF_gene_path = "/mnt/Data5/23zxy/scPEFT-main - A6000/TF_names_v_1.01.txt"
with open(TF_gene_path, 'r') as file:
    TF_genes = [line.strip() for line in file.readlines()]
    # Filter not in vocab FT genes
    TF_genes_in_vocab = [TF_gene for TF_gene in TF_genes if TF_gene in vocab]

# set up the preprocessor, use the args to config the workflow
preprocessor = TFPreprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=False,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    TF_genes=TF_genes_in_vocab,
)

adata = preprocessor(adata, batch_key=None)
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

# For batch correction
batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids)) if do_batch_classification else None
batch_ids = np.array(batch_ids)

(
    train_data,
    valid_data,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, batch_ids, test_size=0.1, shuffle=True
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


def random_mask_value(
        values: Union[torch.Tensor, np.ndarray],
        mask_ratio: float = 0.15,
        mask_value: int = -1,
        pad_value: int = 0,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        # expect cls token
        non_padding_idx = non_padding_idx[1:]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()


def prepare_data(sort_seq_batch: bool = False, mask_ratio: float = 0.4) -> Tuple[Dict[str, torch.Tensor]]:
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
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

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

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
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
) -> DataLoader:
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
    )
    return data_loader


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config.GEPC,
    do_dab=do_batch_classification,  # default is false
    num_batch_labels=num_batch_types,
    use_batch_labels=False,
    n_input_bins=n_input_bins,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
    **prompt_settings
)
if config.load_model is not None:
    # only load params that are in the model and match the size
    use_flash_attn = getattr(model, "use_fast_transformer", True)
    pretrained_dict = torch.load(model_file, map_location=device)
    if not use_flash_attn and config.prompt_type != "LoRA":
        pretrained_dict = {
            k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_dict.items()
        }

    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    logger.info("-" * 89)
    for k, v in pretrained_dict.items():
        logger.info(f"Loading params {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

for name, para in model.named_parameters():
    para.requires_grad = False
for param in model.decoder.parameters():
    param.requires_grad = True
for param in model.mvc_decoder.parameters():
    param.requires_grad = True
for name, para in model.named_parameters():
    if 'lora_' in name:
        para.requires_grad = True
    if 'prompt_embeddings' in name:
        para.requires_grad = True
    if 'Adapter' in name:
        para.requires_grad = True

if config.do_batch_classification:
    for param in model.grad_reverse_discriminator.parameters():
        param.requires_grad = True

logger.info("-" * 89)
post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
learnable_params = {k: v for k, v in model.named_parameters() if v.requires_grad == True}
for k, v in learnable_params.items():
    logger.info(f"Learnable params {k} with shape {v.shape}")

logger.info("Total Pre freeze Params: %.2fM" % (pre_freeze_param_count / 1e6,))
logger.info("Total Post freeze Params: %.2fM" % (post_freeze_param_count / 1e6,))

model.to(device)

criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


def eval_scib_metrics(
        adata: AnnData,
        batch_key: str = "str_batch",
        label_key: str = "celltype",
) -> Dict:
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed="X_scGPT",
        isolated_labels_asw_=False,
        silhouette_=False,
        hvg_score_=False,
        graph_conn_=False,
        pcr_=False,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )
    result_dict = results[0].to_dict()

    from sklearn import metrics

    # ARI
    ari = metrics.adjusted_rand_score(labels_true=adata.obs[label_key], labels_pred=adata.obs["cluster"])

    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}
    result_dict["ARI"] = ari
    logger.info(
        "Clustering Metrics: \n"
        f"ARI(cell-type): {result_dict['ARI']:.4f}, NMI(cell-type): {result_dict['NMI_cluster/label']:.4f}, "
    )

    return result_dict


def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse, total_gepc, total_zero_log_prob, total_dab = 0.0, 0.0, 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                MVC=config.GEPC,
                ECS=False,
            )

            masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )

            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
            if config.GEPC:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc

            if config.GEPC and explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_gepc_zero_log_prob

            if config.do_batch_classification:
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + loss_dab

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

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_dab += loss_dab.item() if config.do_batch_classification else 0.0
        total_error += mre.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_dab = total_dab / log_interval if config.do_batch_classification else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                + (f"dab {cur_dab:5.2f} |" if config.do_batch_classification else "")
            )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_error = 0
            total_dab = 0
            start_time = time.time()


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_values, target_values, masked_positions)

            total_loss += loss.item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    return total_loss / total_num, total_error / total_num


def eval_testdata(
        model: nn.Module,
        adata_t: AnnData,
        include_types: List[str] = ["cls"],
        best_model_epoch: int = 0,
        epoch: int = 0
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()

    # copy adata_t to avoid reuse previously computed results stored in adata_t
    adata_t = adata_t.copy()

    all_counts = (
        adata_t.layers[input_layer_key].A
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )

    # Evaluate cls cell embeddings
    if "cls" in include_types:
        logger.info("Evaluating cls cell embeddings")
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=include_zero_gene,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=config.batch_size,
                batch_labels=None,
                time_step=0,
                return_np=True,
            )
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_scGPT"] = cell_embeddings

        results = {}
        try:
            results = eval_scib_metrics(adata_t)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["celltype"],
            title=[
                f"celltype, ARI = {results.get('ARI', 0.0):.4f}",
            ],
            frameon=False,
            show=False,
            save=f"_{dataset_name}_Unsupervised_{config.prompt_type.replace('-', '_').upper()}_embeddings_celltype_e{best_model_epoch}"
        )

    if epoch == config.epochs:
        n = 20
        cluster_key = "cluster"
        resolutions = [2 * x / n for x in range(1, n + 1)]
        sc.pp.neighbors(adata_t, use_rep="X_scGPT")

        for res in resolutions:
            sc.tl.louvain(adata_t, resolution=res, key_added=cluster_key)
            clustering = adata_t.obs[cluster_key]
            adata_t.obs[f"resolution_{res}"] = clustering
            del adata_t.obs[cluster_key]

        adata_t.write_h5ad(save_dir / "adata_with_cluster.h5ad")

    if len(include_types) == 1:
        return results


best_val_loss = float("inf")
best_model = None

for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    mask_ratio = np.random.choice(config.mlm_probability)
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample, mask_ratio=mask_ratio)

    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

    if config.do_train:
        train(
            model,
            loader=train_loader,
        )
    val_loss, val_mre = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        logger.info(f"Saving model to {save_dir}")
        torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

    if epoch == config.epochs:
        # eval on testdata
        results = eval_testdata(
            best_model,
            adata_t=adata,
            include_types=["cls"],
            best_model_epoch=best_model_epoch,
            epoch=epoch
        )

    scheduler.step()

# save the best model
torch.save(best_model.state_dict(), save_dir / "best_model.pt")
