# Heterogeneous biomedical entity representation learning for gene-disease association prediction


## Installation



```bash
pip install pytdc
pip install wandb
pip install lightgbm
pip install -U adapter-transformers
pip install pytorch-metric-learning
```

## Executing program

Make sure you are in the directory ~/dpa_pretrain/scripts
You adjust the required parameters directly.

Pre-training phase
```
bash run_pretrain_gda_ml_adapter_infoNCE.sh
```
Fine-tuning phase
```
bash run_finetune_gda_lightgbm_infoNCE.sh
```
Check your results in the wandb account.

## Datasets
We store all required datasets in the Google Drive. [Here](https://drive.google.com/file/d/16O090S73EMqhhGfgwBiuTgz0zX1QY8Es/view?usp=share_link)



