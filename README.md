# Heterogeneous biomedical entity representation learning for gene-disease association prediction


## Installation



```bash
# Anaconda installing
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
RUN bash Anaconda3-2021.11-Linux-x86_64.sh -b
RUN rm Anaconda3-2021.11-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN conda update conda

# Install pytorch  pytorch-geometric
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
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

We store all required datasets in the Google Drive. [Here](https://drive.google.com/file/d/16O090S73EMqhhGfgwBiuTgz0zX1QY8Es/view?usp=sharing)




