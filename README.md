# Heterogeneous biomedical entity representation learning for gene-disease association prediction

<div align="left">

[![Paper](https://img.shields.io/badge/Briefings%20in%20Bioinformatics-2406.01651-B31B1B.svg)](https://academic.oup.com/bib/article/25/5/bbae380/7735275)
[![Demo](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/Gla-AI4BioMed-Lab/FusionGDA)

</div>

<div style="display: flex; align-items: center;">
    <img src="Figure/FusionGDA.jpg" alt="FusionGDA" style="max-width: 40%; height: auto;"/>
    <img src="Figure/Fusion_Module.jpg" alt="Fusion Module" style="max-width: 60%; height: auto; margin-left: 20px;"/>
</div>

## Installation

```bash
# Download the latest Anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh

# Install Anaconda
bash Anaconda3-latest-Linux-x86_64.sh -b

# Clean up the installer to save space
rm Anaconda3-latest-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/anaconda3/bin:$PATH

# Updating Anaconda packages
conda update --all

# Install the latest version of PyTorch and related libraries with CUDA support
# Note: Replace 'cudatoolkit=x.x' with the version compatible with your CUDA version
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Install other Python packages
pip install pytdc
pip install wandb
pip install lightgbm
pip install -U adapter-transformers
pip install pytorch-metric-learning
```

## Executing program

Make sure you are in the directory ~/dpa_pretrain/scripts
You adjust the required parameters directly.

###  Pre-training phase
```
bash run_pretrain_gda_ml_adapter_infoNCE.sh
```
###  Fine-tuning phase
TDC Dataset
```
bash run_finetune_gda_lightgbm_infoNCE_tdc.sh
```
DisGeNET Dataset
```
bash run_finetune_gda_lightgbm_infoNCE.sh
```
Check your results in the wandb account.

## Datasets

We store all required datasets in the Google Drive. [Here](https://drive.google.com/file/d/1o4h2Dwfb4DtYgKD2K0hgHneowhE1OYEn/view?usp=share_link)

## Citation
Please cite our [paper](https://arxiv.org/abs/2406.01651) if you find our work useful in your own research.
```
@article{meng2024heterogeneous,
  title={Heterogeneous biomedical entity representation learning for gene-disease association prediction},
  author={Meng, Zhaohan and Liu, Siwei and Liang, Shangsong and Jani, Bhautesh and Meng, Zaiqiao},
  journal={Briefings in Bioinformatics},
  volume={25},
  number={5},
  pages={bbae380},
  year={2024},
  publisher={Oxford University Press}
}
```




