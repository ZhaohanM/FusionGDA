# Gene-Disease Associations Prediction Task Manual


## Data Format

Pretraining data in the directory ~/dpa_pretrain/src/pretrain/data/pretrain/disgnet.csv
Finetuning data is API, you can run it directly.


## Model

### You can check the structure of the model.

 Pretraining model in the directory ~/dpa_pretrain/src/pretrain 

 
 Finetuning model in the directory ~/dpa_pretrain/src/finetune


## Parameters

You can experiment with different parameters to find the best performance. 

### You adjust the Pretraining/Finetuning parameters in this script file.
In the directory ~/dpa_pretrain/src/scripts

### You adjust the loss function parameters in this file.
In the directory ~/dpa_pretrain/src/utils/metric_learning_models.py

## Model Evaluation
ROC-AUC AUPR F1max
