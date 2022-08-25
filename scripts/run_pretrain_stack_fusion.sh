lr_list=(0.0001)
device='cuda:1'
reduction_factor=8
step=1
batch_size=6
disease_encoder_path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
prot_encoder_path="Rostlab/prot_bert"
disease_model_path='../../save_model_ckp/pretrain/dda_adapter/reduction_factor_'$reduction_factor'/'
disease_model_step=50000
prot_model_path='../../save_model_ckp/pretrain/ppi_2m_adapter/reduction_factor_'$reduction_factor'/'
prot_model_step=40000
for lr in ${lr_list[@]}; 
do
    cd ../src/pretrain; python pretrain_gda_adapter_fusion.py \
            --prot_encoder_path $prot_encoder_path \
            --disease_encoder_path $disease_encoder_path \
            --prot_model_path $prot_model_path \
            --disease_model_path $disease_model_path \
            --prot_model_step $prot_model_step \
            --disease_model_step $disease_model_step \
            --device $device \
            --loss ms_loss \
            --reduction_factor $reduction_factor \
            --batch_size $batch_size \
            --use_stack \
            --lr $lr
done