lr_list=(0.00001)
device='cuda:0'
reduction_factor=8
step=1
batch_size=32
disease_encoder_path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
model_path='../../save_model_ckp/pretrain/dda_adapter/reduction_factor_'$reduction_factor'/'
for step in {1..34};
do
    for lr in ${lr_list[@]}; 
    do
        cd ../src/finetune; python finetune_dda_with_adapter_cls.py \
                --model_path $model_path \
                --disease_encoder_path $disease_encoder_path \
                --device $device \
                --reduction_factor $reduction_factor \
                --batch_size $batch_size \
                --patience 8 \
                --use_adapter \
                --step $step'00' \
                --lr $lr
    done
done