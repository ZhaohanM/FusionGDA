lr=0.000003
device='cuda:0'
reduction_factor=8
batch_size=5
model_path='../../save_model_ckp/pretrain/ppi_2m_adapter/reduction_factor_'$reduction_factor'/'
prot_encoder_path="Rostlab/prot_bert"
for step in {1..8};
do
    cd ../src/finetune; python finetune_ppi_with_adapter_cls.py \
            --prot_encoder_path $prot_encoder_path \
            --device $device \
            --reduction_factor $reduction_factor \
            --model_path $model_path \
            --batch_size $batch_size \
            --patience 8 \
            --use_adapter \
            --step $step'00' \
            --lr $lr
done

for step in {1..20};
do
    cd ../src/finetune; python finetune_ppi_with_adapter_cls.py \
            --prot_encoder_path $prot_encoder_path \
            --device $device \
            --reduction_factor $reduction_factor \
            --model_path $model_path \
            --batch_size $batch_size \
            --patience 8 \
            --use_adapter \
            --step $step'0000' \
            --lr $lr
done

