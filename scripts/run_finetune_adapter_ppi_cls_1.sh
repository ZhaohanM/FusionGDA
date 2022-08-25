lr_list=(0.00001 0.00002 0.00005 0.0001 0.0005)
device='cuda'
reduction_factor=8
step=1
batch_size=10

for lr in ${lr_list[@]}; 
do
    model_path='../save_model_ckp/pretrain/ppi_2m_adapter/reduction_factor_'$reduction_factor'/'
    cd ../src/downstream; python finetune_ppi_with_adapter_cls.py \
            --device $device \
            --reduction_factor $reduction_factor \
            --model_path $model_path \
            --batch_size $batch_size \
            --test \
            --patience 5 \
            --step $step'0000' \
            --lr $lr
done
