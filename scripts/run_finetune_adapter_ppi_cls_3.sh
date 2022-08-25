lr_list=(0.00001 0.00005)
device='cuda:0'
reduction_factor=8

for step in {1..20};
do
    for lr in ${lr_list[@]}; 
    do
        model_path='../save_model_ckp/pretrain/ppi_2m_adapter/reduction_factor_'$reduction_factor'/'
        cd ../src/; python finetune_ppi_with_adapter_cls.py \
                --device $device \
                --reduction_factor $reduction_factor \
                --model_path $model_path \
                --batch_size 4 \
                --step $step'0000' \
                --lr $lr
    done
done