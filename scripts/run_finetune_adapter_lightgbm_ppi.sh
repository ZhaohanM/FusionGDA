lr_list=(0.0001 0.001 0.05 0.1)
max_depth=6
device='cuda:1'
reduction_factor=8

lr_list=(0.00001)

for step in {1..20};
do
    for lr in ${lr_list[@]}; 
    do
        model_path='../save_model_ckp/pretrain/ppi_2m_adapter/reduction_factor_'$reduction_factor'/'
        cd ../src/; python finetune_gda_with_adapter.py \
                --device $device \
                --reduction_factor $reduction_factor \
                --model_path $model_path \
                --batch_size 6 \
                --step $step'0000' \
                --test \
                --lr $lr
    done
done

for step in {1..20};
do
    for lr in ${lr_list[@]}; 
    do
        model_path='../save_model_ckp/pretrain/ppi_2m_adapter/reduction_factor_'$reduction_factor'/'
        cd ../src/; python finetune_gda_with_adapter.py \
                --device $device \
                --reduction_factor $reduction_factor \
                --model_path $model_path \
                --batch_size 6 \
                --step $step'0000' \
                --lr $lr
    done
done