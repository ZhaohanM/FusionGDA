lr_list=(0.00001)
device='cuda:1'
reduction_factor=8

for step in {1..3};
do
    for lr in ${lr_list[@]}; 
    do
        model_path='../save_model_ckp/pretrain/gda_adapter/reduction_factor_'$reduction_factor'/'
        cd ../src/finetune; python finetune_gda_with_adapter_cls.py \
                --device $device \
                --reduction_factor $reduction_factor \
                --model_path $model_path \
                --batch_size 6 \
                --step $step'0000' \
                --lr $lr
    done
done