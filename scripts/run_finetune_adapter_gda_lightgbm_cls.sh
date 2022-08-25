lr_list=(0.05 0.075 0.1 0.15)
max_depth=6
device='cuda:1'
reduction_factor=8

for step in {1..4};
do
    for lr in ${lr_list[@]}; 
    do
        model_path='../../save_model_ckp/pretrain/gda_adapter/reduction_factor_'$reduction_factor'/'
        num_leaves=$((2**($max_depth-1)))
        cd ../src/finetune/; python finetune_gda_with_adapter_lightgbm_cls.py \
                --device $device \
                --reduction_factor $reduction_factor \
                --model_path $model_path \
                --batch_size 256 \
                --step $step'00' \
                --lr $lr \
                --num_leaves $num_leaves \
                --max_depth $max_depth

        cd ../src/finetune/; python finetune_gda_with_adapter_lightgbm_cls.py \
                --device $device \
                --reduction_factor $reduction_factor \
                --model_path $model_path \
                --batch_size 256 \
                --step $step'00' \
                --lr $lr \
                --use_both_feature \
                --num_leaves $num_leaves \
                --max_depth $max_depth
    done
done

