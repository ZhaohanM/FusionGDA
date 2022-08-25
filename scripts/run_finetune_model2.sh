#!/bin/bash


lr_list=(0.05 0.1)
max_depth=6
save_path_prefix='../save_model_ckp/model2_pretrain_freeze_prot/'
device='cuda:0'

for step in {10..40};
do
model_path=$save_path_prefix'model2_step_'$step'0000.bin'

    for lr in ${lr_list[@]}; 
    do
        num_leaves=$((2**($max_depth-1)))
        cd ../src/;python lightgbm_with_bert.py \
                --device $device \
                --model_path $model_path \
                --batch_size 128 \
                --test 1 \
                --num_leaves $num_leaves \
                --lr $lr \
                --max_depth $max_depth
    done

done