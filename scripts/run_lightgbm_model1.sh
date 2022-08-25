lr_list=(0.05 0.1)
max_depth=6
save_path_prefix='../save_model_ckp/model1_freeze_disease_encoder_0605/'
device='cuda:0'

for step in {1..20};
do
model_path=$save_path_prefix'step_'$step'0000_model.bin'

    for lr in ${lr_list[@]}; 
    do
        num_leaves=$((2**($max_depth-1)))
        python lightgbm_with_bert.py \
                --device $device \
                --model_path $model_path \
                --batch_size 128 \
                --test 1 \
                --num_leaves $num_leaves \
                --lr $lr \
                --max_depth $max_depth
    done

    for lr in ${lr_list[@]}; 
    do
        num_leaves=$((2**($max_depth-1)))
        python lightgbm_with_bert.py \
                --device $device \
                --model_path $model_path \
                --batch_size 128 \
                --num_leaves $num_leaves \
                --lr $lr \
                --max_depth $max_depth
    done
done