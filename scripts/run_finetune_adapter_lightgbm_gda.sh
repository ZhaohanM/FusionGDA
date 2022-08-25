lr_list=(0.01 0.05 0.1)
max_depth=6
device='cuda:1'
reduction_factor=8

for step in {1..20};
do
    for lr in ${lr_list[@]};
    do
        model_path='../save_model_ckp/pretrain/gda_adapter/reduction_factor_'$reduction_factor'/'
        num_leaves=$((2**($max_depth-1)))
        cd ../src/; python fine-tune-adapter-with-lightgbm.py \
                --device $device \
                --reduction_factor $reduction_factor \
                --model_path $model_path \
                --batch_size 256 \
                --test 1 \
                --step $step'0000' \
                --num_leaves $num_leaves \
                --lr $lr \
                --max_depth $max_depth
    done

    for lr in ${lr_list[@]}; 
    do
        model_path='../save_model_ckp/pretrain/dda_adapter/reduction_factor_'$reduction_factor'/'
        num_leaves=$((2**($max_depth-1)))
        cd ../src/; python fine-tune-adapter-with-lightgbm.py \
                --device $device \
                --reduction_factor $reduction_factor \
                --model_path $model_path \
                --batch_size 256 \
                --test 1 \
                --step $step'0000' \
                --num_leaves $num_leaves \
                --lr $lr \
                --max_depth $max_depth
    done
    for lr in ${lr_list[@]}; 
    do
        model_path='../save_model_ckp/pretrain/ppi_adapter/reduction_factor_'$reduction_factor'/'
        num_leaves=$((2**($max_depth-1)))
        cd ../src/; python fine-tune-adapter-with-lightgbm.py \
                --device $device \
                --reduction_factor $reduction_factor \
                --model_path $model_path \
                --batch_size 256 \
                --test 1 \
                --step $step'0000' \
                --num_leaves $num_leaves \
                --lr $lr \
                --max_depth $max_depth
    done
done

for step in {1..20};
do
    for lr in ${lr_list[@]}; 
    do
        model_path='../save_model_ckp/pretrain/gda_adapter/reduction_factor_'$reduction_factor'/'
        num_leaves=$((2**($max_depth-1)))
        cd ../src/; python fine-tune-adapter-with-lightgbm.py \
                --device $device \
                --reduction_factor $reduction_factor \
                --model_path $model_path \
                --batch_size 256 \
                --step $step'0000' \
                --num_leaves $num_leaves \
                --lr $lr \
                --max_depth $max_depth
    done

    for lr in ${lr_list[@]}; 
    do
        model_path='../save_model_ckp/pretrain/dda_adapter/reduction_factor_'$reduction_factor'/'
        num_leaves=$((2**($max_depth-1)))
        cd ../src/; python fine-tune-adapter-with-lightgbm.py \
                --device $device \
                --reduction_factor $reduction_factor \
                --model_path $model_path \
                --batch_size 256 \
                --step $step'0000' \
                --num_leaves $num_leaves \
                --lr $lr \
                --max_depth $max_depth
    done
    for lr in ${lr_list[@]}; 
    do
        model_path='../save_model_ckp/pretrain/ppi_adapter/reduction_factor_'$reduction_factor'/'
        num_leaves=$((2**($max_depth-1)))
        cd ../src/; python fine-tune-adapter-with-lightgbm.py \
                --device $device \
                --reduction_factor $reduction_factor \
                --model_path $model_path \
                --batch_size 256 \
                --step $step'0000' \
                --num_leaves $num_leaves \
                --lr $lr \
                --max_depth $max_depth
    done
done