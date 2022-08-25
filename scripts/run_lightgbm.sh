lr_list=(0.025 0.05 0.1)
max_depth_list=(5 6 7 8 9)

for lr in ${lr_list[@]}; 
do
    for max_depth in ${max_depth_list[@]}; 
    do
        num_leaves=$((2**($max_depth-1)))
        python lightgbm_with_bert.py \
                --device cuda:0 \
                --batch_size 128 \
                --test 1 \
                --num_leaves $num_leaves \
                --lr $lr \
                --max_depth $max_depth
    done
done

for lr in ${lr_list[@]}; 
do
    for max_depth in ${max_depth_list[@]}; 
    do
        num_leaves=$((2**($max_depth-1)))
        python lightgbm_with_bert.py \
                --device cuda:0 \
                --batch_size 128 \
                --num_leaves $num_leaves \
                --lr $lr \
                --max_depth $max_depth
    done
done