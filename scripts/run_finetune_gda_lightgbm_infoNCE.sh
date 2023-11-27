lr_list=(0.15 0.2 0.25 0.3 0.35)
max_depth=6
device='cuda:0'
reduction_factor=8
model_path_list=( 
    "../../save_model_ckp/pretrain/gda_infoNCE_fusion_esm" \
)
cd ../src/finetune/
for save_model_path in ${model_path_list[@]};    
    do
    for lr in ${lr_list[@]}; 
        do   
        for step in {1..27};
            do
            num_leaves=$((2**($max_depth-1)))
            python finetune_gda_with_adapter_lightgbm_infonce.py \
                    --save_model_path $save_model_path \
                    --device $device \
                    --batch_size 256 \
                    --step $step"00" \
                    --use_pooled \
                    --lr $lr \
                    --num_leaves $num_leaves \
                    --max_depth $max_depth
            done
        done
    done

