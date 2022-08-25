lr_list=(0.00005 0.00025 0.0001)
device='cuda:0'
for lr in ${lr_list[@]}; 
do
    cd ../src/finetune; python finetune_gda_with_adapter_cls.py \
            --device $device \
            --batch_size 12 \
            --use_adapter \
            --lr $lr

    prot_encoder_path="zjukg/OntoProtein"
    cd ../src/finetune; python finetune_gda_with_adapter_cls.py \
            --device $device \
            --prot_encoder_path $prot_encoder_path \
            --use_adapter \
            --batch_size 12 \
            --lr $lr
done