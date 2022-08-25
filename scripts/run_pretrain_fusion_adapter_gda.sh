prot_model_path='../save_model_ckp/pretrain/ppi_2m_adapter/reduction_factor_'$reduction_factor'/'
disease_model_path='../save_model_ckp/pretrain/dda_adapter/reduction_factor_'$reduction_factor'/'
step=6

cd ../src/; python pretrain_gda_adapter_fusion.py \
    --device 'cuda:0' \
    --prot_model_path $prot_model_path \
    --disease_model_path $disease_model_path \
    --lr 0.0001 \
    --use_adapter \
    --batch_size 12 \
    --step $step'0000' \
    --reduction_factor 8 \
    --save_step 10000