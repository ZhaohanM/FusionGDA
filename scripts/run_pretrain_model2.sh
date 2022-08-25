save_path_prefix='../save_model_ckp/model2_pretrain_freeze_prot/'
ppi_dir='../data/string_ppi_920_300k.csv'
python pretrain_model2.py \
    --device 'cuda:0' \
    --freeze_prot_encoder \
    --ppi_dir $ppi_dir \
    --save_path_prefix $save_path_prefix \
    --batch_size 24 \
    --save_step 10000