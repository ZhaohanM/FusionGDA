save_path_prefix='../save_model_ckp/pretrain/epoch_2/'
for i in 60 75 90;
do
    step=$save_path_prefix'modelstep_'$i'000_model.bin'
    echo $step
    python finetune_dpa.py --model_path $step --device 'cuda:7'
done
