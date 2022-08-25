save_path_prefix='../save_model_ckp/pretrain/epoch_1/'
for i in 15 30 45;
do
    step=$save_path_prefix'modelstep_'$i'000_model.bin'
    echo $step
    python finetune_dpa.py --model_path $step --device 'cuda:2'
done
