JOB_ID=`cat job_id.log`
echo $JOB_ID
NEXT_JOB_ID=`expr $JOB_ID + 1`
echo $NEXT_JOB_ID > job_id.log

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.3 --learn_sigma False --num_input_channels 3"

DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --predict_xstart True"

TRAIN_FLAGS="--save_interval 10000 --lr 1e-4 --batch_size 100 --lr_anneal_steps 0 --stop_steps 10000 --microbatch -1 --class_cond False --save_early_model True --load_model False --model_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/80/model000100.pt"

ADV_FLAGS="--mode train --poison_mode gradient_matching --output_index True --output_class True --adv_noise_num 5000 --adv_step 100 --save_forward_clean_sample False --single_target_image_id 5000 --adv_loss_type test_t_emb_emb_loss --group_model_dir /egr/research-dselab/renjie3/renjie/improved-diffusion/results/58 --group_model False --group_model_num 6 --random_noise_every_adv_step False --t_seg_num 8 --t_seg_start 0 --t_seg_end 4 --eot_gaussian_num 2"

GM_FLAGS="--source_dir /localscratch/renjie/cifar_train_5000_red_bird --source_clean_dir /localscratch/renjie/cifar_train_5000_bird --source_class 0 --one_class_image_num 5000 --optim_mode pgd --debug False"

POISON_FLAGS="--poisoned False --poisoned_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/61/adv_noise"


# model102400.pt ema_0.9999_102400.pt


GPU_ID='1'
MY_CMD="mpiexec -n 1 python -u scripts/image_train.py --data_dir /localscratch/renjie/cifar_train_3class_9000 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $ADV_FLAGS $GM_FLAGS $POISON_FLAGS"
MY_ROOT_PATH=`pwd`

echo "cd ${MY_ROOT_PATH}" > ./cmd/cmd_${JOB_ID}.sh
echo "MY_CMD=\"${MY_CMD} --job_id $JOB_ID \"" >> ./cmd/cmd_${JOB_ID}.sh
echo "CUDA_VISIBLE_DEVICES='${GPU_ID}' \${MY_CMD}" >> ./cmd/cmd_${JOB_ID}.sh
echo "\nif [ \$? -eq 0 ];then" >> ./cmd/cmd_${JOB_ID}.sh
echo "echo -e \"grandriver JobID:${JOB_ID} \\\n Python_command: \\\n ${MY_CMD} \\\n \" | mail -s \"[Done] grandriver ${SLURM_JOB_ID}\" thurenjie@outlook.com" >> ./cmd/cmd_${JOB_ID}.sh
echo "else" >> ./cmd/cmd_${JOB_ID}.sh
echo "echo -e \"grandriver JobID:${JOB_ID} \\\n Python_command: \\\n ${MY_CMD} \\\n \" | mail -s \"[Fail] grandriver ${SLURM_JOB_ID}\" thurenjie@outlook.com" >> ./cmd/cmd_${JOB_ID}.sh
echo "fi" >> ./cmd/cmd_${JOB_ID}.sh

nohup sh ./cmd/cmd_${JOB_ID}.sh >./logfile/${JOB_ID}.log 2>./logfile/${JOB_ID}.err &

echo $MY_CMD

date >>./history_job.log
echo ${JOB_ID}>>./history_job.log
echo "GPU_ID=${GPU_ID}">>./history_job.log
echo ${MY_CMD}>>./history_job.log
echo "---------------------------------------------------------------" >>./history_job.log
