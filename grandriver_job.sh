JOB_ID=`cat job_id.log`
echo $JOB_ID
NEXT_JOB_ID=`expr $JOB_ID + 1`
echo $NEXT_JOB_ID > job_id.log

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.3 --learn_sigma False"

DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

TRAIN_FLAGS="--save_interval 1000 --lr 1e-4 --batch_size 256 --stop_steps 200000 --microbatch -1 --class_cond False --num_workers 4"

ADV_FLAGS="--mode train --output_index True --output_class True --adv_noise_num 5000 --load_model False --model_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/10/ema_0.9999_102400.pt --adv_step 30 --save_forward_clean_sample False --single_target_image_id 10002"

POISON_FLAGS="--poisoned False --poisoned_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/19/adv_noise"

# SAMPLE_FLAGS="--batch_size 4 --num_samples 4 --model_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/66247912_1/ema_0.9999_300000.pt"


GPU_ID='2'
MY_CMD="mpiexec -n 1 python -u scripts/image_train.py --data_dir /localscratch/renjie/cifar_train_gray_normal $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $ADV_FLAGS $POISON_FLAGS"
MY_ROOT_PATH=`pwd`

echo "cd ${MY_ROOT_PATH}" > ./cmd/cmd_${JOB_ID}.sh
echo "MY_CMD=\"${MY_CMD} --job_id $JOB_ID \"" >> ./cmd/cmd_${JOB_ID}.sh
echo "CUDA_VISIBLE_DEVICES='${GPU_ID}' \${MY_CMD}" >> ./cmd/cmd_${JOB_ID}.sh
echo "\nif [ \$? -eq 0 ];then" >> ./cmd/cmd_${JOB_ID}.sh
echo "echo -e \"grandriver JobID:${JOB_ID} \\\n Python_command: \\\n ${MY_CMD} \\\n \" | mail -s \"[Done] grandriver ${SLURM_JOB_ID}\" renjie2179@outlook.com" >> ./cmd/cmd_${JOB_ID}.sh
echo "else" >> ./cmd/cmd_${JOB_ID}.sh
echo "echo -e \"grandriver JobID:${JOB_ID} \\\n Python_command: \\\n ${MY_CMD} \\\n \" | mail -s \"[Fail] grandriver ${SLURM_JOB_ID}\" renjie2179@outlook.com" >> ./cmd/cmd_${JOB_ID}.sh
echo "fi" >> ./cmd/cmd_${JOB_ID}.sh

nohup sh ./cmd/cmd_${JOB_ID}.sh >./logfile/${JOB_ID}.log 2>./logfile/${JOB_ID}.err &

echo $MY_CMD

date >>./history_job.log
echo ${JOB_ID}>>./history_job.log
echo "GPU_ID=${GPU_ID}">>./history_job.log
echo ${MY_CMD}>>./history_job.log
echo "---------------------------------------------------------------" >>./history_job.log
