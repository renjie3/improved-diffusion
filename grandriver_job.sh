JOB_ID=`cat job_id.log`
echo $JOB_ID
NEXT_JOB_ID=`expr $JOB_ID + 1`
echo $NEXT_JOB_ID > job_id.log

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.3 --learn_sigma True"

DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"

TRAIN_FLAGS="--save_interval 10000 --lr 1e-4 --batch_size 256 --stop_steps 125000 --microbatch -1 --class_cond False --num_workers 4"

ADV_FLAGS="--mode train --output_index True --output_class True --adv_noise_num 5000 --load_model False --model_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/10/model102400.pt --adv_step 30 --save_forward_clean_sample False --single_target_image_id 10002 --adv_loss_type forward_bachword_loss"

POISON_FLAGS="--poisoned True --poisoned_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/170/adv_noise"

# SAMPLE_FLAGS="--batch_size 4 --num_samples 4 --model_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/66247912_1/ema_0.9999_300000.pt"
SAMPLE_FLAGS="--batch_size 100 --num_samples 10000 --model_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/196/ema_0.9999_080000.pt --out_dir /egr/research-dselab/renjie3/renjie/improved-diffusion/results/196/ema_0.9999_080000"

CLASSIFIER_FLAGS="--iterations 200000 --anneal_lr True --random_padding_crop True --batch_size 256 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 --image_size 32 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --poisoned False --poison_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/EMN_cifar10_c2.pt"

CLASSIFIER_SAMPLE_FLAGS="--batch_size 25  --classifier_depth 2 --num_samples 25 --classifier_width 128 --classifier_scale 10.0 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --model_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/official_checkpoint/cifar10_uncond_50M_500K.pt --classifier_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/176/model199999.pt --out_dir /egr/research-dselab/renjie3/renjie/improved-diffusion/results/official_checkpoint/cifar10_uncond_50M_500K"

GPU_ID='0'
# MY_CMD="mpiexec -n 1 python -u scripts/image_train.py --data_dir /egr/research-dselab/renjie3/renjie/improved-diffusion/datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $ADV_FLAGS $POISON_FLAGS"
# MY_CMD="python -u scripts/classifier_train.py --data_dir /localscratch/renjie/cifar_train --val_data_dir /localscratch/renjie/cifar_train $DIFFUSION_FLAGS $CLASSIFIER_FLAGS"
# MY_CMD="python -u scripts/classifier_sample.py $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_SAMPLE_FLAGS"
MY_CMD="python scripts/image_sample.py $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS"
# MY_CMD="python -u scripts/classifier_unlearnable.py --data_dir /localscratch/renjie/cifar_train --val_data_dir /localscratch/renjie/cifar_train --adv_step 10 --adv_epsilon 0.1256 --adv_alpha 0.01256 $DIFFUSION_FLAGS $CLASSIFIER_FLAGS"
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
