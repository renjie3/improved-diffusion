#!/bin/bash
cd `dirname $0`
MY_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $MY_JOB_ROOT_PATH

MYTIME="3:59:00"
MYNTASKS="1"
MYCPU="4"
MYGRES="gpu:v100s:1"

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --dropout 0.3 --learn_sigma False"

DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"

TRAIN_FLAGS="--save_interval 10000 --lr 1e-4 --batch_size 128 --stop_steps 300000 --microbatch -1 --class_cond False --num_workers 4"

ADV_FLAGS="--mode train --output_index True --output_class True --adv_noise_num 5000 --load_model False --model_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/10/ema_0.9999_102400.pt --adv_step 30 --save_forward_clean_sample False --single_target_image_id 10002"

POISON_FLAGS="--poisoned False --poisoned_path /egr/research-dselab/renjie3/renjie/improved-diffusion/results/19/adv_noise"

SAMPLE_FLAGS="--batch_size 128 --num_samples 1000 --model_path /mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/results/127/model045000.pt --out_dir /mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/results/127/model045000"

CLASSIFIER_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --lr 3e-4 --save_interval 10000 --weight_decay 0.05 --image_size 32 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"

JOB_INFO="train stepsize"
# MYCOMMEND="mpiexec -n 4 python -u scripts/image_train.py --data_dir datasets/cifar_train_sky_blue $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $ADV_FLAGS $POISON_FLAGS"
# MYCOMMEND="python scripts/image_sample.py $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS"
MYCOMMEND="python -u scripts/classifier_train.py --data_dir /mnt/home/renjie3/Documents/unlearnable/diffusion/improved-diffusion/datasets/cifar_train $CLASSIFIER_FLAGS"

MYCOMMEND2="python3 test.py -gpu_id 0 -model 1 -attack 1 --pgd_norm 7 -batch_size 50 -path Final/VANILLA_62162198_1/iter_50 --alpha 1000 --num_iter 100 --num_stop 2000 --test_subset --seed 1"

MYCOMMEND3="python3 test.py -gpu_id 0 -model 1 -attack 1 --pgd_norm 7 -batch_size 50 -path Final/VANILLA_62162198_1/iter_50 --alpha 5000 --num_iter 100 --num_stop 2000 --test_subset --seed 1"

MYCOMMEND2="No_commend2"
MYCOMMEND3="No_commend3"

cat ./slurm_files/sconfigs1_cmse.sb > submit.sb
# cat ./slurm_files/sconfigs1_scavenger.sb > submit.sb
# cat ./slurm_files/sconfigs1.sb > submit.sb
echo "#SBATCH --time=${MYTIME}             # limit of wall clock time - how long the job will run (same as -t)" >> submit.sb
echo "#SBATCH --cpus-per-task=${MYCPU}           # number of CPUs (or cores) per task (same as -c)" >> submit.sb
echo "#SBATCH --ntasks=${MYNTASKS}                  # number of tasks - how many tasks (nodes) that you require (same as -n)" >> submit.sb
echo "#SBATCH --gres=${MYGRES}" >> submit.sb
# echo "#SBATCH --nodelist=nvl-001" >> submit.sb
echo "#SBATCH -o ${MY_JOB_ROOT_PATH}/logfile/%j.log" >> submit.sb
echo "#SBATCH -e ${MY_JOB_ROOT_PATH}/logfile/%j.err" >> submit.sb
cat ./slurm_files/sconfigs2.sb >> submit.sb
echo "JOB_INFO=\"${JOB_INFO}\"" >> submit.sb
echo "MYCOMMEND=\"${MYCOMMEND} --job_id \${SLURM_JOB_ID}_1\"" >> submit.sb
echo "MYCOMMEND2=\"${MYCOMMEND2} --job_id \${SLURM_JOB_ID}_2\"" >> submit.sb
echo "MYCOMMEND3=\"${MYCOMMEND3} --job_id \${SLURM_JOB_ID}_3\"" >> submit.sb
cat ./slurm_files/sconfigs3.sb >> submit.sb
MY_RETURN=`sbatch submit.sb`

echo $MY_RETURN

MY_SLURM_JOB_ID=`echo $MY_RETURN | awk '{print $4}'`

#print the information of a job into one file
date >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MY_SLURM_JOB_ID >>${MY_JOB_ROOT_PATH}/history_job.log
echo $JOB_INFO >>${MY_JOB_ROOT_PATH}/history_job.log
echo "#SBATCH --time=${MYTIME}" >>${MY_JOB_ROOT_PATH}/history_job.log
echo "#SBATCH --cpus-per-task=${MYCPU}" >>${MY_JOB_ROOT_PATH}/history_job.log
echo "#SBATCH --gres=${MYGRES}" >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MYCOMMEND >>${MY_JOB_ROOT_PATH}/history_job.log
if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
then
    echo $MYCOMMEND2 >>${MY_JOB_ROOT_PATH}/history_job.log
fi
if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
then
    echo $MYCOMMEND3 >>${MY_JOB_ROOT_PATH}/history_job.log
fi
echo "---------------------------------------------------------------" >>${MY_JOB_ROOT_PATH}/history_job.log
