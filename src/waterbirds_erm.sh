# start time
start_time=$(date +%s)

python src/waterbirds_erm.py --seed=0 --gpu=4 > logs/waterbirds_erm_time.log 

# print running time in xx hrs xx mins xx secs
echo "Running time: $(date -u --date @$(( $(date +%s) - $start_time )) +%H:%M:%S)"
# sleep 10

# python src/waterbirds_erm.py --seed=1 --gpu=6 > logs/waterbirds_erm_4.log &

# sleep 10

# python src/waterbirds_erm.py --seed=2 --gpu=7 > logs/waterbirds_erm_5.log &

# gpus=(0 1 2 3 4 5 6 7)

# lr_list=(1e-1 1e-2 1e-3)
# # weight_decay_list=(1e-3 1e-4)


# # for the first seed 
# for lr in ${lr_list[@]}; do

#     # if [ ${lr} == 1e-4 ]; then
#     #     weight_decay_list=(1e-2 1e-3)
#     # else
#     #     weight_decay_list=(1e-1 1e-2 1e-3 1e-4)
#     # fi

#     if [ ${lr} == 1e-3 ]; then
#         weight_decay_list=(1e-3)
#     else
#         weight_decay_list=(1e-3 1e-4)
#     fi


#     for weight_decay in ${weight_decay_list[@]}; do

#         # while the job is not running on any gpu
#         while ! ps -ef | grep "python src/waterbirds_erm.py --lr ${lr} --weight_decay ${weight_decay}" | grep -v grep; do

#             # find gpus that have at least 15000 MB free memory at the moment after running the job for 10 seconds 
#             for gpu in ${gpus[@]}; do
#                 if [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk "NR==${gpu}+1" | awk '{print $1}') -gt 15000 ]; then
#                     # try to run the job on this gpu. If it fails, try the next gpu

#                     # if the job is running on this gpu in the background, break the loop
#                     if python src/waterbirds_erm.py --lr ${lr} --weight_decay ${weight_decay} --gpu ${gpu} > logs/waterbirds_erm_lr${lr}_wd${weight_decay}.log & then

#                         # wait for 10 seconds and try again
#                         sleep 100

#                         # if the job is still running, break the loop
#                         break
#                     fi
#                 fi
#             done

#             # if the job is not running on any gpu, wait for 10 seconds and try again
#             sleep 10
#         done

#     done
# done