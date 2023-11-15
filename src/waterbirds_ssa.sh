# start time
start_time=$(date +%s)

python src/waterbirds_ssa.py --seed=0 --gpu=4 --pretrained > logs/waterbirds_ssa_time.log

# print running time in xx hrs xx mins xx secs
echo "SSA Running time: $(date -u --date @$(( $(date +%s) - $start_time )) +%H:%M:%S)"

# sleep 10

# python src/waterbirds_ssa.py --seed=1 --gpu=6 --pretrained > logs/waterbirds_ssa_1.log &

# sleep 10

# python src/waterbirds_ssa.py --seed=2 --gpu=7 --pretrained > logs/waterbirds_ssa_2.log &