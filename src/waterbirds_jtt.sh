python src/waterbirds_jtt.py --seed=0 --gpu=2 --lr=1e-4 --weight_decay=1e-4 > logs/waterbirds_jtt_3.log &

sleep 10

python src/waterbirds_jtt.py --seed=1 --gpu=3 --lr=1e-4 --weight_decay=1e-4 > logs/waterbirds_jtt_4.log &

sleep 10

python src/waterbirds_jtt.py --seed=2 --gpu=4 --lr=1e-4 --weight_decay=1e-4 > logs/waterbirds_jtt_5.log &