python src/waterbirds_group_balance.py --seed=0 --gpu=7 --lr=1e-4 --weight_decay=1e-4 > logs/waterbirds_group_balance_0.log &

sleep 10

python src/waterbirds_group_balance.py --seed=1 --gpu=5 --lr=1e-4 --weight_decay=1e-4 > logs/waterbirds_group_balance_1.log &

sleep 10

python src/waterbirds_group_balance.py --seed=2 --gpu=6 --lr=1e-4 --weight_decay=1e-4 > logs/waterbirds_group_balance_2.log &