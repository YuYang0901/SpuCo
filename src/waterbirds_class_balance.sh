python src/waterbirds_class_balance.py --seed=0 --gpu=2 > logs/waterbirds_class_balance_0.log &

sleep 10

python src/waterbirds_class_balance.py --seed=1 --gpu=3 > logs/waterbirds_class_balance_1.log &

sleep 10

python src/waterbirds_class_balance.py --seed=2 --gpu=7 > logs/waterbirds_class_balance_2.log &