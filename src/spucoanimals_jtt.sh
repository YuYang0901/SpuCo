python src/spucoanimals_jtt.py --seed=0 --gpu=2 --lr=1e-3 --weight_decay=1e-4 > logs/spucoanimals_jtt_1e-3_0.log &

sleep 10

python src/spucoanimals_jtt.py --seed=1 --gpu=3 --lr=1e-3 --weight_decay=1e-4 > logs/spucoanimals_jtt_1e-3_1.log &

sleep 10

python src/spucoanimals_jtt.py --seed=2 --gpu=4 --lr=1e-3 --weight_decay=1e-4 > logs/spucoanimals_jtt_1e-3_2.log &