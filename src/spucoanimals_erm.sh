python src/spucoanimals_erm.py --seed=0 --gpu=5 > logs/spucoanimals_erm_0.log &

sleep 10

python src/spucoanimals_erm.py --seed=1 --gpu=6 > logs/spucoanimals_erm_1.log &

sleep 10

python src/spucoanimals_erm.py --seed=2 --gpu=7 > logs/spucoanimals_erm_2.log &