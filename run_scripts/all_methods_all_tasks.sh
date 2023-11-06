python main.py configs/mgn_training/exp_1.yml -o -s
sleep 1m
python main.py configs/constant_posterior_training/exp_1.yml -o -s
sleep 1m
python main.py configs/cnp_training/exp_1.yml -o -s
sleep 1m
python main.py configs/task_properties_training/exp_1.yml -o -s
sleep 1m
python main.py configs/lts_gns_training/exp_1.yml -o -s

