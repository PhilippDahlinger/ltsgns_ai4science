python main.py configs/exp_2/tissue_manipulation/lts_gns_prodmp.yml -o -s
sleep 10s
python main.py configs/exp_2/tissue_manipulation/lts_gns_prodmp_more_blocks.yml -o -s
sleep 10s
python main.py configs/exp_2/tissue_manipulation/mgn.yml -o -s
sleep 10s
python main.py configs/exp_2/tissue_manipulation/mgn_poisson_decoder.yml -o -s
sleep 10s
python main.py configs/exp_2/tissue_manipulation/mgn_prodmp.yml -o -s

