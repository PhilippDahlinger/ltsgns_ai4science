python main.py configs/exp_2/deformable_plate/lts_gns_prodmp.yml -o -s
sleep 10s
python main.py configs/exp_2/deformable_plate/lts_gns_prodmp_big_collider_radius.yml -o -s
sleep 10s
python main.py configs/exp_2/deformable_plate/lts_gns_prodmp_more_blocks.yml -o -s
sleep 10s
#python main.py configs/exp_2/deformable_plate/mgn.yml -o -s
#sleep 10s
#python main.py configs/exp_2/deformable_plate/mgn_poisson_decoder.yml -o -s
#sleep 10s
python main.py configs/exp_2/deformable_plate/mgn_prodmp.yml -o -s

