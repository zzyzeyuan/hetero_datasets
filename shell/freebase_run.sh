#!/bin/bash
num_gnns="3 7"
num_layers="1 2 3"
dropout="0 0.2 0.4 0.5"
temperature="0.1 2.0"
num_heads="1 2"
for l in $num_layers; do
    for gnn in $num_gnns; do
        for h in $num_heads; do
            for dr in $dropout; do
                for t in $temperature; do
                    python run.py --dataset Freebase --feats-type 2 --hidden-dim 512 --num-heads $h \
                            --ffn-dim 64 --num-gnns $gnn --num-layers $l --dropout $dr --attn-dropout $dr \
                            --temperature $t --device 2 --cut 4 --seed 1 
                    done
                done
            done
        done
    done

echo "Finished.========"