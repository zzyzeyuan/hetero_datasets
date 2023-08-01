#!/bin/bash
num_gnns="5 7"
dropout="0 0.1 0.2 0.5"
lr="0.0001 0.00001"
num_layers="1 2 3"
cut="1 2 4"
# attndropout="0 0.1 0.2 0.5"
# temperature="0.1 0.2 0.5 1.0 2.0"
# num_heads="1 2"
for c in $cut; do
    for l in $lr; do
        for la in $num_layers; do
            for dr in $dropout; do
                for g in $num_gnns; do
                    python run_multi.py --dataset IMDB --feats-type 0 --hidden-dim 512 \
                            --ffn-dim 64 --lr $l --seed 2 --dropout $dr --attn-dropout $dr --num-heads 2 \
                            --temperature 2.0 --device 4 --num-gnns $g --num-layers $la --repeat 5 --cut $c
                    done
                done
            done
        done
    done 


echo "Parameters search finished !"