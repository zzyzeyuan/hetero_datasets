#!/bin/bash

# Define parameter values
hidden="128 256 512"
ffn_dim="64 128"
dropout="0 0.2 0.3"
num_heads="1 2"
attn_dropout="0 0.2 0.3"
temperature="0.1 1.0 2.0"
num_layers="1 2"
num_gnns="2 10"

# Loop through all parameter combinations
for h in $hidden; do
    for f in $ffn_dim; do
        for d in $dropout; do
            for n_h in $num_heads; do
                for a_d in $attn_dropout; do
                    for t in $temperature; do
                        for n_l in $num_layers; do
                            for n_g in $num_gnns; do
                                # Execute Python script with parameter values
                                python run.py --dataset DBLP --feats-type 2 --lr 0.0001 --seed 816 --hidden $h --ffn-dim $f --dropout $d \
                                        --num-heads $n_h --attn-dropout $a_d --temperature $t \
                                        --num-layers $n_l --num-gnns $n_g --device 3
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Shell finished!"