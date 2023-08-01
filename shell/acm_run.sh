#!/bin/bash
hidden="256 512"
num_gnns="3 6 9"
num_layers="1"
num_heads="1 2"
cut="2 4"
dropout="0 0.1 0.2 0.5"
# temperature="0.1 0.2 0.5 1.0 2.0"
for hi in $hidden;do
    for l in $num_layers; do
        for h in $num_heads; do
            for g in $num_gnns; do
                for c in $cut;do
                    for dr in $dropout;do
                        python run.py --dataset ACM --feats-type 2 \
                                --hidden-dim $hi \
                                --ffn-dim 64 \
                                --num-gnns $g \
                                --num-layers $l \
                                --num-heads $h \
                                --dropout $dr \
                                --attn-dropout $dr \
                                --temperature 2.0 \
                                --device 3 \
                                --cut $c \
                                --seed 1 
                        done
                    done
                done
            done
        done
    done

echo "======================================================"
echo "Find best num-gnns, num-layers, num-heads, Finished."

