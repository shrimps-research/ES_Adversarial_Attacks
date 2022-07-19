#!/bin/bash

python attack_main.py \
-model "xception_torch" \
-device "cuda" \
-dataloader \
-batches 128 16 \
-eval "crossentropy" \
-in "../data/dragon/" \
-tl 48 \
-min \
-b 3000 \
-ps 12 -os 85 \
-d 1 \
-e 0.05 \
-r "global_discrete" \
-m "individual" \
-s "comma_selection" \
-fp 5 \
-v 2