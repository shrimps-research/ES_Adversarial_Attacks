#!/bin/bash

python attack_main.py \
-model "xception_classifier" \
-device "cuda" \
-dataloader \
-batches 64 16 \
-eval "crossentropy" \
-in "../data/dragon/" \
-tl 48 \
-min \
-b 100 \
-ps 8 -os 50 \
-d 1 \
-e 0.01 \
-r "global_discrete" \
-m "individual" \
-s "comma_selection" \
-fp 5 \
-v 2