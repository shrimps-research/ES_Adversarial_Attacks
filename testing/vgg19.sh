#!/bin/bash

python attack_main.py \
-model "vgg_classifier" \
-device "cuda" \
-batches 2000 32 \
-eval "crossentropy" \
-in "../data/temp/" \
-tl 0 \
-min \
-b 2000 \
-ps 8 -os 50 \
-e 0.01 \
-r "global_discrete" \
-m "individual" \
-s "comma_selection" \
-fp 5 \
-v 2