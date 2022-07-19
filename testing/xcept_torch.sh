#!/bin/bash

python attack_main.py \
-model "xception_torch" \
-device "cuda" \
-batches 128 16 \
-eval "crossentropy" \
-in "../data/temp/" \
-tl 0 \
-min \
-b 3000 \
-ps 12 -os 85 \
-e 0.05 \
-r "global_discrete" \
-m "individual" \
-s "comma_selection" \
-fp 5 \
-v 2