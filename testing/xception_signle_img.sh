#!/bin/bash

python attack_main.py \
-model xception_classifier \
-device "cuda" \
-batches 2000 128 \
-eval "blind_evaluation" \
-in "../data/temp/" \
-tl 48 \
-min \
-b 1000 \
-ps 8 -os 50 \
-d 1 \
-e 0.01 \
-r "global_discrete" \
-m "individual" \
-s "comma_selection" \
-fp 5 \
-v 2