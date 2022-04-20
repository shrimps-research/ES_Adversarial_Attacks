#!/bin/bash

python attack_main.py \
-model xception_classifier \
-eval crossentropy_similarity \
-in ../data/img_data/xcept_299/tench.png \
-tl 0 \
-min \
-b 6000 \
-ps 8 -os 56 \
-d 1 \
-e 0.02 \
-r discrete \
-m individual \
-fp 50 \
-s comma_selection \
-v 2