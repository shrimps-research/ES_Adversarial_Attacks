#!/bin/bash

python attack_main.py \
-model xception_classifier \
-eval  crossentropy_similarity \
-in ../data/img_data/xcept_299/tench.png \
-tl 0 \
-min \
-b 10000 \
-ps 6 -os 36 \
-d 1 \
-e 0.02 \
-r discrete \
-m individual \
-fp 5 \
-s comma_selection \
-v 2
