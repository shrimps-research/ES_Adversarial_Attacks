#!/bin/bash

python attack_main.py \
-model xception_classifier \
-eval classification_crossentropy \
-in ../data/img_data/xcept_299/tench.png \
-tl 0 \
-min \
-b 10000 \
-ps 6 -os 36 \
-d 1 \
-e 0.02 \
-r discrete \
-m one_fifth \
-s plus_selection \
-v 2