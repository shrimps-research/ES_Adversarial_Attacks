#!/bin/bash

python attack_main.py \
-model xception_classifier \
-eval classification_crossentropy \
-in ../data/img_data/xcept_299/tench.png \
-tl 0 \
-min \
-b 5000 \
-ps 4 -os 24 \
-d 0.7 \
-e 0.04 \
-r discrete \
-m one_fifth \
-s plus_selection \
-v 2