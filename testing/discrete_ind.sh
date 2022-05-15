#!/bin/bash

python attack_main.py \
-model xception_classifier \
-eval  crossentropy \
-in ../data/img_data/xcept_299/ \
-tl 0 \
-min \
-b 10000 \
-ps 2 -os 4 \
-d 1 \
-e 0.2 \
-r discrete \
-m individual \
-fp 5 \
-s comma_selection \
-v 2
