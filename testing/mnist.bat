python attack_main.py ^
-model mnist_classifier ^
-eval classification_crossentropy ^
-in ../data/img_data/mnist/zero.png ^
-tl 0 ^
-min ^
-b 10000 ^
-ps 2 -os 2 ^
-d 1 ^
-e 0.2 ^
-m individual ^
-r intermediate ^
-s plus_selection ^
-v 2