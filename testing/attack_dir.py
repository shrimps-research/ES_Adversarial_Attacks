import os
import subprocess
from attack_main import main

src_dir = "../data/img_data/imagenet/komodo"
dst_dir = "../data/img_data/test"
for img_name in os.listdir(src_dir):
    os.replace(f"{src_dir}/{img_name}", f"{dst_dir}/{img_name}")
    attack = subprocess.run(["python", "attack_main.py", "-model", "xception_classifier", "-device", "cuda", "-eval", "crossentropy", "-in", "../data/img_data/test/", "-tl", "48", "-min", "-b", "1000", "-ps", "12", "-os", "50", "-d", "0.9", "-e", "0.05", "-r", "global_discrete", "-m", "individual", "-s", "comma_selection", "-fp", "5", "-v", "2"], capture_output=True, text=True)
    output = attack.stdout[attack.stdout.find("Best function evaluation:"):]
    with open("noisy_acc_komodo_single.txt", "a") as res_file:
        res_file.write(f"{img_name}:\n{output}\n")
    os.replace(f"{dst_dir}/{img_name}", f"{src_dir}/{img_name}")