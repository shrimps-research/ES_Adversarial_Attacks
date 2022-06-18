import numpy as np

with open("test.txt", "r") as res_file:
    noisy_preds = []
    fooled_noisy_pred = 0
    not_fooled_img_names = []
    for run in res_file.read().split("\n\n"):
        res = run.split("\n")
        best_eval = float(res[1].split(" ")[-1])
        og_acc = float(res[2].split(" ")[-1][:-1])
        noisy_acc = float(res[3].split(" ")[-1][:-1])
        if og_acc == 0.0:
            continue
        noisy_preds.append(np.exp(best_eval))
        if noisy_acc == 0.0:
            fooled_noisy_pred += 1
        else:
            not_fooled_img_names.append(res[0][:-1])
    
    # mean, std and median of model predictions on noisy images
    mean_noisy_pred = np.mean(noisy_preds)
    std_noisy_pred = np.std(noisy_preds)
    median_noisy_pred = np.median(noisy_preds)

print(f"Fooled images: {fooled_noisy_pred}/{len(noisy_preds)} ({fooled_noisy_pred/len(noisy_preds)})\n" \
        +f"Mean noisy pred: {mean_noisy_pred} +- {std_noisy_pred}\n" \
        +f"Median noisy pred: {median_noisy_pred}")
# print(f"List of not fooled images:\n{not_fooled_img_names}")