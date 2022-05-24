import tensorflow as tf
from PIL import Image
import numpy as np
import argparse
import time


def main():
    # parse cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', action='store', 
                        dest='repetitions', type=int,
                        default=100)
    parser.add_argument('-bs', action='store', 
                        dest='batch_size', type=int,
                        default=64)
    parser.add_argument('-e', action='store', 
                        dest='epsilon', type=float,
                        default=0.5)
    parser.add_argument('-m', action='store', 
                        dest='model', type=str,
                        default="xception")
    parser.add_argument('-img', action='store', 
                        dest='img_path', type=str,
                        default="data/images/tench.png")                
    args = parser.parse_args()

    # timer to benchmark runtime
    start_time = time.time()

    # define parameters
    repetitions = args.repetitions
    batch_size = args.batch_size
    total_experiments = repetitions * batch_size
    epsilon = args.epsilon

    # define model
    if args.model == "xception":
        input_shape = (299,299,3)
        model = tf.keras.applications.Xception( weights='imagenet', 
                                                include_top=True, 
                                                input_shape=input_shape,
                                                classifier_activation='softmax'
                                                )
    else:
        print("Please select a valid model")
        exit()

    # open input image
    img = Image.open(args.img_path)
    img_class_idx = 0

    # prepreocess input depending on model
    if args.model == "xception":
        img = img.resize(input_shape[:2])
        img_arr = np.array(img) / 255.

    # original prediction check
    pred_orig = np.array(model(np.expand_dims(img_arr, axis=0)))
    print(f"Initial prediction: {pred_orig.argmax(axis=1)[0]}, \
    correct label: {img_class_idx}")

    results = []
    for _ in range(repetitions):
        # create adversarial attack
        atk_batch = np.array([ img_arr + 
                                (np.random.uniform(0,1,size=img_arr.shape) * epsilon)  
                                    for i in range(batch_size)]).clip(0,1)
        # pass batch to model
        preds = np.array(model(atk_batch))
        # get the predicted class
        preds_argmax = preds.argmax(axis=1)
        results.append(preds_argmax)
    # flatten results array to 1 dimension
    results = np.reshape(results,(-1))

    # calculate and print runtime
    end_time = time.time()
    tot_secs = np.round(end_time-start_time,2)
    print(f"runtime: {tot_secs}")

    # count wrong predictions
    n_wrong_preds = results[results != img_class_idx].size

    # print results
    print(f"epsilon value: {epsilon}")
    print(f"Succesful attacks: {n_wrong_preds}/{total_experiments}")


if __name__ == "__main__":
    main()