# This is cut-out from train.py file to only evaluate models


import sys

sys.path.append(
    "C:/Users/masik/Workspace/inzynierka/repos/crossview_localisation_tensorflow-2.x/src/CVM-Net"
)

from cvm_net import cvm_net_I, cvm_net_II
from structures.InputData import InputDataEval
from structures.InputData import InputDataContainer
from MatchesChart import MatchesChart
import math
import csv

import argparse, json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.patches as patches

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def calculate_distance(grd_descriptor, sat_descriptor):
    return 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))


def calculate_accuracy(dist_array, input_data):
    topk_percent = input_data.top_k
    unique_sat_idx_list = input_data.dataset.get_unique_sat_idx_list()
    number_of_unique_sat = len(unique_sat_idx_list)
    images_number_list = topk_percent_to_numbers(number_of_unique_sat)
    topk_given = math.ceil(number_of_unique_sat * topk_percent)

    accuracy_by_percents = np.array([0.0 for _ in images_number_list])
    accuracy_by_top100 = np.array([0.0 for _ in range(100)])
    data_amount = 0.0

    # print("Correct match if it is in top", images_number_list, "images.")

    for j in range(dist_array.shape[1]):
        gt_dist = dist_array[j, j]
        false_prediction = 0

        for i in unique_sat_idx_list:
            if dist_array[i, j] < gt_dist:
                false_prediction += 1

        for idx, images_number in enumerate(images_number_list):
            if false_prediction < images_number:
                if images_number == topk_given:
                    print("Correct match for pano with id:\t", j)
                accuracy_by_percents[idx] += 1.0

        for topk in range(100):
            if false_prediction < topk:
                accuracy_by_top100[topk] += 1.0

        data_amount += 1.0

    accuracy_by_percents /= data_amount
    accuracy_by_top100 /= data_amount
    return accuracy_by_percents, accuracy_by_top100


def topk_percent_to_numbers(number_of_unique_sat):
    topk_percent_list = np.arange(0, 1, 0.01).tolist()
    return [
        math.ceil(number_of_unique_sat * topk_percent)
        for topk_percent in topk_percent_list
    ]


def evaluate(input_data, grd_descriptor, sat_descriptor):
    dist_array = calculate_distance(grd_descriptor, sat_descriptor)
    # accuracy_by_percent, accuracy = calculate_top100_accuracy(
    #     dist_array, topk_percent, unique_sat_idx_list
    # )
    accuracy_by_percent, accuracy = calculate_accuracy(dist_array, input_data)

    # np.savetxt('./Result/sorted_img_idx_by_distances.csv', tmp, fmt="%d", delimiter=',')

    return accuracy_by_percent, accuracy, dist_array


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Arguments for analyzing and displaying results from siamese models."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the config file with other args. It overrides args passed in command line",
    )

    return parser.parse_args()


def prepare_and_run_model(input_data):
    # define placeholders
    model_type = input_data.model
    dataset = input_data.dataset
    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name="sat_x")
    grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name="grd_x")
    keep_prob = tf.placeholder(tf.float32)

    # build model
    if model_type == "CVM-NET-I":
        sat_global, grd_global = cvm_net_I(sat_x, grd_x, keep_prob, False)
    elif model_type == "CVM-NET-II":
        sat_global, grd_global = cvm_net_II(sat_x, grd_x, keep_prob, False)
    else:
        print(
            "CONFIG ERROR: wrong network type, only CVM-NET-I and CVM-NET-II are valid"
        )

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    sat_global_descriptor = None
    grd_global_descriptor = None

    # run model
    print("run model...")
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print("load model...")
        saver.restore(sess, input_data.weights_path)
        print("   Model loaded from: %s" % input_data.weights_path)
        print("load model...FINISHED")

        # ---------------------- validation ----------------------
        print("validate...")
        print("   compute global descriptors")
        dataset.reset_scan()
        sat_global_descriptor = np.zeros([dataset.get_sat_dataset_size(), 4096])
        grd_global_descriptor = np.zeros([dataset.get_pano_dataset_size(), 4096])
        val_i = 0
        while True:
            print("      progress %d" % val_i)
            batch_sat, batch_grd = dataset.next_batch_scan(12)
            if batch_sat is None:
                break
            feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
            sat_global_val, grd_global_val = sess.run(
                [sat_global, grd_global], feed_dict=feed_dict
            )

            sat_global_descriptor[val_i : val_i + sat_global_val.shape[0], :] = (
                sat_global_val
            )
            grd_global_descriptor[val_i : val_i + grd_global_val.shape[0], :] = (
                grd_global_val
            )
            val_i += sat_global_val.shape[0]
    return grd_global_descriptor, sat_global_descriptor


if __name__ == "__main__":
    args = parse_arguments()

    if args.config is not None:
        input_data_container = InputDataContainer(args.config)
    else:
        raise Exception("No config file.")

    for input_data in input_data_container:
        model_name = input_data.name
        results = input_data.results

        grd_global_descriptor, sat_global_descriptor = prepare_and_run_model(input_data)

        results.set_pano_descriptors(grd_global_descriptor)
        results.set_sat_descriptors(sat_global_descriptor)

        # putted here thus it can reduce memory overload during plots generating.
        tf.reset_default_graph()

        print("   compute accuracy")

        topk_percent = input_data.top_k

        accuracy_by_percent_list, accuracy_by_top100_list, dist_array = evaluate(
            input_data,
            grd_global_descriptor,
            sat_global_descriptor,
        )

        results.set_distances(dist_array)

        accuracy_by_percent = accuracy_by_percent_list[int(input_data.top_k * 100)]
        print("model :  " + input_data.weights_name)

        print(
            "\taccuracy = ",
            (accuracy_by_percent * 100.0),
            "(for topk = ",
            topk_percent * 100,
            "% )",
        )

        results.set_percentage_accuracy(accuracy_by_percent_list)
        results.set_top100_accuracy(accuracy_by_top100_list)

        if input_data.charts.matches_chart:
            print("   generate matches chart")
            MatchesChart(input_data, display_real_distance=True)

        print("Job finished for model: ")
        print("\t", model_name)

    # ---------------------------------------------------------
