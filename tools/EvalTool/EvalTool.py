# This is cut-out from train.py file to only evaluate models


import sys

sys.path.append(
    "C:/Users/masik/Workspace/inżynierka/repos/crossview_localisation_tensorflow-2.x/src/CVM-Net"
)

from cvm_net import cvm_net_I, cvm_net_II
from input_data import InputDataEval
import math
import csv

import argparse, json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.patches as patches

from configs import load_configs

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def calculate_distance(grd_descriptor, sat_descriptor):
    return 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))


def calculate_accuracy(dist_array, topk_percent_given, unique_sat_idx_list):
    number_of_unique_sat = len(unique_sat_idx_list)
    topk_list = prepare_topk_list(number_of_unique_sat)
    topk_given = math.ceil(number_of_unique_sat * topk_percent_given)

    accuracy = np.array([0.0 for _ in topk_list])
    data_amount = 0.0

    print("Correct match if it is in top", topk_list, "photos.")

    for j in range(dist_array.shape[1]):
        gt_dist = dist_array[j, j]
        false_prediction = 0

        for i in unique_sat_idx_list:
            if dist_array[i, j] < gt_dist:
                false_prediction += 1

        for idx, topk in enumerate(topk_list):
            if false_prediction < topk:
                if topk == topk_given:
                    print("Correct match for pano with id:\t", j)
                accuracy[idx] += 1.0
        data_amount += 1.0

    accuracy /= data_amount
    return accuracy


def prepare_topk_list(number_of_unique_sat):
    topk_percent_list = np.arange(0, 1, 0.01).tolist()
    return [
        math.ceil(number_of_unique_sat * topk_percent)
        for topk_percent in topk_percent_list
    ]


def evaluate(dataset, grd_descriptor, sat_descriptor, topk_percent):
    dist_array = calculate_distance(grd_descriptor, sat_descriptor)
    unique_sat_idx_list = dataset.get_unique_sat_idx_list()
    print("Unique IDs: ", unique_sat_idx_list)
    accuracy_list = calculate_accuracy(dist_array, topk_percent, unique_sat_idx_list)

    # np.savetxt('./Result/sorted_img_idx_by_distances.csv', tmp, fmt="%d", delimiter=',')

    return accuracy_list, dist_array


def prepare_plot(args, dist_array, dataset, accuracy, topk_percent):
    root_dir = args["root_dir"]
    chart_args = args["results"]["charts"]
    path_to_save = chart_args["save_path"] + args["name"] + ".png"

    number_of_panos = len(dist_array[0, :])

    if_recall_chart = chart_args["recall_chart"]
    if_compare_chart = chart_args["compare_chart"]

    accuracy_to_display = ""
    if if_recall_chart and len(accuracy) == 100:
        accuracy_to_display = (
            f"{accuracy[int(topk_percent*100)]:.1f}% (For topK = {topk_percent*100}%)"
        )
    else:
        accuracy_to_display = f"{accuracy[0]}% (For topK = {topk_percent*100}%)"

    fig, rows, cols, offset = configure_plot_params(
        chart_args["display_nearest"], number_of_panos, if_recall_chart
    )

    grid = (rows, cols)

    add_description(
        fig,
        model=args["model"],
        dataset=root_dir.split("/")[-2],
        accuracy=accuracy_to_display,
        number_of_panos=number_of_panos,
        crop=dataset.get_crop(),
    )

    ax = plt.subplot2grid(grid, (0, 1), colspan=offset - 2, rowspan=offset - 1)

    if if_recall_chart:
        display_recall(ax, np.arange(0, 1, 0.01).tolist(), accuracy)

    if if_compare_chart:
        display_matches(dataset, root_dir, dist_array, grid, offset)

    plt.savefig(path_to_save, bbox_inches="tight", dpi=150)
    plt.clf()
    plt.close()


def configure_plot_params(display_n_matches, number_of_panos, recall_chart):

    cols = display_n_matches + 1

    offset = 0
    if recall_chart:
        offset = cols

    rows = number_of_panos + offset

    fig = plt.figure(figsize=(min(20, 2 * cols), max(200, 2 * rows)))

    fig.subplots_adjust(top=0.94, wspace=0.01, hspace=0.3)

    return fig, rows, cols, offset


def display_recall(ax, topk_percent_list, accuracy):
    x = [t * 100 for t in topk_percent_list]
    ax.plot(x, accuracy)
    ax.set_title("Recall TopK% Chart")
    ax.set_ylabel("Accuracy [%]")
    ax.set_xlabel("Top K [%]")
    ax.set_xticks([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    for i in [1, 5, 10, 20]:
        ax.annotate(
            f"{accuracy[i]:.1f}%",
            (x[i], accuracy[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=14,
            fontweight="bold",
        )

    ax.grid(True)


def display_matches(dataset, img_root, dist_array, grid, offset):
    rows, cols = grid

    number_of_panos = len(dist_array[0, :])

    for pano_id in range(len(dist_array[0, :])):
        row = offset + pano_id

        if (pano_id + 1) % (number_of_panos // 10) == 0:
            print(f"Generates comp. no. {pano_id+1} / {number_of_panos} ")

        # display panorama photo to match
        ax = plt.subplot2grid(grid, (row, 0))

        img_path = dataset.get_pano(pano_id)
        display_pano(ax, img_root + img_path, dataset.get_crop())

        matched_sat_path = dataset.get_sat(pano_id)
        nearest_sat_indexes = get_nearest_sat_idx_list(
            dist_array[:, pano_id], dataset.get_unique_sat_idx_list()
        )

        # display n next matches
        # 0 image in row already is pano, and it is done already, so start from second
        for col in range(1, cols):
            ax = plt.subplot2grid(grid, (row, col))
            next_nearest_sat_idx = nearest_sat_indexes[col - 1]
            img_path = dataset.get_sat(next_nearest_sat_idx)

            display_match(
                ax,
                img_root + img_path,
                distance=dist_array[next_nearest_sat_idx, pano_id],
                if_correct=True if img_path == matched_sat_path else False,
            )


def get_nearest_sat_idx_list(distances, unique_sat_idx_list):
    # TODO optimize
    min_idx = np.argsort(distances)
    unique_min_idx = [id for id in min_idx if id in unique_sat_idx_list]
    return unique_min_idx


def display_pano(ax, path, crop):
    img = display_image(ax, path, crop)

    path_splited = path.split("/")
    # write the photo description
    ax.set_ylabel(f"Location: {path_splited[-5]}", fontsize=8)
    ax.set_xlabel(f"{path_splited[-1]}", fontsize=8)

    return img


def display_match(ax, path, distance, if_correct):
    img = display_image(ax, path)
    path_splited = path.split("/")
    ax.set_xlabel(f"{path_splited[-1]}\nDistance: {distance:.4f}", fontsize=8)

    rect = patches.Rectangle(
        (0, 0),
        img.shape[1],
        img.shape[0],
        linewidth=3,
        edgecolor="red",
        facecolor="none",
    )

    if if_correct:
        rect = patches.Rectangle(
            (0, 0),
            img.shape[1],
            img.shape[0],
            linewidth=4,
            edgecolor="blue",
            facecolor="none",
        )

    ax.add_patch(rect)

    return img


def display_image(ax, path, crop=None):

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # make cropped part of image darker
    if crop is not None:
        bottom_crop = len(img[:, 0, 0]) - crop[1]
        img[0 : crop[0], :, :] //= 2
        img[bottom_crop:, :, :] //= 2

    ax.imshow(img, aspect="equal")
    ax.set_xticks([])  # Turn off x-axis ticks
    ax.set_yticks([])  # Turn off y-axis ticks
    ax.spines["top"].set_visible(False)  # Turn off top spine
    ax.spines["right"].set_visible(False)  # Turn off right spine
    ax.spines["left"].set_visible(False)  # Turn off left spine
    ax.spines["bottom"].set_visible(False)  # Turn off bottom spine

    return img


def add_description(
    fig,
    model=None,
    dataset=None,
    accuracy=None,
    number_of_panos=None,
    crop=(0, 0),
    **additional,
):
    # Add text to the whole plot
    fig.suptitle(f"Image Matches Overview on {dataset}", fontsize=22)

    description = (
        f"Model: {model}\n"
        + f"Dataset: {dataset}\n"
        + f"Accuracy: {str(accuracy)}\n"
        + f"Number of panos: {number_of_panos}\n"
        + f"Top crop: {crop[0]}\n"
        + f"Bottom crop: {crop[1]}\n"
    )
    for name, text in additional.items():
        description += name + ": " + text + "\n"

    fig.text(
        0.5,
        0.975,
        description,
        ha="center",
        va="top",
        fontsize=18,
    )


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


def prepare_and_run_model(model_name, input_data):
    # define placeholders
    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name="sat_x")
    grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name="grd_x")
    keep_prob = tf.placeholder(tf.float32)

    # build model
    if model_name == "CVM-NET-I":
        sat_global, grd_global = cvm_net_I(sat_x, grd_x, keep_prob, False)
    elif model_name == "CVM-NET-II":
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
        saver.restore(sess, args["weights_path"])
        print("   Model loaded from: %s" % args["weights_path"])
        print("load model...FINISHED")

        # ---------------------- validation ----------------------
        print("validate...")
        print("   compute global descriptors")
        input_data.reset_scan()
        sat_global_descriptor = np.zeros([input_data.get_sat_dataset_size(), 4096])
        grd_global_descriptor = np.zeros([input_data.get_pano_dataset_size(), 4096])
        val_i = 0
        while True:
            print("      progress %d" % val_i)
            batch_sat, batch_grd = input_data.next_batch_scan(12)
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


def save_pano_descriptors(file_path, descriptors, dataset):
    list_of_img_paths = [dataset.get_pano(i) for i in range(len(descriptors[:, 0]))]
    save_descriptors(file_path, descriptors, list_of_img_paths)


def save_sat_descriptors(file_path, descriptors, dataset):
    list_of_img_paths = [dataset.get_sat(i) for i in range(len(descriptors[:, 0]))]
    save_descriptors(file_path, descriptors, list_of_img_paths)


def save_descriptors(file_path, descriptors, list_of_img_paths):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        for i, img_path in enumerate(list_of_img_paths):
            # extract panorama img name
            writer.writerow([img_path, *descriptors[i, :]])


def save_distances(file_path, distances, dataset):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        sat_names = [dataset.get_sat(i) for i in range(len(distances[:, 0]))]

        writer.writerow(["pano\sat", *sat_names])
        for j in range(len(distances[0, :])):
            # extract panorama img name
            pano_name = dataset.get_pano(j)  # .split("/")[-1].split(".")[-2]
            writer.writerow([pano_name, *distances[:, j]])


def save_accuracy(file_path, model_name, accuracy):
    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([model_name, *accuracy])


if __name__ == "__main__":
    args = parse_arguments()

    if args.config is not None:
        args_list = load_configs(args.config)
    else:
        raise Exception("No config file.")

    for args in args_list:
        model_name = args["name"]
        input_data = InputDataEval(
            crop=args["dataset"]["crop"],
            root_dir=args["root_dir"],
            splits="splits/" + args["dataset"]["splits"],
        )

        grd_global_descriptor, sat_global_descriptor = prepare_and_run_model(
            args["model"], input_data
        )

        # putted here thus it can reduce memory overload during plots generating.
        tf.reset_default_graph()

        print("   compute accuracy")

        results_args = args["results"]

        if results_args["descriptors"]["save"]:
            save_pano_descriptors(
                results_args["descriptors"]["pano_save_path"],
                grd_global_descriptor,
                input_data,
            )
            save_sat_descriptors(
                results_args["descriptors"]["sat_save_path"],
                sat_global_descriptor,
                input_data,
            )

        topk_percent = args["top_k"]

        accuracy_list, dist_array = evaluate(
            input_data,
            grd_global_descriptor,
            sat_global_descriptor,
            topk_percent,
        )

        if results_args["distances"]["save"]:
            save_distances(
                results_args["distances"]["save_path"],
                dist_array,
                input_data,
            )

        accuracy = accuracy_list[int(args["top_k"] * 100)]
        print("model :  " + args["weights_name"])

        print("\taccuracy = ", (accuracy * 100.0), "(for topk = )", topk_percent)

        if results_args["accuracy"]["save"]:
            save_accuracy(
                results_args["accuracy"]["save_path"], model_name, accuracy_list
            )

        prepare_plot(
            args,
            dist_array,
            input_data,
            accuracy_list * 100.0,
            topk_percent=topk_percent,
        )

    # ---------------------------------------------------------
