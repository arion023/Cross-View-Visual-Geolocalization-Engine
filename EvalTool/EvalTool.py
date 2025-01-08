# This is cut-out from train.py file to only evaluate models


import sys

sys.path.append(
    "C:/Users/masik/Workspace/inżynierka/repos/crossview_localisation_tensorflow-2.x/src/CVM-Net"
)

from cvm_net import cvm_net_I, cvm_net_II
from input_data import InputDataEval
import math

import argparse, json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.patches as patches

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def calculate_distance(grd_descriptor, sat_descriptor):
    return 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))


def calculate_accuracy(dist_array, topk_list, unique_sat_idx_list):

    accuracy = np.array([0.0 for _ in topk_list])
    data_amount = 0.0

    print("Correct match if it is in top", topk_list, "photos.")

    for j in range(dist_array.shape[0]):
        gt_dist = dist_array[j, j]
        false_prediction = 0

        for i in unique_sat_idx_list:
            if dist_array[i, j] < gt_dist:
                false_prediction += 1

        for idx, topk in enumerate(topk_list):
            if false_prediction < topk:
                # print("Correct match for pano with id:\t", j)
                accuracy[idx] += 1.0
        data_amount += 1.0

    return accuracy / data_amount


def prepare_topk_list(topk_percent_list, number_of_unique_sat):
    return [
        math.ceil(number_of_unique_sat * topk_percent)
        for topk_percent in topk_percent_list
    ]


def evaluate(dataset, grd_descriptor, sat_descriptor, topk_percent_list):
    dist_array = calculate_distance(grd_descriptor, sat_descriptor)
    unique_sat_idx_list = dataset.get_unique_sat_idx_list()
    print("unique IDs: ", unique_sat_idx_list)
    topk_list = prepare_topk_list(topk_percent_list, len(unique_sat_idx_list))

    accuracy_list = calculate_accuracy(dist_array, topk_list, unique_sat_idx_list)

    # np.savetxt('./Result/sorted_img_idx_by_distances.csv', tmp, fmt="%d", delimiter=',')

    return accuracy_list, dist_array


def prepare_plot(args_dict, dist_array, dataset, accuracy, topk_percent_list):
    root_dir = args_dict["root_dir"]
    path_to_save = args_dict["save_chart"]

    number_of_panos = len(dist_array[0, :])

    fig, rows, cols, offset = configure_plot_params(
        root_dir, args_dict["display"], number_of_panos
    )

    grid = (rows, cols)

    ax = plt.subplot2grid(grid, (0, 1), colspan=offset - 2, rowspan=offset - 1)

    display_recall(ax, topk_percent_list, accuracy)

    display_matches(dataset, path_to_save, root_dir, dist_array, grid, offset)


def configure_plot_params(img_root, display_n_matches, number_of_panos):

    cols = display_n_matches + 1

    offset = cols

    rows = number_of_panos + offset

    fig = plt.figure(figsize=(2 * cols, 2 * rows))

    add_description(
        fig,
        model=args_dict["model"],
        dataset=img_root.split("/")[-2],
        accuracy=accuracy,
        number_of_images=rows,
        additional=f'\nsplits = {args_dict["splits"].split("/")[-1]}',
    )

    fig.subplots_adjust(top=0.96, wspace=0.01, hspace=0.3)

    return fig, rows, cols, offset


def display_recall(ax, topk_percent_list, accuracy):
    ax.plot(topk_percent_list, accuracy, label="Recall TopK% ")


def display_matches(dataset, path_to_save, img_root, dist_array, grid, offset):
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

    plt.savefig(path_to_save, bbox_inches="tight", dpi=150)


def get_nearest_sat_idx_list(distances, unique_sat_idx_list):
    # TODO optimize
    min_idx = np.argsort(distances)
    unique_min_idx = [id for id in min_idx if id in unique_sat_idx_list]
    return unique_min_idx


def display_pano(ax, path, crop):
    img = display_image(ax, path, crop)

    path_splited = path.split("/")
    # write the photo description
    ax.set_ylabel(f"Location: {path_splited[-4]}", fontsize=8)
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

    if crop is not None:
        img[0 : crop[0], :, :] = img[0 : crop[0], :, :] // 2
        img[-crop[1] :, :, :] = img[-crop[1] :, :, :] // 2

    ax.imshow(img, aspect="equal")
    ax.set_xticks([])  # Turn off x-axis ticks
    ax.set_yticks([])  # Turn off y-axis ticks
    ax.spines["top"].set_visible(False)  # Turn off top spine
    ax.spines["right"].set_visible(False)  # Turn off right spine
    ax.spines["left"].set_visible(False)  # Turn off left spine
    ax.spines["bottom"].set_visible(False)  # Turn off bottom spine

    return img


def add_description(
    fig, model=None, dataset=None, accuracy=None, number_of_images=None, additional=None
):
    # Add text to the whole plot
    fig.suptitle("Image Matches Overview")
    fig.text(
        0.5,
        0.97,
        f"Model: {model}\nDataset: {dataset}\nOverall accuracy: {str(accuracy)}%\nNumber of images: {number_of_images}\nAdditional: {additional}",
        ha="center",
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

    # Model-related arguments
    parser.add_argument("--model", type=str, help="Type of model.")
    parser.add_argument("--weights", type=str, help="Path to the weights of model.")

    # Satellite image arguments
    parser.add_argument(
        "--load-sat", type=str, help="Path to load satellite images from."
    )
    parser.add_argument(
        "--save-sat", type=str, help="Path to save processed satellite images."
    )

    # Ground image arguments
    parser.add_argument(
        "--load-gnd", type=str, help="Path to load ground truth images from."
    )
    parser.add_argument(
        "--save-gnd", type=str, help="Path to save processed ground truth images."
    )

    # Results management
    parser.add_argument(
        "--load-results", type=str, help="Path to load previously calculated results."
    )
    parser.add_argument("--save-results", type=str, help="Path to save the results.")
    parser.add_argument(
        "--save-chart", type=str, help="Path to save a visual chart of the results."
    )

    # Associations and matching
    parser.add_argument(
        "--associations",
        type=str,
        help="File defining associations between satellite and ground images.",
    )
    parser.add_argument(
        "--top-x",
        type=float,
        default="5",
        help="Number of correct matches to consider, should be a fraction (e.g., '0.01' to consider as a success if match is in 1 percent best matches).",
    )

    # Display options
    parser.add_argument(
        "--display",
        type=int,
        default=5,
        help="Number of best-matched images to display for evaluation.",
    )

    # Single image evaluation
    parser.add_argument(
        "--img",
        type=str,
        help="Path to the image to evaluate. Requires --load-sat to be set.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.config is not None:
        with open(args.config, "r", encoding="UTF-8") as df:
            args_dict = json.load(df)
    else:
        args_dict = vars(args)

    # import data, crop = (top_crop, bottom_crop)
    input_data = InputDataEval(crop=(400, 1200), root_dir=args_dict["root_dir"])

    # define placeholders
    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name="sat_x")
    grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name="grd_x")
    keep_prob = tf.placeholder(tf.float32)

    # build model
    if args_dict["model"] == "CVM-NET-I":
        sat_global, grd_global = cvm_net_I(sat_x, grd_x, keep_prob, False)
    elif args_dict["model"] == "CVM-NET-II":
        sat_global, grd_global = cvm_net_II(sat_x, grd_x, keep_prob, False)
    else:
        print(
            "CONFIG ERROR: wrong network type, only CVM-NET-I and CVM-NET-II are valid"
        )

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # run model
    print("run model...")
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print("load model...")
        saver.restore(sess, args_dict["weights"])
        print("   Model loaded from: %s" % args_dict["weights"])
        print("load model...FINISHED")

        # ---------------------- validation ----------------------
        print("validate...")
        print("   compute global descriptors")
        input_data.reset_scan()
        sat_global_descriptor = np.zeros([input_data.get_dataset_size(), 4096])
        grd_global_descriptor = np.zeros([input_data.get_dataset_size(), 4096])
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

        print("   compute accuracy")
        topk_percent_list = np.arange(0, 1, 0.05).tolist()
        accuracy, dist_array = evaluate(
            input_data,
            grd_global_descriptor,
            sat_global_descriptor,
            topk_percent_list=topk_percent_list,  # args_dict["top_k"],
        )

        print("model :  " + args_dict["weights"])
        print("\taccuracy = ", (accuracy * 100.0))

        prepare_plot(
            args_dict, dist_array, input_data, accuracy * 100.0, topk_percent_list
        )

        # ---------------------------------------------------------
