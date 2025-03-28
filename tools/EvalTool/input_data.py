import json
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from munch import Munch

from SplitFileGenerator import SplitFileGenerator


class InputDataContainer:
    def __init__(self, config_path):
        self.inputdata_list = []
        with open(config_path, "r") as cf:
            json_configs = json.load(cf)
            for config in json_configs:
                self.inputdata_list.append(InputDataEval(config))
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx < len(self.inputdata_list):
            self.current_idx += 1
            return self.inputdata_list[self.current_idx - 1]
        else:
            self.current_idx = 0
            raise StopIteration


class InputDataEval:

    # crop is tuple (top_crop, bottom_crop), if there is no crop, there should be (None, None)
    def __init__(self, config_dict):

        self.__dict__.update(config_dict)
        self.update()

    def update(self):
        self.dataset = Dataset(self.dataset, self)
        self.update_name()
        self.results = Results(self.results, self)

    def update_name(self):
        if self.name == "auto":
            self.name = (
                self.model
                + "_"
                + self.weights_name
                + "_"
                + "_".join(self.dataset.districts)
                + "_"
                + "_".join(self.dataset.cameras)
                + "_"
                + "_".join(self.dataset.seasons)
                + "_"
                + self.dataset.modification
                + "_"
                + "crop"
                + "_"
                + "_".join([str(c) for c in self.dataset.crop])
                + "_"
                + self.dataset.angle
                + "_"
                + str(self.dataset.zoom_level)
                + "zl"
            )


class Results(Munch):

    sat_descriptors_subpath = "descriptors/sat/"
    pano_descriptors_subpath = "descriptors/panos/"
    distances_subpath = "distances/"
    accuracy_subpath = "accuracy/"
    charts_subpath = "charts/"

    def __init__(self, results_config, input_data):
        super().__init__(results_config)
        self.input_data = input_data
        self.name = input_data.name
        self.descriptors = Munch()
        self.distances = Munch()
        self.accuracy = Munch(self.accuracy)
        self.charts = Munch(self.charts)

        rasults_prefix_path = self.path

        self.descriptors.sat_save_path = (
            rasults_prefix_path + self.sat_descriptors_subpath + self.name + ".csv"
        )

        self.descriptors.pano_save_path = (
            rasults_prefix_path + self.pano_descriptors_subpath + self.name + ".csv"
        )

        self.distances.save_path = (
            rasults_prefix_path + self.distances_subpath + self.name + ".csv"
        )

        self.accuracy.percentage_save_path = (
            rasults_prefix_path
            + self.accuracy_subpath
            + self.accuracy.filename
            + "_percentage.csv"
        )

        self.accuracy.top100_save_path = (
            rasults_prefix_path
            + self.accuracy_subpath
            + self.accuracy.filename
            + "_top100.csv "
        )

        self.charts.path = rasults_prefix_path + self.charts_subpath

    def set_sat_descriptors(self, sat_descriptors):
        self.descriptors.sat_values = sat_descriptors
        self.save_sat_descriptors()

    def set_pano_descriptors(self, pano_descriptors):
        self.descriptors.pano_values = pano_descriptors
        self.save_pano_descriptors()

    def set_distances(self, distances):
        self.distances.values = distances
        self.save_distances()

    def set_percentage_accuracy(self, accuracy):
        self.accuracy.percentage_values = accuracy
        self.save_percentage_accuracy()

    def set_top100_accuracy(self, accuracy):
        self.accuracy.top100_values = accuracy
        self.save_top100_accuracy()

    def save_pano_descriptors(self):
        descriptors = self.descriptors.pano_values
        list_of_img_paths = [
            self.input_data.dataset.get_pano(i) for i in range(len(descriptors[:, 0]))
        ]
        with open(self.descriptors.pano_save_path, "w", newline="") as file:
            writer = csv.writer(file)
            for i, img_path in enumerate(list_of_img_paths):
                # extract panorama img name
                writer.writerow([img_path, *descriptors[i, :]])

    def save_sat_descriptors(self):
        descriptors = self.descriptors.sat_values
        list_of_img_paths = [
            self.input_data.dataset.get_sat(i) for i in range(len(descriptors[:, 0]))
        ]
        with open(self.descriptors.sat_save_path, "w", newline="") as file:
            writer = csv.writer(file)
            for i, img_path in enumerate(list_of_img_paths):
                # extract panorama img name
                writer.writerow([img_path, *descriptors[i, :]])

    def save_distances(self):
        with open(self.distances.save_path, "w", newline="") as file:
            writer = csv.writer(file)
            sat_names = [
                self.input_data.dataset.get_sat(i)
                for i in range(len(self.distances.values[:, 0]))
            ]

            writer.writerow(["pano\sat", *sat_names])
            for j in range(len(self.distances.values[0, :])):
                # extract panorama img name
                pano_name = self.input_data.dataset.get_pano(
                    j
                )  # .split("/")[-1].split(".")[-2]
                writer.writerow([pano_name, *self.distances.values[:, j]])

    def save_percentage_accuracy(self):
        accuracy = self.accuracy.percentage_values
        with open(self.accuracy.percentage_save_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.name, *accuracy])

    def save_top100_accuracy(self):
        accuracy = self.accuracy.top100_values
        with open(self.accuracy.top100_save_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.name, *accuracy])

    def prepare_plot(self):
        charts = self.charts
        path_to_save = charts.path + self.input_data.name + ".png"

        if_recall_chart = charts.recall_chart
        if_compare_chart = charts.compare_chart

        self.charts.fig, rows, cols, offset = self.configure_plot_params()
        self.charts.offset = offset

        grid = (rows, cols)
        self.charts.grid = grid

        self.add_description()

        ax = plt.subplot2grid(grid, (0, 1), colspan=offset - 2, rowspan=offset - 1)

        if if_recall_chart:
            self.display_recall(ax)

        if if_compare_chart:
            self.display_matches()

        plt.savefig(path_to_save, bbox_inches="tight", dpi=150)
        plt.clf()
        plt.close()

    def configure_plot_params(self):

        cols = self.charts.display_nearest + 2  # 1 for pano and 1 for correct sat match

        number_of_panos = len(self.distances.values[0, :])

        offset = 0
        if self.charts.recall_chart:
            offset = cols

        rows = number_of_panos + offset

        fig = plt.figure(figsize=(min(20, 2 * cols), max(200, 2 * rows)))

        fig.subplots_adjust(top=0.94, wspace=0.05, hspace=0.3)

        return fig, rows, cols, offset

    def display_recall(self, ax):
        x = [t * 100 for t in np.arange(0, 1, 0.01).tolist()]
        accuracy = self.accuracy.percentage_values
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

    def display_matches(self):
        grid = self.charts.grid
        rows, cols = grid
        dataset = self.input_data.dataset
        root_dir = self.input_data.root_dir
        dist_array = self.distances.values

        number_of_panos = len(dist_array[0, :])

        for pano_id in range(len(dist_array[0, :])):
            row = self.charts.offset + pano_id

            if (pano_id + 1) % (number_of_panos // 10) == 0:
                print(f"Generates comp. no. {pano_id+1} / {number_of_panos} ")

            # display panorama photo to match
            ax = plt.subplot2grid(grid, (row, 0))

            img_path = dataset.get_pano(pano_id)
            self.display_pano(ax, root_dir + img_path)

            matched_sat_path = dataset.get_sat(pano_id)
            nearest_sat_indexes = self.get_nearest_sat_idx_list(pano_id)

            # display n next matches
            # 0 image in row already is pano, and it is done already, so start from second
            # last column is sat image with correct match
            for col in range(1, cols - 1):
                ax = plt.subplot2grid(grid, (row, col))
                next_nearest_sat_idx = nearest_sat_indexes[col - 1]
                img_path = dataset.get_sat(next_nearest_sat_idx)

                self.display_match(
                    ax,
                    root_dir + img_path,
                    distance=dist_array[next_nearest_sat_idx, pano_id],
                    if_correct=True if img_path == matched_sat_path else False,
                )

            ax = plt.subplot2grid(grid, (row, cols - 1))

            self.display_match(
                ax,
                root_dir + matched_sat_path,
                distance=dist_array[pano_id, pano_id],
                if_correct=True,
                color="yellow",
            )

    def get_nearest_sat_idx_list(self, pano_id):
        # TODO optimize
        min_idx = np.argsort(self.distances.values[:, pano_id])
        unique_min_idx = [
            id
            for id in min_idx
            if id in self.input_data.dataset.get_unique_sat_idx_list()
        ]
        return unique_min_idx

    def display_pano(self, ax, path):
        img = self.display_image(ax, path)

        path_splited = path.split("/")
        # write the photo description
        ax.set_ylabel(f"{path_splited[-5]}", fontsize=8)
        ax.set_xlabel(f"{path_splited[-1]}", fontsize=8)

        return img

    def display_match(self, ax, path, distance, if_correct, color="blue"):
        img = self.display_image(ax, path)
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
                edgecolor=color,
                facecolor="none",
            )

        ax.add_patch(rect)

        return img

    def display_image(self, ax, path):
        crop = self.input_data.dataset.crop
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

    def add_description(self, custom=""):
        # Add text to the whole plot
        self.charts.fig.suptitle(
            f"Image Matches Overview on\n{self.input_data.dataset.splitfile}",
            fontsize=22,
        )

        accuracy_to_display = ""

        # TODO correct this
        if self.charts.recall_chart:
            accuracy_to_display = f"{self.accuracy.percentage_values[int(self.input_data.top_k*100)]*100:.1f}% (For topK = {self.input_data.top_k*100}%)"

        number_of_panos = len(self.distances.values[0, :])

        description = (
            f"Model: {self.input_data.model}\n"
            + f"Dataset: {self.input_data.dataset.splitfile}\n"
            + f"Accuracy [%]: {str(accuracy_to_display)}\n"
            + f"Number of panos: {number_of_panos}\n"
            + f"Top crop: {self.input_data.dataset.crop[0]}\n"
            + f"Bottom crop: {self.input_data.dataset.crop[1]}\n"
        )

        self.charts.fig.text(
            0.5,
            0.975,
            description,
            ha="center",
            va="top",
            fontsize=18,
        )


class Dataset:
    def __init__(self, datset_config, input_data):
        self.root_dir = input_data.root_dir
        self.input_data = input_data
        self.__dict__.update(datset_config)
        print(self.__dict__)
        self.update()

        print("InputData::__init__: load %s" % self.splitfile)
        self.__cur_test_id = 0  # for training
        self.id_dataset_list = []
        self.id_dataset_idx_list = []
        self.id_matches_dict = {}
        self.pano_dataset_size = 0
        self.sat_dataset_size = 0
        with open(self.split_path, "r") as file:
            idx = 0
            for line in file:
                data = line.split(",")
                pano_id = (data[1].split("/")[-1]).split(".")[0]
                # satellite filename, streetview filename, pano_id
                self.id_dataset_list.append([data[0], data[1], pano_id])
                self.id_dataset_idx_list.append(idx)

                if data[1] != "":
                    self.pano_dataset_size += 1
                self.sat_dataset_size += 1
                idx += 1
        self.dataset_size = len(self.id_dataset_list)

        self.make_unique_sat_idx_list()
        print(
            "InputData::__init__: load",
            self.splitfile,
            " data_size =",
            self.dataset_size,
        )

    def update(self):
        self.update_split()

    def update_split(self):
        if self.splitfile == "auto":
            sfg = SplitFileGenerator(self, self.input_data.root_dir)
            self.splitfile = sfg.generate()
        else:
            print(
                "Attributes for dataset in config file will be ignored since split file is provided."
            )
        self.split_path = self.input_data.root_dir + self.splitfile

    def next_batch_scan(self, batch_size):
        if self.__cur_test_id >= self.dataset_size:
            self.__cur_test_id = 0
            return None, None

        sat_batch_size = batch_size
        pano_batch_size = batch_size

        if self.__cur_test_id + batch_size >= self.sat_dataset_size:
            sat_batch_size = self.sat_dataset_size - self.__cur_test_id
            batch_size = sat_batch_size

        if self.__cur_test_id + batch_size >= self.pano_dataset_size:
            pano_batch_size = max(0, self.pano_dataset_size - self.__cur_test_id)

        batch_grd = np.zeros([pano_batch_size, 224, 1232, 3], dtype=np.float32)
        batch_sat = np.zeros([sat_batch_size, 512, 512, 3], dtype=np.float32)
        sat_only_counter = 0
        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            # satellite
            img = cv2.imread(self.root_dir + self.id_dataset_list[img_idx][0])
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

            img = img.astype(np.float32)
            # img -= 100.0
            img[:, :, 0] -= 103.939  # Blue
            img[:, :, 1] -= 116.779  # Green
            img[:, :, 2] -= 123.6  # Red
            batch_sat[i, :, :, :] = img

            # ground
            if self.id_dataset_list[img_idx][1] != "":
                img = cv2.imread(self.root_dir + self.id_dataset_list[img_idx][1])

                img = img[self.crop[0] : len(img[:, 0, 0]) - self.crop[1], :, :]
                img = cv2.resize(img, (1232, 224), interpolation=cv2.INTER_AREA)

                img = img.astype(np.float32)
                # img -= 100.0
                img[:, :, 0] -= 103.939  # Blue
                img[:, :, 1] -= 116.779  # Green
                img[:, :, 2] -= 123.6  # Red
                batch_grd[i, :, :, :] = img
            else:
                sat_only_counter += 1

        self.__cur_test_id += batch_size
        print(
            "In that batch were",
            sat_only_counter,
            "/",
            batch_size,
            "images without panorama image.",
        )
        return batch_sat, batch_grd

    def get_crop(self):
        # crop is tuple (bottom_crop, top_crop)
        return self.crop

    def get_sat(self, idx):
        # 0 image is sat img path
        return self.id_dataset_list[idx][0]

    def get_pano(self, idx):
        # 1 image, is panorama img path
        return self.id_dataset_list[idx][1]

    def get_unique_sat_idx_list(self):
        # returns ids of unique satelite paths
        return self.unique_sat_idx_list

    def make_unique_sat_idx_list(self):
        unique_indices = {}
        for idx, elem in enumerate(self.id_dataset_list):
            if elem[0] not in unique_indices:
                unique_indices[elem[0]] = idx

        self.unique_sat_idx_list = list(unique_indices.values())

    def get_id_dataset_list(self):
        return self.id_dataset_list

    def get_dataset_size(self):
        return self.dataset_size

    def get_sat_dataset_size(self):
        return self.sat_dataset_size

    def get_pano_dataset_size(self):
        return self.pano_dataset_size

    def reset_scan(self):
        self.__cur_test_id = 0
