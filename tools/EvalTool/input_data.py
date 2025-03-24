import cv2
import random
import numpy as np
import json
from SplitFileGenerator import SplitFileGenerator
from munch import Munch
import csv


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
