import csv

import numpy as np

from munch import Munch


class Results(Munch):

    sat_descriptors_subpath = "descriptors/sat/"
    pano_descriptors_subpath = "descriptors/panos/"
    distances_subpath = "distances/"
    accuracy_subpath = "accuracy/"

    def __init__(self, results_config, input_data):
        super().__init__(results_config)
        self.input_data = input_data
        self.name = input_data.name
        self.descriptors = Munch()
        self.distances = Munch()
        self.accuracy = Munch(self.accuracy)

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
