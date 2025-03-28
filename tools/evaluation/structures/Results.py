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

    def load(self):
        self.load_sat_descriptors()
        self.load_pano_descriptors()
        self.load_distances()
        self.load_percentage_accuracy()
        self.load_top100_accuracy()
        print("All results loaded successfully.")

    def load_percentage_accuracy(self):
        with open(self.accuracy.percentage_save_path, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == self.name:
                    self.accuracy.percentage_values = np.array(row[1:], dtype=float)
                    break
        print("Percentage accuracy loaded successfully.")

    def load_top100_accuracy(self):
        with open(self.accuracy.top100_save_path, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == self.name:
                    self.accuracy.top100_values = np.array(row[1:], dtype=float)
                    break
        print("Top100 accuracy loaded successfully.")

    def load_distances(self):
        with open(self.distances.save_path, "r") as file:
            distances = []

            reader = csv.reader(file)
            header = next(reader)
            for i, sat_name in enumerate(header[1:]):
                if sat_name != self.input_data.dataset.get_sat(i):
                    raise ValueError(
                        "Incorrect satellites order in the distances file."
                    )

            for i, row in enumerate(reader):
                if row[0] == self.input_data.dataset.get_pano(i):
                    distances.append(row[1:])
                else:
                    raise ValueError("Incorrect panoramas order in the distances file.")

            self.distances.values = np.array(distances, dtype=float)
            print("Distances loaded successfully.")

    def load_sat_descriptors(self):
        with open(self.descriptors.sat_save_path, "r") as file:
            descriptors = []

            reader = csv.reader(file)

            for i, row in enumerate(reader):
                if row[0] == self.input_data.dataset.get_sat(i):
                    descriptors.append(row[1:])
                else:
                    raise ValueError(
                        "Incorrect satelites order in the descriptors file."
                    )

            self.descriptors.sat_values = np.array(descriptors, dtype=float)
            print("Satellite descriptors loaded successfully.")

    def load_pano_descriptors(self):
        with open(self.descriptors.pano_save_path, "r") as file:
            descriptors = []

            reader = csv.reader(file)

            for i, row in enumerate(reader):
                if row[0] == self.input_data.dataset.get_pano(i):
                    descriptors.append(row[1:])
                else:
                    raise ValueError("Incorrect panos order in the descriptors file.")

            self.descriptors.sat_values = np.array(descriptors, dtype=float)
            print("Satellite descriptors loaded successfully.")
