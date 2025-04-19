import os

import csv
import math
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MatchesChart:
    def __init__(self, input_data, display_nearest=None, display_real_distance=False):
        self.input_data = input_data
        self.results = self.input_data.results
        self.display_real_distance = display_real_distance
        if display_nearest is None:
            self.display_nearest = input_data.charts.display_nearest
        else:
            self.display_nearest = display_nearest

        self.locations = None
        if display_real_distance:
            with open(self.input_data.root_dir + "locations.csv", "r") as file:
                reader = csv.reader(file)
                header = next(reader)
                self.locations = {row[0].zfill(3): (row[1], row[2]) for row in reader}

        directory_path = self.input_data.charts.path + self.input_data.name
        try:
            os.mkdir(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_path}' already exists.")
        self.save_path = directory_path + "/matches.png"

        self.fig, rows, cols, offset = self.configure_plot_params()
        self.offset = offset

        grid = (rows, cols)
        self.grid = grid

        self.add_description()

        ax = plt.subplot2grid(grid, (0, 1), colspan=offset - 2, rowspan=offset - 1)

        self.display_recall(ax)
        self.display_matches()

        plt.savefig(self.save_path, bbox_inches="tight", dpi=150)
        plt.clf()
        plt.close()

    def configure_plot_params(self):

        cols = self.display_nearest + 2  # 1 for pano and 1 for correct sat match

        number_of_panos = len(self.results.distances.values[0, :])

        offset = 0
        offset = cols

        rows = number_of_panos + offset

        fig = plt.figure(figsize=(min(20, 2 * cols), 2 * rows))

        fig.subplots_adjust(top=0.94, wspace=0.15, hspace=0.25)

        return fig, rows, cols, offset

    def display_recall(self, ax):
        x = [t * 100 for t in np.arange(0, 1, 0.01).tolist()]
        accuracy = self.results.accuracy.percentage_values
        ax.plot(x, accuracy)
        ax.set_title("Recall TopK% Chart")
        ax.set_ylabel("Accuracy [%]")
        ax.set_xlabel("Top K [%]")
        ax.set_xticks([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        for i in [1, 5, 10, 20]:
            ax.annotate(
                f"{accuracy[i]*100:.1f}%",
                (x[i], accuracy[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=14,
                fontweight="bold",
            )

        ax.grid(True)

    def display_matches(self):
        grid = self.grid
        rows, cols = grid
        dataset = self.input_data.dataset
        root_dir = self.input_data.root_dir
        dist_array = self.results.distances.values

        number_of_panos = len(dist_array[0, :])

        for pano_id in range(len(dist_array[0, :])):
            row = self.offset + pano_id

            if (pano_id + 1) % (number_of_panos // 10) == 0:
                print(f"Generates matches... progress {pano_id+1} / {number_of_panos} ")

            # display panorama photo to match
            ax = plt.subplot2grid(grid, (row, 0))

            pano_img_path = dataset.get_pano(pano_id)
            self.display_pano(ax, root_dir + pano_img_path)

            matched_sat_path = dataset.get_sat(pano_id)
            nearest_sat_indexes = self.get_nearest_sat_idx_list(pano_id)
            nearest_sat_indexes_paths =  list(map(dataset.get_sat, nearest_sat_indexes))
            # display n next matches
            # 0 image in row already is pano, and it is done already, so start from second
            # last column is sat image with correct match
            for col in range(1, cols - 1):
                ax = plt.subplot2grid(grid, (row, col))
                next_nearest_sat_idx = nearest_sat_indexes[col - 1]
                img_path = dataset.get_sat(next_nearest_sat_idx)

                color = "red"
                if img_path == matched_sat_path:
                    color = "blue"

                real_distance = None
                if self.display_real_distance:
                    real_distance = self.calculate_real_distance(
                        pano_img_path, img_path
                    )

                self.display_match(
                    ax,
                    root_dir + img_path,
                    distance=dist_array[next_nearest_sat_idx, pano_id],
                    real_distance=real_distance,
                    color=color,
                )

            ax = plt.subplot2grid(grid, (row, cols - 1))

            self.display_match(
                ax,
                root_dir + matched_sat_path,
                distance=dist_array[pano_id, pano_id],
                real_distance=0,
                color="yellow",
                custom_text=f'; no. {nearest_sat_indexes_paths.index(matched_sat_path)}'
            )

    def get_nearest_sat_idx_list(self, pano_id):
        # TODO optimize
        min_idx = np.argsort(self.results.distances.values[:, pano_id])
        unique_min_idx = [
            id
            for id in min_idx
            if id in self.input_data.dataset.get_unique_sat_idx_list()
        ]
        return unique_min_idx

    def display_pano(self, ax, path):
        img = self.display_image(ax, path, apply_crop=True)

        path_splited = path.split("/")
        # write the photo description
        ax.set_ylabel(f"{path_splited[-5]}", fontsize=8)
        ax.set_xlabel(f"{path_splited[-1]}", fontsize=8)

        return img

    def display_match(self, ax, path, distance, real_distance=None, color="blue", custom_text=""):
        img = self.display_image(ax, path)
        path_splited = path.split("/")

        label = f"{path_splited[-1]}\nDistance: {distance:.4f}" + custom_text
        ax.set_xlabel(
            label,
            fontsize=8,
        )
        if self.display_real_distance:
            ax.set_ylabel(
                f"Real distance: {real_distance:.2f} km",
                fontsize=8,
            )

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

    def display_image(self, ax, path, apply_crop=False):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # make cropped part of image darker
        if apply_crop:
            crop = self.input_data.dataset.crop
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

    def calculate_real_distance(self, pano_path, sat_path):
        pano_id = pano_path.split("/")[-1].split("_")[0]
        sat_id = sat_path.split("/")[-1].split(".")[0]

        if self.locations is None:
            raise ValueError("Locations are not loaded.")

        if pano_id not in self.locations or sat_id not in self.locations:
            raise ValueError("Location is not in the locations list.")

        (lat1, lon1) = self.locations[pano_id]
        (lat2, lon2) = self.locations[sat_id]

        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)

        x = (lon2 - lon1) * math.cos((lat1 + lat2) / 2 * math.pi / 180)
        y = lat2 - lat1
        return math.sqrt(x**2 + y**2) * 111.32

    # TODO add custom description
    def add_description(self):
        # Add text to the whole plot
        self.fig.suptitle(
            f"Image Matches Overview on\n{self.input_data.dataset.splitfile}",
            fontsize=22,
        )

        accuracy_to_display = ""

        accuracy_to_display = f"{self.results.accuracy.percentage_values[int(self.input_data.top_k*100)]*100:.1f}% (For topK = {self.input_data.top_k*100}%)"

        number_of_panos = len(self.results.distances.values[0, :])

        description = (
            f"Model: {self.input_data.model}\n"
            + f"Weights: {self.input_data.weights_name}\n"
            + f"Dataset: {self.input_data.dataset.splitfile}\n"
            + f"Accuracy [%]: {str(accuracy_to_display)}\n"
            + f"Number of panos: {number_of_panos}\n"
            + f"Top crop: {self.input_data.dataset.crop[0]}\n"
            + f"Bottom crop: {self.input_data.dataset.crop[1]}\n"
            + f"\n{self.input_data.charts.custom_info}\n"
        )

        self.fig.text(
            0.5,
            0.975,
            description,
            ha="center",
            va="top",
            fontsize=18,
        )
