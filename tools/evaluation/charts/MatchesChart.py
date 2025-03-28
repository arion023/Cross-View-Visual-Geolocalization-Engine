import os

import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MatchesChart:
    def __init__(self, input_data, display_nearest=None):
        self.input_data = input_data
        self.results = self.input_data.results

        if display_nearest is None:
            self.display_nearest = input_data.charts.display_nearest
        else:
            self.display_nearest = display_nearest

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

        fig = plt.figure(figsize=(min(20, 2 * cols), max(200, 2 * rows)))

        fig.subplots_adjust(top=0.94, wspace=0.05, hspace=0.3)

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
        grid = self.grid
        rows, cols = grid
        dataset = self.input_data.dataset
        root_dir = self.input_data.root_dir
        dist_array = self.results.distances.values

        number_of_panos = len(dist_array[0, :])

        for pano_id in range(len(dist_array[0, :])):
            row = self.offset + pano_id

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
        min_idx = np.argsort(self.results.distances.values[:, pano_id])
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
