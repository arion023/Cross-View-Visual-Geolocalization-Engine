import cv2
import numpy as np


import csv, munch, os

# camera types: "PIXPRO" or "Theta"
class AugmentedGenerator:

    sat_format = "JPG"
    ground_format = "JPG"
    districts_mapped = {
        "Wilanow": (1, 30),
        "Center": (31, 50),
        "Ursynow": (51, 70),
    }

    districts_aliases = {"Warsaw": ["Wilanow", "Center", "Ursynow"]}

    def __init__(self, dataset, root_dir):
        self.root_dir = root_dir
        self.dataset = dataset
        self.n_directions = 8  # number of perspectives to generate

        self.sat_prefix = (
            "sat/waw/2022/"
            + self.dataset.angle
            + "/"
            + str(self.dataset.zoom_level)
            + "/"
        )
        self.panos_prefix = "panos/"

        self.districts = {}
        tmp_districts = []

        for dist in self.dataset.districts:
            if dist in self.districts_aliases:
                tmp_districts += self.districts_aliases[dist]
            else:
                tmp_districts.append(dist)

        for dist in tmp_districts:
            self.districts[dist] = self.districts_mapped[dist]

        self.splits_prefix = "splits/perspectives/" + self.dataset.modification + "/"
        self.generate_filename()

    def generate_filename(self):
        filename = (
            "_".join(self.dataset.districts)
            + "_"
            + "_".join(self.dataset.cameras)
            + "_"
            + "_".join(self.dataset.seasons)
            + "_"
            + self.dataset.modification
            + "_"
            + "perspectives"
            + "_"
            + str(self.n_directions)
            + "_"
            + self.dataset.angle
            + "_"
            + str(self.dataset.zoom_level)
            + "zl"
            + ".csv"
        )
        self.set_filename(filename)

    def set_filename(self, filename):
        self.filename = self.splits_prefix + filename

    def generate(self, filename=None):
        if filename is not None:
            self.set_filename(filename)

        rows = []
        for dist, ids in self.districts.items():
            for camera in self.dataset.cameras:
                for season in self.dataset.seasons:
                    panos_full_prefix = (
                        self.panos_prefix
                        + season
                        + "/"
                        + dist
                        + "/"
                        + camera
                        + "/"
                        + self.dataset.modification
                        + "/"
                    )

                    for i in range(ids[0], ids[1] + 1):
                        rows += self.generate_rows(panos_full_prefix, i, 2)

        self.save_split(rows)
        print("Split file generated: ", self.filename)
        return self.filename

    def save_split(self, rows):
        with open(self.root_dir + self.filename, mode="w", newline="") as file:
            # Create a csv.writer object
            writer = csv.writer(file)
            # Write data to the CSV file
            writer.writerows(rows)

    def generate_rows(self, panos_full_prefix, image_id, number_of_panos):
        rows = []
        for i in range(1, number_of_panos + 1):
            sat_path = self.sat_prefix + f"{image_id:03}." + self.sat_format
            pano_path = panos_full_prefix + f"{image_id:03}_{i}." + self.ground_format
            try:
                pano_paths = self.generate_perspectives(pano_path)
                for pano_path in pano_paths:
                    row = []
                    row.append(sat_path)
                    row.append(pano_path)
                    row.append("empty")
                    rows.append(row)
            except FileNotFoundError as e:
                print(f"Error generating perspectives for {pano_path}: {e}")
                break
        return rows

    def generate_perspectives(self, pano_path):
        img = cv2.imread(self.root_dir + pano_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {pano_path}")

        # Assuming the image is a square panorama

        height, width, _ = img.shape
        delta_width = width // self.n_directions
        # Generate perspectives by shifting the image
        paths = []

        perspective_path_split = pano_path.split("/")
        perspective_path_split[-2] += "/perspectives"
        perspective_dir = "/".join(perspective_path_split[:-1])
        perspective_path_base = "/".join(perspective_path_split)
        perspective_path_base = perspective_path_base[:-4]
        if not os.path.exists(self.root_dir + perspective_dir):
            os.makedirs(self.root_dir + perspective_dir)
            print(f"Created directory: {self.root_dir + perspective_dir}")

        for i in range(self.n_directions):
            perspective = np.copy(img)

            perspective[:, :width-i*delta_width] = img[:, i*delta_width:]
            perspective[:,  width-i*delta_width:] = img[:, : i*delta_width]

            perspective_path = perspective_path_base+f"_{i*45}.jpg"

            paths.append(perspective_path)
            cv2.imwrite(self.root_dir + perspective_path, perspective)
        return paths

    def get_filename(self):
        return self.filename

    def get_split_path(self):
        return self.split_path

if __name__ == "__main__":
        root_dir = "D:/datasets/Warsaw/"

        dataset_mock = munch.Munch()
        dataset_mock.angle = "orto"
        #it generates 8 persepctives for each zoom so it is redundant for diffrent zoom levels, bc panos are the same
        dataset_mock.zoom_level = 19
        dataset_mock.districts = [ "Center"]
        dataset_mock.seasons = ["spring"] #,
        dataset_mock.cameras = ["PIXPRO"]
        dataset_mock.modification = "straighten"

        ag = AugmentedGenerator(dataset_mock, root_dir)
        ag.generate()



        # img = cv2.imread("D:/datasets/Warsaw/panos/autumn/Wilanow/PIXPRO/straighten/010_1.jpg")
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        # height, width, _ = img.shape
        # delta_width = width // 8
        # img2 = np.copy(img)
        # img2[:, :-delta_width] = img[:, delta_width:]
        # img2[:, -delta_width:] = img[:, :delta_width]
        # cv2.imshow("img2", img2)
        # cv2.waitKey(0)