import csv


# camera types: "PIXPRO" or "Theta"
class SplitFileGenerator:

    sat_format = "JPG"
    ground_format = "JPG"
    districts_mapped = {
        "Wilanow": (1, 30),
        "Center": (31, 50),
        "Ursynow": (51, 70),
    }

    def __init__(self, config):
        self.dataset_config = config["dataset"]

        self.sat_prefix = (
            "sat/waw/2022/"
            + self.dataset_config["angle"]
            + "/"
            + str(self.dataset_config["zoom_level"])
            + "/"
        )
        self.panos_prefix = "panos/"
        self.seasons = self.dataset_config["seasons"]
        self.modification = self.dataset_config["modification"]
        self.cameras = self.dataset_config["cameras"]

        self.districts = {}
        for dist in self.dataset_config["districts"]:
            self.districts[dist] = self.districts_mapped[dist]

        self.splits_prefix = config["root_dir"] + "splits/"
        self.generate_filename()

    def generate_filename(self):
        filename = (
            "_".join(self.dataset_config["districts"])
            + "_"
            + "_".join(self.cameras)
            + "_"
            + "_".join(self.seasons)
            + "_"
            + self.modification
            + "_"
            + self.dataset_config["angle"]
            + "_"
            + str(self.dataset_config["zoom_level"])
            + "zl"
            + ".csv"
        )
        self.set_filename(filename)

    def set_filename(self, filename):
        self.filename = filename
        self.splits_path = self.splits_prefix + self.filename

    def generate(self, filename=None):
        if filename is not None:
            self.set_filename(filename)

        rows = []
        for dist, ids in self.districts.items():
            for camera in self.cameras:
                for season in self.seasons:
                    panos_full_prefix = (
                        self.panos_prefix
                        + season
                        + "/"
                        + dist
                        + "/"
                        + camera
                        + "/"
                        + self.modification
                        + "/"
                    )

                    for i in range(ids[0], ids[1] + 1):
                        rows += self.generate_rows(panos_full_prefix, i, 2)

        self.save_split(rows)

    def save_split(self, rows):
        with open(self.splits_path, mode="w", newline="") as file:
            # Create a csv.writer object
            writer = csv.writer(file)
            # Write data to the CSV file
            writer.writerows(rows)

    def generate_rows(self, panos_full_prefix, image_id, number_of_panos):
        rows = []
        for i in range(1, number_of_panos + 1):
            row = []
            row.append(self.sat_prefix + f"{image_id:03}." + self.sat_format)
            row.append(panos_full_prefix + f"{image_id:03}_{i}." + self.ground_format)
            row.append("empty")
            rows.append(row)
        return rows

    def get_filename(self):
        return self.filename

    def get_splits_path(self):
        return self.splits_path
