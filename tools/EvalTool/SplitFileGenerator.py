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

    districts_aliases = {"Warsaw": ["Wilanow", "Center", "Ursynow"]}

    def __init__(self, dataset, root_dir):
        self.root_dir = root_dir
        self.dataset = dataset

        self.sat_prefix = (
            "sat/waw/2022/"
            + self.dataset.angle
            + "/"
            + str(self.dataset.zoom_level)
            + "/"
        )
        self.panos_prefix = "panos/"

        self.districts = {}
        tmp_districts = self.dataset.districts

        for dist in tmp_districts:
            if dist in self.districts_aliases:
                tmp_districts += self.districts_aliases[dist]
                tmp_districts.remove(dist)

        for dist in tmp_districts:
            self.districts[dist] = self.districts_mapped[dist]

        self.splits_prefix = "splits/"
        self.generate_filename()

    def generate_filename(self):
        filename = (
            "_".join(self.districts)
            + "_"
            + "_".join(self.dataset.cameras)
            + "_"
            + "_".join(self.dataset.seasons)
            + "_"
            + self.dataset.modification
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
            row = []
            row.append(self.sat_prefix + f"{image_id:03}." + self.sat_format)
            row.append(panos_full_prefix + f"{image_id:03}_{i}." + self.ground_format)
            row.append("empty")
            rows.append(row)
        return rows

    def get_filename(self):
        return self.filename

    def get_split_path(self):
        return self.split_path
