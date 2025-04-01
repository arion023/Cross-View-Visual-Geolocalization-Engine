import csv
import numpy as np

DISTANCE_FILE = "C:/Users/masik/Workspace/inzynierka/repos/BEng-MK/results/distances/CVM-NET-I_original_Warsaw_PIXPRO_autumn_straighten_crop_600_1000_orto_20zl.csv"

if __name__ == "__main__":
    with open(DISTANCE_FILE, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        sat_popularity = [0 for _ in range(len(header) - 1)]

        for row in reader:
            sat_sorted_indexes = np.argsort(row[1:])
            for n, idx in enumerate(sat_sorted_indexes):
                sat_popularity[idx] += n
        sat_popularity_order = [header[1 + i] for i in np.argsort(sat_popularity)]

        with open(
            "C:/Users/masik/Workspace/inzynierka/repos/BEng-MK/results/popularity/"
            + DISTANCE_FILE.split("/")[-1],
            "w",
            newline="",
        ) as popularity_file:
            writer = csv.writer(popularity_file)
            writer.writerow(["sat_id", "popularity"])
            for sat_id, popularity in zip(sat_popularity_order, sorted(sat_popularity)):
                writer.writerow([sat_id, popularity])
