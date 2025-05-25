import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd


ROOT_DIR = "D:/datasets/Warsaw/"
ID = "014"
FULL_ID = ID+"_1_90"
GRID = (10, 20)  # (height, width)
ZL=19
GRID_STR = f"{GRID[0]}x{GRID[1]}"
PANO = f"tmp/{FULL_ID}/"+GRID_STR +f"/{FULL_ID}.jpg"
DISTANCE_FILE = f"C:/Users/masik/Workspace/inzynierka/repos/BEng-MK/results/distances/CVM-NET-I_occluded_"+FULL_ID+"_"+GRID_STR+f"_{ZL}zl.csv"


def normalize_image(image):
    """Normalize the image to the range [0, 1]."""
    return image / np.amax(image)

grid_distances = pd.read_csv(DISTANCE_FILE, index_col="pano\\sat")

img = cv2.imread(ROOT_DIR + PANO)

tile = (int(img.shape[0]/GRID[0]), int(img.shape[1]/GRID[1]))
print(tile)

img_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float64)

part_path = PANO.split(".")[-2]
sat_path = f"sat/waw/2022/orto/{ZL}/{ID}.jpg"
black_dist= grid_distances[sat_path][part_path + "_black.jpg"]
normal_dist= grid_distances[sat_path][PANO]

deltas = grid_distances[sat_path] - normal_dist


for idx, delta in zip(grid_distances.index, deltas):
    filename = idx.split("/")[-1]
    splited_filename = filename.split("_")
    if len(splited_filename) >= 4 and splited_filename[-1] != "black.jpg":
        x = int(splited_filename[-1].split(".")[0])
        y = int(splited_filename[-2])
        # print(f"X: {x}, Y: {y}")
        color = 0 #bgr

        if delta < 0:
            #RED
            img_mask[y*tile[0]:(y+1)*tile[0], x*tile[1]:(x+1)*tile[1],2] = abs(delta)
        else:
            #BLUE
            img_mask[y*tile[0]:(y+1)*tile[0], x*tile[1]:(x+1)*tile[1],0] = abs(delta)

img_mask=normalize_image(img_mask) * 255
img_mask=img_mask.astype(np.uint8)

img_heatmap = cv2.add(img, img_mask)

img_heatmap = np.clip(img_heatmap, 0, 255).astype(np.uint8)

print(img_heatmap.shape)
# print(mask)
cv2.imwrite(ROOT_DIR + f"tmp/{FULL_ID}/"+GRID_STR + f"/{FULL_ID}_heatmap.jpg", img_heatmap)


