import geopandas as gpd
import mercantile
from shapely.geometry import box

from geojson import Polygon, FeatureCollection, Feature

import numpy as np
import os

import cv2

ZOOM = 21

ROOT = "D:/datasets/Warsaw/sat/waw/2022/"


def read_tile_img(tile, size=256):
    path = (
        ROOT
        + f"all-{size}/"
        + str(tile.z)
        + "/"
        + str(tile.x)
        + "/"
        + str(tile.y)
        + ".jpg"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tile {tile} not found at {path}")

    return cv2.imread(path)


# 1. Wczytaj dane obszaru Warszawy (np. z pliku GeoJSON)
area = gpd.read_file("./tools/geojson/warszawa.geojson").to_crs(
    "EPSG:4326"
)  # Web Mercator
print(area.total_bounds)

# 2. Oblicz bounding box
minx, miny, maxx, maxy = area.total_bounds
tiles = list(mercantile.tiles(minx, miny, maxx, maxy, ZOOM - 1))
error_counter = 0
for i, tile in enumerate(tiles):
    childrens = mercantile.children(tile)
    top_left_tile, top_right_tile, bot_right_tile, bot_left_tile = childrens
    try:
        top_left_img = read_tile_img(top_left_tile)
        top_right_img = read_tile_img(top_right_tile)
        bot_left_img = read_tile_img(bot_left_tile)
        bot_right_img = read_tile_img(bot_right_tile)
    except FileNotFoundError as e:
        # print(e)
        error_counter += 1
        continue

    top_part = np.hstack([top_left_img, top_right_img])
    bot_part = np.hstack([bot_left_img, bot_right_img])
    img = np.vstack([top_part, bot_part])

    path = ROOT + "all-512/" + f"{ZOOM}/{tile.x}/"

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

    cv2.imwrite(path + f"{tile.y}.jpg", img)

    if i % 1000 == 0:
        print(
            f"Proccessed {i/len(tiles)*100:.1f}% tiles. Success {i-error_counter}, Errors: {error_counter}"
        )
