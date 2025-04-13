import requests, time, os

from pyproj import Transformer
import geopandas as gpd
import mercantile
from shapely.geometry import box

from pygeotile.tile import Tile

# number of columns in one batch
BATCH_SIZE = 100

# start from batch no. N
N = 0

# scrap next SCRAP_BATCHES batches
SCRAP_BATCHES = 14

BASE_DIR = "D:/datasets/Warsaw/sat/waw/2022/all-256/"

ZOOM_LEVEL = 20
BASE_URL = f"https://ukosne.um.warszawa.pl/Data/2022/{ZOOM_LEVEL}/"

GEOJSON = "warszawa.geojson"

if __name__ == "__main__":

    geo_structure = gpd.read_file(GEOJSON).to_crs("EPSG:4326")  # Web Mercator

    minlon, minlat, maxlon, maxlat = geo_structure.total_bounds

    print(geo_structure.total_bounds)

    # transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # # Przeliczenie na długość i szerokość geograficzną (WGS 84)
    # minlon, minlat = transformer.transform(minlon, minlat)
    # maxlon, maxlat = transformer.transform(minlon, maxlat)

    leftTopTile = Tile.for_latitude_longitude(
        maxlat, minlon, ZOOM_LEVEL
    )  # max lat becouse Warsaw is on the North
    rightBottomTile = Tile.for_latitude_longitude(minlat, maxlon, ZOOM_LEVEL)

    print(leftTopTile.google)
    print(rightBottomTile.google)

    counter = 0
    error_counter = 0
    success_counter = 0

    # split to batches
    n = N

    for base_x in range(
        leftTopTile.google[0] + N * BATCH_SIZE,
        rightBottomTile.google[0] + 1,
        BATCH_SIZE,
    ):
        # additional code to split scrapping to batches, to avoid block from server

        print(f"This is {n} batch.")
        if n == N + SCRAP_BATCHES:
            print(f"Scrapped {n} batches. Exiting...")
            break
        n += 1

        if base_x + BATCH_SIZE > rightBottomTile.google[0]:
            curr_batch_size = rightBottomTile.google[0] - base_x + 1
        else:
            curr_batch_size = BATCH_SIZE

        with requests.Session() as session:
            for i in range(curr_batch_size):
                x = base_x + i
                print(f"Downloading column: {x} / {rightBottomTile.google[0]}")
                path = BASE_DIR + f"{ZOOM_LEVEL}/{x}"
                if not os.path.exists(path):
                    os.makedirs(path)
                    print(f"Created directory: {path}")

                for y in range(leftTopTile.google[1], rightBottomTile.google[1]):
                    url = BASE_URL + f"{x}/{y}.jpg"
                    response = session.get(url)
                    # print("y: ", y)
                    if response.status_code == 200:
                        with open(
                            BASE_DIR + f"/{ZOOM_LEVEL}/{x}/{y}.jpg", "wb"
                        ) as file:
                            file.write(response.content)
                        success_counter += 1
                    elif response.status_code == 404:
                        error_counter += 1
                    else:
                        print("Error occured!!!")
                        print(response.status_code)
                    counter += 1
                    if counter % 100 == 0:
                        print(
                            f"\tDownloaded {counter} images, success (200): {success_counter}, error (404): {error_counter}"
                        )
                    time.sleep(0.01)
                print(
                    f"Number of correct images in the column: {success_counter} / {success_counter + error_counter}"
                )
