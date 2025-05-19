import folium
import pandas as pd
import csv
import mercantile
import geopandas as gpd
import numpy as np
import cv2

ZOOM = 19
SEQUENCE_N = 1
# DISTANCE_PATH = "./results/distances/CVM-NET-I_original_Warsaw_Tile_19_PIXPRO_autumn_straighten_crop_600_1000_orto_19zl.csv"
DISTANCE_PATH = "results/distances/CVM-NET-I_original_Sequences_Warsaw_512px_PIXPRO_spring_raw_crop_800_1000_orto_19zl.csv"
# panos/spring/Wilanow/PIXPRO/raw/


root_dir = "D:/datasets/Warsaw/"
tileurl = "https://ukosne.um.warszawa.pl/Data/2022/{z}/{x}/{y}.jpg"

m = folium.Map(
    (52.22983601573939, 21.011703206081123),
    zoom_start=15,
    tiles=tileurl,
    attr="Warszawa-ukosna",
)


tileArcGIS = folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Esri Satellite",
    overlay=False,
    control=True,
    show=False
).add_to(m)


area = gpd.read_file("./tools/geojson/warszawa-dzielnice.geojson").to_crs(
    "EPSG:4326"
)  # Web Mercator


print(area.total_bounds)

# 2. Oblicz bounding box
minlon, minlat, maxlon, maxlat = area.total_bounds
tiles = list(mercantile.tiles(minlon, minlat, maxlon, maxlat, ZOOM-1))

minx = tiles[1].x
miny = tiles[1].y

maxx = tiles[-1].x
maxy = tiles[-1].y

# tiles_values = pd.read_csv(f"./../results/distances/CVM-NET-I_original_Warsaw_Tile_19_PIXPRO_autumn_straighten_crop_600_1000_orto_19zl")
tiles_values = pd.read_csv(DISTANCE_PATH, index_col="pano\\sat")
# tiles_values =  tiles_values.transpose()
tile_ids = []

pano_ids = []
pano_paths = []
pano_coords =[]
# "panos/spring/Wilanow/PIXPRO/raw/"
pano_base = "panos/spring/"
with open(root_dir + f"locations/s{SEQUENCE_N}.csv" ) as f:
    reader = csv.reader(f)
    header = next(reader)
    district = ""
    ending = ""
    for row in reader:
        pid = row[0]
        pano_ids.append(pid)
        if pid.startswith("S"):
            if int(pid[1])==4:
                district = "Center"
            elif int(pid[1])==5:
                district="Ursynow"
            else:
                district="Wilanow"
            ending = f"/PIXPRO/raw/sequences/S{SEQUENCE_N}/"

        else:
            number_id = int(pid)
            if number_id<31:
                district="Wilanow"
            elif number_id<51:
                district="Center"
            else:
                district="Ursynow"
            ending = "/PIXPRO/raw/"

        pano_paths.append(pano_base + district + ending + pid +"_1.jpg")
        #TODO add _2
        lat = float(row[1])
        lon = float(row[2])
        pano_coords.append((lat, lon))

print(pano_paths)

x_shape = maxx - minx
y_shape = maxy - miny

distances = np.ones(dtype=np.float64, shape=(y_shape, x_shape, len(pano_ids)))
probability = np.ones(dtype=np.float64, shape=(y_shape, x_shape, len(pano_ids)))
seq_probability = np.zeros(dtype=np.float64, shape=(y_shape, x_shape, len(pano_ids)))
sat_list = tiles_values.columns.to_list()

# idx = tiles_values.index.to_list().index(pano_paths[0])
# print(idx)
# print(tiles_values.iloc[idx])

print(tiles_values.loc[pano_paths[0]])

for i, pano_path in enumerate(pano_paths):
    for sat, distance in zip(sat_list, tiles_values.loc[pano_path]):
        splited_path=sat.split("/")
        if len(splited_path)==7 and splited_path[3][0:3] == "all":
            zoom = splited_path[4]
            x = splited_path[5]
            y = splited_path[6].split(".")[0]
            # print(distance)
            distances[ int(y)-miny, int(x)-minx, i] *= float(distance)

np.clip(distances, 0, 1, out=distances)
probability = probability - distances
seq_probability[:,:,0] = probability[:,:,0]

kernel = np.ones((3, 3), np.float32)/9

# kernel = np.array([
#     [-1, -2, -1],
#     [ 0,  0,  0],
#     [ 1,  2,  1]
# ], dtype=np.float32)

for i in range(1, len(pano_paths)):
    seq_probability[:,:,i] = seq_probability[:,:, i-1]
    seq_probability[:,:,i] = cv2.filter2D(seq_probability[:,:,i], ddepth=-1, kernel=kernel)
    seq_probability[:,:,i] *= probability[:,:,i]


# PANO = "panos/autumn/Ursynow/PIXPRO/straighten/062_1.JPG"

prev_id = 0

sequence_group = folium.FeatureGroup(name=f"LINE S{SEQUENCE_N}").add_to(m)
folium.PolyLine(pano_coords, color="orange", weight=4, opacity=1).add_to(sequence_group)

for i, pano_id in enumerate(pano_ids):
    print("Starting pano: ", pano_id)

    # gradient_img = gradient_img**2
    # print(gradient_img)

    # if prev_id != id:

    folium.Marker(
    location=[pano_coords[i][0], pano_coords[i][1]],
    tooltip="ID "+str(id),
    icon=folium.Icon(icon="camera", color="red"),
    ).add_to(sequence_group)

    folium.raster_layers.ImageOverlay(
        image=seq_probability[:,:,i],
        name=pano_id,
        bounds=[[minlat, minlon], [maxlat, maxlon]],
        opacity=0.9,
        colormap = lambda x: (1, 1, 0, x) if x > 0.9 else (0, 0, x, x),
        show=False
    ).add_to(m)


folium.LayerControl().add_to(m)


m.save(f"results/heatmap-s{SEQUENCE_N}.html")