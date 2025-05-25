import folium
import pandas as pd
import json
import mercantile
import geopandas as gpd
import numpy as np


ZOOM = 19
# DISTANCE_PATH = "./results/distances/CVM-NET-I_original_Warsaw_Tile_19_PIXPRO_autumn_straighten_crop_600_1000_orto_19zl.csv"
DISTANCE_PATH = "./results/distances/CVM-NET-I_original_Warsaw_Tile_19_PIXPRO_autumn_straighten_crop_600_1000_orto_19zl.csv"

root_dir = "D:\datasets\Warsaw"
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
tiles_values =  tiles_values.transpose()
tile_ids = []

panos_ids = tiles_values.columns.to_list()

x_shape = maxx - minx
y_shape = maxy - miny

gradient_distances = np.zeros(dtype=np.float64, shape=(y_shape, x_shape, len(panos_ids)))

for sat, distances in tiles_values.iterrows():
    splited_path=sat.split("/")
    if len(splited_path)==7 and splited_path[3][0:3] == "all":
        zoom = splited_path[4]
        x = splited_path[5]
        y = splited_path[6].split(".")[0]
        for i, dist in enumerate(distances):
            gradient_distances[ int(y)-miny, int(x)-minx, i] = float(dist)

        # tile_ids.append(
        #     str(x) + "_" + str(y) + "_" + str(ZOOM-1)
        # )
    # else:
    #     tile_ids.append("None")
print("SHAPE")
print(gradient_distances[:,:,0])
print(gradient_distances)



# # tiles_values["tile_id"] = tile_ids
# print(tiles_values)
# # print(tiles_values["panos/autumn/Wilanow/PIXPRO/straighten/001_1.JPG"])
# # print(tiles_values["tile_id"])
# # print(tiles_values[4]) #= tiles_values[4].astype(float)
# # print(max(tiles_values[4]))
# f = None
# with open(f"./tools/geojson/tiles-{ZOOM-1}-256.geojson") as f:
#     f = json.load(f)

# # print(tiles_values)

# PANO = "panos/autumn/Ursynow/PIXPRO/straighten/062_1.JPG"

prev_id = 0

for i, pano in enumerate(panos_ids):
    pano_short = pano.split("/")[-1]
    print("Starting pano: ", pano_short)

    locations = pd.read_csv(root_dir + "/locations/all.csv", index_col=0)
    gradient_img = np.zeros(dtype=np.float64, shape=(y_shape, x_shape))
    loc_probability = np.ones(dtype=np.float64, shape=(y_shape, x_shape))

    np.clip(gradient_distances[:,:,i], 0, 1, out=gradient_img)
    loc_probability = loc_probability - gradient_distances
    # gradient_img = gradient_img**2
    # print(gradient_img)
    id = int(pano_short.split("_")[0])

    if prev_id != id:
        group = folium.FeatureGroup("MARKER " + str(id), show=False).add_to(m)

        folium.Marker(
        location=[locations.iloc[id-1]["latitude"], locations.iloc[id-1]["longitude"]],
        tooltip="ID "+str(id),
        icon=folium.Icon(color="red"),
        ).add_to(group)

    folium.raster_layers.ImageOverlay(
        image=gradient_img,
        name=pano_short,
        bounds=[[minlat, minlon], [maxlat, maxlon]],
        opacity=0.9,
        colormap=lambda x: (0,0,1-x, 1-x/2),
        show=False
    ).add_to(m)



folium.LayerControl().add_to(m)


m.save("results/heatmap.html")