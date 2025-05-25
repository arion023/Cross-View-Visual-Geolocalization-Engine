import folium
import pandas as pd
import json
import mercantile
import geopandas as gpd
import numpy as np


ZOOM = 18
root_dir = "D:\datasets\Warsaw"
tileurl = "https://ukosne.um.warszawa.pl/Data/2022/{z}/{x}/{y}.jpg"


tileArcGIS = folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Esri Satellite",
    overlay=False,
    control=True,
)

m = folium.Map(
    (52.22036678017412, 21.011035209401882),
    zoom_start=17,
    tiles=tileurl,
    attr="Warszawa-ukosna",
)

ZL = 20
RESOLUTION = 256

name = f"tiles-{ZL}-{RESOLUTION}"

tiles = gpd.read_file(f"./tools/geojson/tiles-pw-{ZL}-{RESOLUTION}.geojson").to_crs(
    "EPSG:4326"
)  # Web Mercator


folium.GeoJson(tiles, name=name).add_to(m)

# #pw cords
tile = mercantile.tile(21.011321621562498, 52.22030319341392, ZL)
print(tile) #Tile(x=104, y=131, z=19)
bounds_obj = mercantile.bounds(tile)
print(bounds_obj)
bounds = [(bounds_obj[1], bounds_obj[0]), (bounds_obj[3], bounds_obj[2])] #[(52.22030319341392, 21.011321621562498), (52.3030319341392, 21.111321621562498)]
print(bounds)

folium.Rectangle(
    bounds=bounds,
    color='red',
    fill=False,  # całkowicie przezroczyste wypełnienie
    weight=6  # grubość obramowania
).add_to(m)

folium.LayerControl().add_to(m)

m.save(f"results/maps/{name}.html")

print(f"Map saved as {name}.html")

# import io
# from PIL import Image

# img_data = m._to_png(10)
# img = Image.open(io.BytesIO(img_data))
# img.save(f'results/{name}.png')
