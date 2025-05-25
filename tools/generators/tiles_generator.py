import geopandas as gpd
import mercantile
from shapely.geometry import box

from geojson import Polygon, FeatureCollection, Feature
import geojson
import random
import pandas as pd

ZOOM = 19

# 1. Wczytaj dane geograficzne Warszawy (np. z pliku GeoJSON)
warszawa = gpd.read_file("./tools/geojson/warszawa.geojson").to_crs(
    "EPSG:4326"
)  # Web Mercator

warszawa_dzielnice = gpd.read_file("./tools/geojson/warszawa-dzielnice.geojson").to_crs(
    "EPSG:4326"
)  # Web Mercator


print(warszawa.total_bounds)

# 2. Oblicz bounding box
minx, miny, maxx, maxy = [20.996436146396395, 52.20946869125238, 21.028251195825906, 52.226324693027955 ] #warszawa.total_bounds
tiles = list(mercantile.tiles(minx, miny, maxx, maxy, ZOOM))

print(tiles[0].x)

tile_geoms = [box(*mercantile.bounds(t)) for t in tiles]
print(list(tile_geoms[0].exterior.coords))


features = []
for geometry, tile in zip(tile_geoms, tiles):
    feature = Feature(
        geometry=Polygon([list(geometry.exterior.coords)]),
        id=str(tile.x) + "_" + str(tile.y) + "_" + str(tile.z),
        properties={"name": "random"},
    )
    features.append(feature)

# with open(f"./tools/geojson/tiles-{ZOOM}-256-values.csv", "w") as f:
#     f.write("tile_id,value\n")
#     for tile in tiles:
#         value = random.randint(0, 10)
#         f.write(f"{tile.x}_{tile.y}_{tile.z},{value}\n")

tiles_geojson = FeatureCollection(features)

with open(f"./tools/geojson/tiles-pw-{ZOOM}-256.geojson", "w") as f:
    geojson.dump(tiles_geojson, f)

print(len(tiles))
print(tiles[0])
print(mercantile.bounds(tiles[0]))


# import folium

# warszawa_tiles = gpd.read_file(f"./tools/geojson/tiles-{ZOOM}-256.geojson").to_crs(
#     "EPSG:4326"
# )  # Web Mercator


# tileArcGIS = folium.TileLayer(
#     tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
#     attr="Esri",
#     name="Esri Satellite",
#     overlay=False,
#     control=True,
# )

# tileurl = "https://ukosne.um.warszawa.pl/Data/2022/{z}/{x}/{y}.jpg"

# m = folium.Map(
#     (52.22983601573939, 21.011703206081123),
#     zoom_start=18,
#     tiles=tileurl,
#     attr="Warszawa-ukosna",
# )

# tiles_values = pd.read_csv(f"./tools/geojson/tiles-{ZOOM}-256-values.csv")

# import json

# f = None
# with open(f"./tools/geojson/tiles-{ZOOM}-256.geojson") as f:
#     f = json.load(f)

# # print(tiles_values)

# folium.Choropleth(
#     geo_data=f,
#     name="choropleth",
#     data=tiles_values,
#     columns=["tile_id", "value"],
#     key_on="feature.id",
#     fill_color="PuBu",
#     fill_opacity=0.6,
#     line_opacity=0.2,
#     legend_name="Values",
# ).add_to(m)


# # folium.GeoJson(warszawa_tiles, name=f"Tiles {ZOOM}").add_to(m)


# folium.LayerControl().add_to(m)


# m.save("map.html")
