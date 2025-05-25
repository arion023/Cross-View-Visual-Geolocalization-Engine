import folium
import pandas as pd
import csv
import mercantile
import geopandas as gpd
import numpy as np


root_dir = "D:\datasets\Warsaw"
tileurl = "https://ukosne.um.warszawa.pl/Data/2022/{z}/{x}/{y}.jpg"



m = folium.Map(
    (52.22036678017412, 21.011035209401882),
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

tileCartodb = folium.TileLayer(
    tiles="cartodb positron",
    overlay=False,
    show=False
).add_to(m)


with open(root_dir + "/locations/all.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)
    group_wilanow = folium.FeatureGroup("wilanow").add_to(m)
    group_center = folium.FeatureGroup("center").add_to(m)
    group_ursynow = folium.FeatureGroup("ursynow").add_to(m)
    group_seq = folium.FeatureGroup("sequnces").add_to(m)

    for i, row in enumerate(reader):
        lat = float(row[1])
        lon = float(row[2])
        if i < 30:
            color="red"
            group=group_wilanow
        elif i < 50:
            color="blue"
            group=group_center
        elif i < 70:
            color="purple"
            group=group_ursynow
        else:
            color="orange"
            group=group_seq

        folium.Marker(
            (lat, lon),
            icon=folium.Icon(icon="camera",color=color),
            popup=f"{row[0]}",
        ).add_to(group)


for no in range(1, 6):
    color = "red"
    if no==4:
        color = "blue"
    if no==5:
        color = "purple"
    sequence_group = folium.FeatureGroup(name=f"S{no}").add_to(m)
    with open(root_dir + f"/locations/s{no}.csv", "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        sequence_coords = []
        for i, row in enumerate(reader):
            lat = float(row[1])
            lon = float(row[2])
            sequence_coords.append((lat, lon))
            folium.Marker(
                (lat, lon),
                icon=folium.Icon(icon="camera",color=color),
                popup=f"{row[0]}",
            ).add_to(sequence_group)
    folium.PolyLine(sequence_coords, color="blue", weight=4, opacity=1).add_to(sequence_group)

folium.LayerControl().add_to(m)

m.save(f"results/maps/locations.html")


