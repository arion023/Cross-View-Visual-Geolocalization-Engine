import csv

filename = "D:/datasets/Warsaw/splits/val-19zl.csv"

sat_prefix = "sat/waw/2022/orto/21/"

panos_prefix = "panos/"

season = "autumn/"
# or
# season = "winter/"

# version =  "raw"
# or
version = "modified/strighten"

sat_format = "JPG"
ground_format = "JPG"

# camera types: "PIXPRO" or "Theta"
# districts = {
#     "Wilanow": ((1, 30), ("PIXPRO",)),
#     "Center": ((31, 50), ("PIXPRO",)),
#     "Ursynow": ((51, 70), ("PIXPRO",)),
# }
# districts = {"Center": ((31, 50), ("PIXPRO",)), "Ursynow": ((51, 70), ("PIXPRO",))}

districts = {"Wilanow": ((1, 30), ("PIXPRO",))}

rows = []

for dist, params in districts.items():
    ids = params[0]
    camera_list = params[1]
    for camera in camera_list:
        panos_full_prefix = (
            panos_prefix + season + dist + "/" + camera + "/" + version + "/"
        )

        for i in range(ids[0], ids[1] + 1):
            row = []
            row.append(sat_prefix + f"{i:03}." + sat_format)
            row.append(panos_full_prefix + f"{i:03}_1." + ground_format)
            row.append("empty")
            rows.append(row)

            row = []
            row.append(sat_prefix + f"{i:03}.JPG")
            row.append(panos_full_prefix + f"{i:03}_2." + ground_format)
            row.append("empty")
            rows.append(row)


with open(filename, mode="w", newline="") as file:
    # Create a csv.writer object
    writer = csv.writer(file)
    # Write data to the CSV file
    writer.writerows(rows)
