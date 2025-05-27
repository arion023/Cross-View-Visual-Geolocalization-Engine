import matplotlib.pyplot as plt
import csv

# DIST_FILE = "C:/Users/masik/Workspace/inzynierka/repos/BEng-MK/results/distances/CVM-NET-I_original_Warsaw_Random_1000_PIXPRO_all_seasons_straighten_crop_600_1000_orto_19zl.csv"
DIST_FILE = "C:/Users/masik/Workspace/inzynierka/repos/BEng-MK/results/distances/CVM-NET-I_finetune_districts_after.csv"

correct_distances = []
wrong_distances = []

with open(DIST_FILE, 'r') as file:
    writer = csv.reader(file)
    header = next(writer)[1:]
    unqiue_sats = {}
    for i, sat in enumerate(header):
        if sat not in unqiue_sats.keys():
            unqiue_sats[sat] = i

    for i, row in enumerate(writer):
        # Assuming the first column is the key and the rest are values
        distances = list(map(float, row[1:]))
        correct_distance = distances[i]
        correct_distances.append(correct_distance)
        for idx in unqiue_sats.values():
            if distances[idx] != correct_distance:
                wrong_distances.append(distances[idx])


plt.rcParams.update({
    'font.size': 18,           # Increase font size globally
    'axes.titlesize': 20,      # Title font size
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})

# Plot the histograms
plt.figure(figsize=(10, 8))
plt.hist(wrong_distances, bins=20, color='red', alpha=0.5, density=True, label='Rozkład dystansów dla błędnych dopasowań')
plt.hist(correct_distances, bins=20, color='blue', alpha=0.5, density=True, label='Rozkład dystansów dla parawidłowych dopasowań')

# Add labels and legend
plt.xlim(0, 3.5)
plt.xlabel('Dystans')
plt.ylabel('Częstość')
plt.title('Histogram dystansów w przestrzeni cech')
plt.legend()

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()
