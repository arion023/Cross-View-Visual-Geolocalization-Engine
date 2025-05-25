import csv
import matplotlib.pyplot as plt

# Read data from CSV file
data = {}
with open("./../../results/distances/tmp.csv", "r") as file:
    reader = csv.reader(file)
    header = next(reader)
    counter = 0
    for row in reader:
        counter += 1
        key = row[0]
        values = list(map(float, row[1:]))
        data[key] = values
        # if counter > 20:
        #     break
# Create subplots
num_plots = len(data)
print(num_plots)
fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(2, 140))
fig.subplots_adjust(top=0.94, wspace=0.3, hspace=0.3)

if num_plots == 1:
    axes = [axes]

for ax, (key, values) in zip(axes, data.items()):
    ax.hist(values, bins=20, alpha=0.7)
    ax.set_title(f"Distribution for {key}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

# plt.tight_layout()
plt.savefig("tmp_distances.png", bbox_inches="tight", dpi=150)
