import csv
import matplotlib.pyplot as plt

# Read data from CSV file
data = {}
with open("./results/results.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        key = row[0]
        values = list(map(float, row[1:]))
        data[key] = values

# Plot data
for key, values in data.items():
    plt.plot(values, label=key)

plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Recall Values")
plt.legend()
plt.show()
