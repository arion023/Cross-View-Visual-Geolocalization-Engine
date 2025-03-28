import argparse, sys

sys.path.append(".")


import matplotlib.pyplot as plt

from structures.InputData import InputDataContainer


def generate_recall_chart(input_data_containter):

    # Read data from CSV file
    accuracies = {}  # {"name": {"percentage_values": [], "top100_values": []}}

    for input_data in input_data_containter:
        input_data.results.load()
        accuracies[input_data.name] = {}
        accuracies[input_data.name][
            "percentage_values"
        ] = input_data.results.accuracy.percentage_values

        accuracies[input_data.name][
            "top100_values"
        ] = input_data.results.accuracy.top100_values

    fig, axs = plt.subplots(2, 1)

    axs[0].set_title("Percentage Recall Values")
    axs[0].set_xlabel("Percent of dataset considered as correct match [%]")
    axs[0].set_ylabel("Percent of correct matches [%]")

    axs[1].set_title("Top 100 Recall Values")
    axs[1].set_xlabel("N nearest photos considered as correct match")
    axs[1].set_ylabel("Percent of correct matches [%]")

    # Plot data
    for key, values in accuracies.items():
        axs[0].plot(values["percentage_values"], label=key)
        axs[1].plot(values["top100_values"], label=key)

    plt.legend()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Arguments for analyzing and displaying results from siamese models."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the config file with other args. It overrides args passed in command line",
    )

    args = parser.parse_args()

    if args.config is not None:
        print("Config loaded from: ", args.config)
        input_data_container = InputDataContainer(args.config)
    else:
        raise Exception("No config file.")

    generate_recall_chart(input_data_container)
