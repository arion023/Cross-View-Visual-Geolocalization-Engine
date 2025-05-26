import argparse, sys
import matplotlib.pyplot as plt
sys.path.append("./evaluation")
from structures.InputData import InputDataContainer

plt.rcParams.update({
    'font.size': 18,           # Increase font size globally
    'axes.titlesize': 20,      # Title font size
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})

def generate_recall_chart(input_data_containter, k=100, plot=0, all=False):

    # Read data from CSV file
    accuracies = {}  # {"name": {"percentage_values": [], "top100_values": []}}

    for input_data in input_data_containter:
        input_data.results.load()
        # print("loaded debug")
        if not all and not input_data.results.show:
            continue
        name = input_data.results.shortcut
        accuracies[name] = {}
        accuracies[name][
            "percentage_values"
        ] = input_data.results.accuracy.percentage_values

        accuracies[name][
            "top100_values"
        ] = input_data.results.accuracy.top100_values

    if plot == 0:
        fig, axs = plt.subplots(1, 2)

        axs[0].set_title(f"Top K%")
        axs[0].set_xlabel("Procent zbioru danych uznany za poprawne dopasowanie [%]")
        axs[0].set_ylabel("Dokładność [%]")
        axs[0].xticks(range(1, k + 1))

        axs[1].set_title(f"Top K")
        axs[1].set_xlabel("Liczba K najbliższych zdjęć uznawana za poprawne dopasowanie")
        axs[1].set_ylabel("Dokładność [%]")
        axs[1].xticks(range(0, k + 1))


        # Plot data
        for key, values in accuracies.items():
            axs[0].plot(values["percentage_values"][:k+1]*100, label=key)
            axs[1].plot(values["top100_values"][:k+1]*100, label=key)
    if plot == 1:
        plt.figure(figsize=(10, 10))
        plt.title(f"Top K%")
        plt.xlabel("Procent K zbioru danych uznany za poprawne dopasowanie [%]")
        plt.ylabel("Dokładność [%]")
        plt.xticks(range(0, k + 1))

        for key, values in accuracies.items():
            plt.plot(values["percentage_values"][:k+1]*100, label=key)
    elif plot == 2:
        plt.figure(figsize=(10, 10))
        plt.title(f"Top K")
        plt.xlabel("Liczba K najbliższych zdjęć uznawana za poprawne dopasowanie")
        plt.ylabel("Dokładność [%]")
        plt.xticks(range(0, k + 1))
        for key, values in accuracies.items():
            plt.plot(values["top100_values"][:k+1]*100, label=key)
    elif plot == 3:
        plt.figure(figsize=(10, 10))
        plt.title(f"Top K vs Top K%")
        plt.xlabel("Liczba K lub K% najbliższych zdjęć uznawana za poprawne dopasowanie")
        plt.ylabel("Dokładność [%]")
        plt.xticks(range(0, k + 1))

        for key, values in accuracies.items():
            plt.plot(values["percentage_values"][:k+1]*100, label=key+"-Top-K%")
            plt.plot(values["top100_values"][:k+1]*100, label=key+"-Top-K")


    rang = [1, 5 ,10]
    print("\tTOP K% VALUES\t\tTOP K VALUES")
    for key, values in accuracies.items():
        print(f"{key}:")
        for k in rang:
            print(f"\tK%={k}%:\t{values['percentage_values'][k]*100:.2f}\t\tK={k}:\t{values['top100_values'][k]*100:.2f}")
            print("")




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

    parser.add_argument(
        "-k",
        type=int,
        required=False,
        help="How much results to show in the chart. Default is 100.",
    )

    parser.add_argument(
        "-p",
        type=int,
        required=False,
        help="Plot type: 0 - both, 1 - percentage, 2 - top100. Default is 0.",
    )

    parser.add_argument(
        "-a",
        type=bool,
        required=False,
        help="Plot all results",
    )


    args = parser.parse_args()

    if args.config is not None:
        print("Config loaded from: ", args.config)
        input_data_container = InputDataContainer(args.config)
    else:
        raise Exception("No config file.")

    a = False
    if args.a is not None:
        a = args.a

    generate_recall_chart(input_data_container, args.k if args.k is not None else 100, plot=args.p if args.p is not None else 0, all=a)
