import json
from SplitFileGenerator import SplitFileGenerator

sat_descriptors_subpath = "descriptors/sat/"
pano_descriptors_subpath = "descriptors/panos/"
distances_subpath = "distances/"
accuracy_subpath = "accuracy/"
charts_subpath = "charts/"


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def load_configs(path):
        configs = []
        with open(path, "r") as f:
            configs_json = json.load(f)
            for config in configs_json:

                dataset = config["dataset"]

                if config["name"] == "auto":
                    config["name"] = (
                        config["model"]
                        + "_"
                        + config["weights_name"]
                        + "_"
                        + "_".join(dataset["districts"])
                        + "_"
                        + "_".join(dataset["cameras"])
                        + "_"
                        + "_".join(dataset["seasons"])
                        + "_"
                        + dataset["modification"]
                        + "_"
                        + "crop"
                        + "_"
                        + "_".join([str(c) for c in dataset["crop"]])
                        + "_"
                        + dataset["angle"]
                        + "_"
                        + str(dataset["zoom_level"])
                        + "zl"
                    )

                if config["dataset"]["splits"] == "auto":
                    sfg = SplitFileGenerator(config)
                    sfg.generate()
                    config["dataset"]["splits"] = sfg.get_filename()
                else:
                    print(
                        "Attributes for dataset in config file will be ignored since split file is provided."
                    )

                results = config["results"]
                rasults_prefix_path = results["path"]

                results["descriptors"]["sat_save_path"] = (
                    rasults_prefix_path
                    + sat_descriptors_subpath
                    + config["name"]
                    + ".csv"
                )

                results["descriptors"]["pano_save_path"] = (
                    rasults_prefix_path
                    + pano_descriptors_subpath
                    + config["name"]
                    + ".csv"
                )

                results["distances"]["save_path"] = (
                    rasults_prefix_path + distances_subpath + config["name"] + ".csv"
                )

                results["accuracy"]["save_path"] = (
                    rasults_prefix_path
                    + accuracy_subpath
                    + results["accuracy"]["filename"]
                )

                results["charts"]["save_path"] = rasults_prefix_path + charts_subpath
            configs.append(config)

        return configs


if __name__ == "__main__":

    c = Config("./configs/config.json")
    print(c.top_k)

    # configs = load_configs("./configs/config.json")
    # print(configs)
