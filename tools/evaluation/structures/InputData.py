import json

from structures.Dataset import Dataset
from structures.Results import Results

from munch import Munch


class InputDataContainer:
    def __init__(self, config_path):
        self.inputdata_list = []
        with open(config_path, "r") as cf:
            json_configs = json.load(cf)
            for config in json_configs:
                self.inputdata_list.append(InputDataEval(config))
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx < len(self.inputdata_list):
            self.current_idx += 1
            return self.inputdata_list[self.current_idx - 1]
        else:
            self.current_idx = 0
            raise StopIteration


class InputDataEval:

    # crop is tuple (top_crop, bottom_crop), if there is no crop, there should be (None, None)
    def __init__(self, config_dict):
        self.generated_name = False
        self.__dict__.update(config_dict)
        self.update()

    def update(self):
        self.dataset = Dataset(self.dataset, self)
        self.update_name()
        self.results = Results(self.results, self)
        self.charts = Munch(self.charts)

    def update_name(self):
        if self.name == "auto":
            if not self.dataset.generated_splitfile:
                raise ValueError(
                    "Auto name generation is not possible without splitfile generation. Name must be custom in that way, because splifile is given also custom."
                )

            self.generated_name = True
            self.name = (
                self.model
                + "_"
                + self.weights_name
                + "_"
                + "_".join(self.dataset.districts)
                + "_"
                + "_".join(self.dataset.cameras)
                + "_"
                + "_".join(self.dataset.seasons)
                + "_"
                + self.dataset.modification
                + "_"
                + "crop"
                + "_"
                + "_".join([str(c) for c in self.dataset.crop])
                + "_"
                + self.dataset.angle
                + "_"
                + str(self.dataset.zoom_level)
                + "zl"
            )
