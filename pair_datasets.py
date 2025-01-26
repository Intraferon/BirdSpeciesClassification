from dataset_generators.parameters import Parameters
from pair_configurations import *
from datasets import *
from subsets import *
from graphing import Grapher
import pprint
from dataset_generators.constructor import *


def save_configuration(constructor, parameters_structure, configuration):
    parameters = construct_pair_configuration(parameters_structure, configuration)
    constructor.save_configuration(configuration, parameters, dataset_type_="pair")


def create_pair(constructor, configuration, partition_solution_type, partition, solution_type_, image_partition_configuration=None, audio_partition_configuration=None, partition_configuration=None):
    if solution_type_ == "source":
        constructor.create_pair_experiment_source(partition_configuration, partition_solution_type, partition)
    elif solution_type_ == "control":
        constructor.create_pair_experiment_control(configuration, image_partition_configuration, audio_partition_configuration, partition_solution_type, partition)
    else:
        constructor.conduct_pair_experiment(configuration, image_partition_configuration, audio_partition_configuration, partition_solution_type, partition)


# Configuration: {base name}_{parameter inclusion or exclusion}_{disqualify}_{invert}_{version}

def get():
    constructor = get_inaturalist_and_xenocanto_1000_dataset_constructor()
    partition_list = ["train", "validation", "test"]

    parameters_structure = {"image-audio": {partition: None for partition in partition_list}}

    return constructor, parameters_structure


# Configuration: {base name}_{parameter inclusion or exclusion}_{version}

if __name__ == "__main__":

    image_partition_configuration_ = "all_0_3_0"
    audio_partition_configuration_ = "all_0_3_0"
    partition_configuration_ = None
    partition_solution_type_ = "best"

    partition_list_ = ["train", "validation", "test"]

    for partition_ in partition_list_:
        pair_configuration_ = "all_0_0"

        constructor_, parameters_structure_ = get()
        pair_solution_type_ = determine_pair_solution_type(pair_configuration_)

        save_configuration(constructor_, parameters_structure_, pair_configuration_)
        # create_pair(constructor_, pair_configuration_, partition_solution_type_, partition_, pair_solution_type_,
        #             image_partition_configuration=image_partition_configuration_, audio_partition_configuration=audio_partition_configuration_,
        #             partition_configuration=partition_configuration_)

        constructor_.evaluate_pair_experiment(pair_configuration_, image_partition_configuration_, audio_partition_configuration_)
