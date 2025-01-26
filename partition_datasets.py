from dataset_generators.parameters import Parameters
from partition_configurations import *
from datasets import *
from subsets import *
from graphing import Grapher
import sys
import pprint


def get_distribution(parameters_structure, distribution_maximum_dictionary, constructor, partition):
    distribution_dictionary = {}

    for modality in parameters_structure:

        distribution = {}

        dataset = constructor.get_dataset_handle(0, modality, partition)
        species_list = constructor.read_species_list(dataset)

        for species in species_list:
            observation_count = constructor.read_observation_count(dataset, species)
            distribution[species] = observation_count

        maximum_observation_count = distribution_maximum_dictionary[modality]

        for species in distribution:
            distribution[species] = float(distribution[species] / maximum_observation_count)

        distribution_dictionary[modality] = distribution

    return distribution_dictionary


def save_configuration(parameters_structure, constructor, configuration, distribution_dictionary):
    parameters = construct_partition_configuration(parameters_structure, constructor, configuration, distribution_dictionary)
    constructor.save_configuration(configuration, parameters, dataset_type_="partition")


def create_partition(subset, constructor, configuration, modality, partition, solution_type_):
    if solution_type_ == "control":
        constructor.create_partition_experiment_control(subset, configuration, modality, partition)
    elif solution_type_ == "context":
        constructor.create_partition_experiment_context(configuration, modality, partition)
    else:
        constructor.conduct_partition_experiment(subset, configuration, modality, partition)


def evaluate_partition(constructor, configuration, modality, partition):
    constructor.evaluate_partition_experiment(configuration, modality, partition)


def compare_partition(constructor, target_configuration, reference_configuration, modality, partition):
    constructor.compare_partition_experiment(target_configuration, reference_configuration, modality, partition)


def get():
    constructor = get_inaturalist_and_xenocanto_1000_dataset_constructor()
    partition = "train-validation-test"
    modality_list = ["image", "audio"]
    distribution_maximum_list = [25000, 800]

    parameters_structure = {modality: {partition: None} for modality in modality_list}
    distribution_maximum_dictionary = {modality: distribution_maximum_list[i] for i, modality in enumerate(modality_list)}

    return constructor, partition, parameters_structure, distribution_maximum_dictionary


# {base name}_{parameter inclusion or exclusion}_{maximum attribute size}_{version}

if __name__ == "__main__":

    subset_ = get_xenocanto()
    modality_ = "audio"
    configuration_ = "all_0_3_0"

    constructor_, partition_, parameters_structure_, distribution_maximum_dictionary_ = get()
    solution_type_ = determine_partition_solution_type(configuration_)

    distribution_dictionary_ = get_distribution(parameters_structure_, distribution_maximum_dictionary_, constructor_, partition_)
    save_configuration(parameters_structure_, constructor_, configuration_, distribution_dictionary_)
    # create_partition(subset_, constructor_, configuration_, modality_, partition_, solution_type_)
    evaluate_partition(constructor_, configuration_, modality_, partition_)