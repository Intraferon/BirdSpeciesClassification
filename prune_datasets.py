import numpy as np
from datasets import *
from dataset_generators.parameters import Parameters


def prune_inaturalist_and_xenocanto_dataset(version):
    parameters = {}
    parameters_structure = {"image": {"train-validation-test": None},
                            "audio": {"train-validation-test": None}}
    pruner_parameters = Parameters(parameters_structure, "pruner")
    version_info = ""
    if version == 1:
        pruner_parameters.add("only_common_species", True)
        version_info = "Pruned to contain only species common to all the source datasets"
    if version == 2:
        pruner_parameters.add("location_uncertainty_threshold", 100, modality="image")
        pruner_parameters.add("duration_threshold", "3:00", modality="audio")
        pruner_parameters.add("observer_minimum_threshold", 20, modality="audio")
        pruner_parameters.add("observation_minimum_threshold", 20, modality="audio")
        pruner_parameters.add("observer_minimum_threshold", 20, modality="image")
        pruner_parameters.add("observation_minimum_threshold", 20, modality="image")
        version_info = "Pruned by duration, observation count and observer count"
    if version == 3:
        pruner_parameters.add("require_sex_detail", True)
        pruner_parameters.add("require_age_detail", True)
        pruner_parameters.add("require_subspecies_detail", True)
        pruner_parameters.add("partition_weight", 1, modality="audio")
        pruner_parameters.add("partition_weight", 2, modality="image")
        pruner_parameters.add("detail_tier", {"female": 0, "young": 0, "male": 0, "adult": 0, "subspecies": 1, "uncertain_subspecies": 1})
        pruner_parameters.add("detail_low_resolution", {"female": False, "young": False, "male": False, "adult": False, "subspecies": False, "uncertain_subspecies": False})
        pruner_parameters.add("detail_weight", {"female": 2, "young": 2, "male": 1, "adult": 1, "subspecies": 2, "uncertain_subspecies": 1})
        pruner_parameters.add("detail_penalty", {"female": 10, "young": 10, "male": 5, "adult": 5, "subspecies": 0, "uncertain_subspecies": 0})
        pruner_parameters.add("maximum_species_count", 1000)
        version_info = "Pruned by detail"
    parameters["pruner"] = pruner_parameters.get()
    get_inaturalist_and_xenocanto_dataset_constructor().update_database(parameters, version_info)


def prune_inaturalist_and_xenocanto_derivative_dataset():

    sorted_species_list = get_inaturalist_and_xenocanto_dataset_constructor().get_sorted_species()
    species_list = sorted_species_list[:300]

    version_info = "Pruned to contain only the 300 most detailed species from iNaturalist and Xeno Canto"
    parameters = {}

    parameters_structure = {"image": {"train-validation-test": None},
                            "audio": {"train-validation-test": None}}
    pruner_parameters = Parameters(parameters_structure, "pruner")
    pruner_parameters.add("species", species_list)
    parameters["pruner"] = pruner_parameters.get()
    get_inaturalist_and_xenocanto_300_dataset_constructor().update_database(parameters, version_info)


if __name__ == "__main__":
    prune_inaturalist_and_xenocanto_derivative_dataset()