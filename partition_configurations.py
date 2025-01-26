from dataset_generators.parameters import Parameters


def construct_partition_configuration(parameters_structure, constructor, configuration, distribution_dictionary):

    configuration_parts = configuration.split("_")
    solution_type_ = determine_partition_solution_type(configuration)

    partitioner_parameters = Parameters(parameters_structure, "partitioner")
    partitioner_parameters = construct_base_partition_configuration(partitioner_parameters, constructor, distribution_dictionary, solution_type_)
    partitioner_parameters = construct_experiment_partition_configuration(partitioner_parameters, constructor, configuration_parts)

    parameters = {"partitioner": partitioner_parameters.get()}

    return parameters


def construct_base_partition_configuration(partitioner_parameters, constructor, distribution_dictionary, solution_type_):

    if solution_type_ != "context":

        partitioner_parameters.add("maximum_observation_count", 2000, modality="image")
        partitioner_parameters.add("partition_maxima", [(None, None), (None, None), (None, None)], modality="image")
        partitioner_parameters.add("distribution", distribution_dictionary["image"], modality="image")

        partitioner_parameters.add("maximum_observation_count", 400, modality="audio")
        partitioner_parameters.add("partition_maxima", [(None, None), (None, None), (None, None)], modality="audio")
        partitioner_parameters.add("distribution", distribution_dictionary["audio"], modality="audio")

        partitioner_parameters.add("partition_ratio", [0.5, 0.2, 0.3])
        partitioner_parameters.add("remainder_partition", 0)
        partitioner_parameters.add("priority_partition", "test")
        partitioner_parameters.add("semi_priority_partition", "train")

    else:

        partitioner_parameters.add("maximum_observation_count", 10000, modality="image")
        partitioner_parameters.add("partition_maxima", [(None, None), (None, None), (None, None)], modality="image")

        partitioner_parameters.add("maximum_observation_count", 10000, modality="audio")
        partitioner_parameters.add("partition_maxima", [(None, None), (None, None), (None, None)], modality="audio")

        partitioner_parameters.add("partition_ratio", [0.7, 0.1, 0.2])
        partitioner_parameters.add("remainder_partition", 0)
    return partitioner_parameters


# Configuration: {base name}_{parameter inclusion or exclusion}_{maximum attribute size}_{version}

def construct_experiment_partition_configuration(partitioner_parameters, constructor, configuration_parts):

    if (configuration_parts[0] == "attribute" and "1" in configuration_parts[1]) or (configuration_parts[0] == "all" and "1" not in configuration_parts[1]):

        partitioner_parameters.add("balance_sex", True)
        partitioner_parameters.add("sex_weight", 5.0)

    if (configuration_parts[0] == "attribute" and "2" in configuration_parts[1]) or (configuration_parts[0] == "all" and "2" not in configuration_parts[1]):

        partitioner_parameters.add("balance_age", True)
        partitioner_parameters.add("age_weight", 5.0)

    if (configuration_parts[0] == "attribute" and "3" in configuration_parts[1]) or (configuration_parts[0] == "all" and "3" not in configuration_parts[1]):

        partitioner_parameters.add("balance_subspecies", True)
        partitioner_parameters.add("subspecies_weight", 3.0)

    if (configuration_parts[0] == "attribute" and "4" in configuration_parts[1]) or (configuration_parts[0] == "all" and "4" not in configuration_parts[1]):

        if "xenocanto" in constructor.database_name:

            partitioner_parameters.add("balance_vocalisation", True, modality="audio")
            partitioner_parameters.add("vocalisation_zoom", [""], modality="audio")
            partitioner_parameters.add("vocalisation_weight", [3.0], modality="audio")

    if (configuration_parts[0] == "context" and "1" in configuration_parts[1]) or (configuration_parts[0] == "all" and "5" not in configuration_parts[1]):

        partitioner_parameters.add("balance_location", True)
        partitioner_parameters.add("location_weight", [2.0])
        partitioner_parameters.add("location_zoom", [(10.0, 10.0)])
        partitioner_parameters.add("location_offset", [(0.0, 0.0)])

    if (configuration_parts[0] == "context" and "2" in configuration_parts[1]) or (configuration_parts[0] == "all" and "6" not in configuration_parts[1]):

        partitioner_parameters.add("balance_date", True)
        partitioner_parameters.add("date_weight", [2.0])
        partitioner_parameters.add("date_zoom", [90])
        partitioner_parameters.add("date_offset", [0])

    if (configuration_parts[0] == "context" and "3" in configuration_parts[1]) or (configuration_parts[0] == "all" and "7" not in configuration_parts[1]):

        partitioner_parameters.add("balance_time", True)
        partitioner_parameters.add("time_weight", [1.0])
        partitioner_parameters.add("time_zoom", [180])
        partitioner_parameters.add("time_offset", [0])

    if configuration_parts[0] == "attribute" or configuration_parts[0] == "context" or configuration_parts[0] == "all":

        partitioner_parameters.add("maximum_attribute_size", int(configuration_parts[2]))

    return partitioner_parameters


def determine_partition_solution_type(configuration):

    if "random" in configuration:
        solution_type_ = "control"

    elif "only_context" in configuration:
        solution_type_ = "context"

    else:
        solution_type_ = "best"

    return solution_type_
