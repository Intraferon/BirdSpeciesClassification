from dataset_generators.parameters import Parameters


def construct_pair_configuration(parameters_structure, configuration):

    configuration_parts = configuration.split("_")

    pairer_parameters = Parameters(parameters_structure, "pairer")
    pairer_parameters = construct_experiment_pair_configuration(pairer_parameters, configuration_parts)

    parameters = {"pairer": pairer_parameters.get()}

    return parameters


# Configuration: {base name}_{parameter inclusion or exclusion}_{version}

def construct_experiment_pair_configuration(pairer_parameters, configuration_parts):

    if (configuration_parts[0] == "attribute" and "1" in configuration_parts[1]) or (configuration_parts[0] == "all" and "1" not in configuration_parts[1]):

        pairer_parameters.add("pair_sex", True)
        pairer_parameters.add("sex_weight", 3.0)

    if (configuration_parts[0] == "attribute" and "2" in configuration_parts[1]) or (configuration_parts[0] == "all" and "2" not in configuration_parts[1]):

        pairer_parameters.add("pair_age", True)
        pairer_parameters.add("age_weight", 3.0)

    if (configuration_parts[0] == "attribute" and "3" in configuration_parts[1]) or (configuration_parts[0] == "all" and "3" not in configuration_parts[1]):

        pairer_parameters.add("pair_subspecies", True)
        pairer_parameters.add("subspecies_weight", 2.0)

    if (configuration_parts[0] == "context" and "1" in configuration_parts[1]) or (configuration_parts[0] == "all" and "4" not in configuration_parts[1]):

        pairer_parameters.add("pair_location", True)
        pairer_parameters.add("location_weight", 3.0)
        pairer_parameters.add("location_cap", 5000.0)

    if (configuration_parts[0] == "context" and "2" in configuration_parts[1]) or (configuration_parts[0] == "all" and "5" not in configuration_parts[1]):

        pairer_parameters.add("pair_date", True)
        pairer_parameters.add("date_weight", 3.0)
        pairer_parameters.add("date_cap", 120)

    if (configuration_parts[0] == "context" and "3" in configuration_parts[1]) or (configuration_parts[0] == "all" and "6" not in configuration_parts[1]):

        pairer_parameters.add("pair_time", True)
        pairer_parameters.add("time_weight", 1.0)
        pairer_parameters.add("time_cap", 360)

    return pairer_parameters


def determine_pair_solution_type(configuration):

    if "source" in configuration:
        solution_type_ = "source"

    elif "random" in configuration:
        solution_type_ = "control"

    else:
        solution_type_ = "best"

    return solution_type_