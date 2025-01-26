import math
import pprint

from graphing import *
from datetime import datetime

dataset_name_translator = {
    "xenocanto": "Xeno Canto",
    "inaturalist_image": "iNaturalist (Image)",
    "inaturalist_audio": "iNaturalist (Audio)",
    "observationorg": "Observation.org"
}

def parse_dataset_id(dataset_id):
    version_configuration = dataset_id[0]
    modality = dataset_id[1]
    partition = dataset_id[2]
    if len(dataset_id) > 3:
        dataset_type_ = dataset_id[3]
        solution_type_ = dataset_id[4]
    else:
        dataset_type_ = None
        solution_type_ = ""
    return version_configuration, modality, partition, dataset_type_, solution_type_


def parse_source_metadata_id(source_metadata_id, dataset_id):
    if source_metadata_id is not None:
        version = source_metadata_id[0]
        modality = source_metadata_id[1]
        source_partition = source_metadata_id[2]
    else:
        version = dataset_id[0]
        modality = dataset_id[1]
        source_partition = dataset_id[2]
    return version, modality, source_partition


def observation_frequency_distribution_graph(dataset_name_list, constructor_list, dataset_id_list):
    # Setup
    x_max = 0
    y_max = 0
    observation_frequency_dictionary_list = []
    for constructor, dataset_id in zip(constructor_list, dataset_id_list):
        observation_frequency_dictionary = {}
        version_configuration, modality, partition, dataset_type_, solution_type_ = parse_dataset_id(dataset_id)
        dataset = constructor.get_dataset_handle(version_configuration, modality, partition, dataset_type_=dataset_type_)
        species_list = constructor.read_species_list(dataset)
        x_max = max(x_max, len(species_list))
        for species in species_list:
            observation_count = constructor.read_observation_count(dataset, species, dataset_type_=dataset_type_, solution_type_=solution_type_)
            y_max = max(y_max, observation_count)
            observation_frequency_dictionary = update_frequency_dictionary_(species, observation_count, observation_frequency_dictionary)
        observation_frequency_dictionary_list.append(observation_frequency_dictionary)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    parameters = {
        "x_label": "Sorted Species",
        "y_label": "Number of Sightings",
        "y_scale": "log",
        "legend_location": "upper left",
        "x_min": 0,
        "x_max": 1000,
        "x_tick": 100,
        "y_max": 1000000,
        "y_min": 1,
        "line_colors": [colors[2], colors[0], colors[1]],
        "average_line_colors": [colors[2], colors[0], colors[1]]
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.frequency_distribution_line_graph(dataset_name_list, observation_frequency_dictionary_list)


def observation_total_frequency_distribution_graph(dataset_name_list, constructor_list, dataset_id_list):
    # Setup
    x_max = 0
    y_max = 0
    observation_frequency_dictionary_list = []
    partition_list = ["train", "validation", "test"]
    for constructor, dataset_id in zip(constructor_list, dataset_id_list):
        count_list = [0, 0, 0]
        observation_frequency_dictionary = {}
        version_configuration, modality, _, dataset_type_, solution_type_ = parse_dataset_id(dataset_id)
        i = 0
        for partition in partition_list:
            dataset = constructor.get_dataset_handle(version_configuration, modality, partition, dataset_type_=dataset_type_)
            species_list = constructor.read_species_list(dataset)
            x_max = max(x_max, len(species_list))
            for species in species_list:
                observation_count = constructor.read_observation_count(dataset, species, dataset_type_=dataset_type_, solution_type_=solution_type_)
                y_max = max(y_max, observation_count)
                observation_frequency_dictionary = update_frequency_dictionary_(species, observation_count, observation_frequency_dictionary)
                count_list[i] += observation_count
            i += 1
        observation_frequency_dictionary_list.append(observation_frequency_dictionary)
        print(count_list)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    parameters = {
        "x_label": "Sorted Species",
        "y_label": "Number of Sightings",
        "y_scale": "log",
        "legend_location": "upper left",
        "x_min": 0,
        "x_max": 1000,
        "x_tick": 100,
        "y_max": 10000,
        "y_min": 1,
        "line_colors": [colors[2], colors[0], colors[1]],
        "average_line_colors": [colors[2], colors[0], colors[1]]
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.frequency_distribution_line_graph(dataset_name_list, observation_frequency_dictionary_list)


def best_fitness_improvement_distribution_graph(dataset_name_list, BFI_lists, observation_count_lists):

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    parameters = {
        "x_label": "Species Size",
        "y_label": "Best Fitness Improvement",
        "y_scale": "linear",
        "legend_location": "upper left",
        "x_min": 10,
        "x_max": 100000,
        "x_scale": "log",
        "y_max": 1,
        "y_min": 0,
        "y_tick": 0.1,
        "line_colors": [colors[2], colors[0], colors[1], colors[3]],
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.scatter_plot_graph(dataset_name_list, BFI_lists, observation_count_lists)

def agreement_improvement_distribution_graph(dataset_name_list, BFI_lists, observation_count_lists):

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    parameters = {
        "x_label": "Species Size",
        "y_label": "Agreement Improvement",
        "y_scale": "linear",
        "legend_location": "upper left",
        "x_min": 0,
        "x_max": 1000,
        "x_tick": 100,
        "y_max": 1,
        "y_min": 0,
        "y_tick": 0.1,
        "line_colors": [colors[2], colors[0], colors[1], colors[3]],
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.scatter_plot_graph(dataset_name_list, BFI_lists, observation_count_lists)

def observation_recall_graph(dataset_name_list, observation_species_size_lists, observation_recall_lists):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    parameters = {
        "x_label": "Species Size",
        "y_label": "Recall",
        "legend_location": "lower right",
        "x_min": 0,
        "x_max": 120,
        "x_tick": 10,
        "y_max": 1,
        "y_min": 0,
        "y_tick": 0.1,
        "line_colors": [colors[3], colors[2], colors[0], colors[1]]
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.scatter_plot_graph(dataset_name_list, observation_recall_lists, observation_species_size_lists)


def attribute_distribution_graph(dataset_name_list, constructor_list, dataset_id_list):
    # Setup
    attribute_ratio_dictionary_list = []
    j = 0
    for constructor, dataset_id in zip(constructor_list, dataset_id_list):
        sex_ratio_dictionary = {}
        age_ratio_dictionary = {}
        subspecies_ratio_dictionary = {}
        vocalisation_ratio_dictionary = {}
        version_configuration, modality, partition, dataset_type_, solution_type_ = parse_dataset_id(dataset_id)
        dataset = constructor.get_dataset_handle(version_configuration, modality, partition, dataset_type_=dataset_type_)
        source_metadata = constructor.get_source_metadata_handle(version_configuration, modality, partition, dataset_type_=dataset_type_)
        species_list = constructor.read_species_list(dataset)
        for species in species_list:
            observation_count = constructor.read_observation_count(dataset, species, dataset_type_=dataset_type_, solution_type_=solution_type_)
            metadata_list = constructor.read_metadata_list(dataset, source_metadata, species, dataset_type_=dataset_type_, solution_type_=solution_type_)
            sex_ratio_dictionary = update_frequency_dictionary_(species, 0, sex_ratio_dictionary)
            age_ratio_dictionary = update_frequency_dictionary_(species, 0, age_ratio_dictionary)
            subspecies_ratio_dictionary = update_frequency_dictionary_(species, 0, subspecies_ratio_dictionary)
            vocalisation_ratio_dictionary = update_frequency_dictionary_(species, 0, vocalisation_ratio_dictionary)
            for metadata in metadata_list:
                if any(sex != "Unknown" for sex in metadata["sex"]):
                    sex_ratio_dictionary = update_frequency_dictionary_(species, 1, sex_ratio_dictionary)
                if any(age != "Unknown" for age in metadata["age"]):
                    age_ratio_dictionary = update_frequency_dictionary_(species, 1, age_ratio_dictionary)
                if metadata["subspecies"] != "":
                    subspecies_ratio_dictionary = update_frequency_dictionary_(species, 1, subspecies_ratio_dictionary)
                if any(sex != "Unknown" for sex in metadata["general_vocalisation"]) or any(sex != "Unknown" for sex in metadata["specific_vocalisation"]):
                    vocalisation_ratio_dictionary = update_frequency_dictionary_(species, 1, vocalisation_ratio_dictionary)
            sex_ratio_dictionary[species] /= observation_count
            age_ratio_dictionary[species] /= observation_count
            subspecies_ratio_dictionary[species] /= observation_count
            vocalisation_ratio_dictionary[species] /= observation_count
            j += 1
            print(j)
        attribute_ratio_dictionary_list.append(sex_ratio_dictionary)
        attribute_ratio_dictionary_list.append(age_ratio_dictionary)
        attribute_ratio_dictionary_list.append(subspecies_ratio_dictionary)
        attribute_ratio_dictionary_list.append(vocalisation_ratio_dictionary)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    parameters = {
        "x_label": "Sorted Species",
        "y_label": "Sighting Ratio",
        "legend_location": "upper right",
        "y_min": 0,
        "y_max": 1.1,
        "y_tick": 0.1,
        "x_min": 0,
        "x_max": 1000,
        "x_tick": 100,
        "line_colors": [colors[2], colors[0], colors[1], colors[3]]
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.frequency_distribution_line_graph(["Sex", "Age", "Subspecies", "Vocalisation"], attribute_ratio_dictionary_list, include_average=True)

def date_density_graph(dataset_name_list, constructor_list, dataset_id_list):
    # Setup
    count_lists = []
    day_lists = []
    j = 0
    for constructor, dataset_id in zip(constructor_list, dataset_id_list):
        day_list = [i for i in range(0, 365)]
        count_list = [0 for _ in range(0, 365)]
        version_configuration, modality, partition, dataset_type_, solution_type_ = parse_dataset_id(dataset_id)
        dataset = constructor.get_dataset_handle(version_configuration, modality, partition, dataset_type_=dataset_type_)
        source_metadata = constructor.get_source_metadata_handle(version_configuration, modality, partition, dataset_type_=dataset_type_)
        species_list = constructor.read_species_list(dataset)
        for species in species_list:
            metadata_list = constructor.read_metadata_list(dataset, source_metadata, species, dataset_type_=dataset_type_, solution_type_=solution_type_)
            for metadata in metadata_list:
                date_of_year_ = metadata["date"].split("-", 1)[1]
                if date_of_year_ == "02-29": date_of_year_ = "02-28"
                day = int(datetime.strptime(f"2019-{date_of_year_}", "%Y-%m-%d").timetuple().tm_yday) - 1
                count_list[day] += 1
            j += 1
            print(j)
        day_lists.append(day_list)
        count_lists.append(count_list)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    parameters = {
        "x_label": "Day of Year",
        "y_label": "Number of Sightings",
        "legend_location": "upper left",
        "y_min": 0,
        "y_max": 1000,
        "y_tick": 100,
        "x_min": 0,
        "x_max": 365,
        "x_ticks": [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
        "x_tick_labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "line_colors": [colors[2], colors[0], colors[1]]
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.line_graph(dataset_name_list, day_lists, count_lists)

def time_density_graph(dataset_name_list, constructor_list, dataset_id_list):
    # Setup
    minute_lists = []
    count_lists = []
    j = 0
    for constructor, dataset_id in zip(constructor_list, dataset_id_list):
        minute_list = [i for i in range(0, 1440)]
        count_list = [0 for _ in range(0, 1440)]
        version_configuration, modality, partition, dataset_type_, solution_type_ = parse_dataset_id(dataset_id)
        dataset = constructor.get_dataset_handle(version_configuration, modality, partition, dataset_type_=dataset_type_)
        source_metadata = constructor.get_source_metadata_handle(version_configuration, modality, partition, dataset_type_=dataset_type_)
        species_list = constructor.read_species_list(dataset)
        for species in species_list:
            metadata_list = constructor.read_metadata_list(dataset, source_metadata, species, dataset_type_=dataset_type_, solution_type_=solution_type_)
            for metadata in metadata_list:
                minute = int((datetime.strptime(metadata["time"], "%H:%M") - datetime.strptime("00:00", "%H:%M")).total_seconds() / 60.0)
                count_list[minute] += 1
            j += 1
            print(j)
        minute_lists.append(minute_list)
        count_lists.append(count_list)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    parameters = {
        "x_label": "Hour of Day",
        "y_label": "Number of Sightings",
        "legend_location": "upper left",
        "x_min": 0,
        "x_max": 1440,
        "y_min": 0,
        "y_max": 300,
        "y_tick": 30,
        "x_ticks": [i * 60 for i in range(0, 24)],
        "x_tick_labels": [f"{str(i)}" for i in range(0, 24)],
        "line_colors": [colors[2], colors[0], colors[1]]
    }

    # # Graph
    grapher = Grapher(parameters)
    grapher.line_graph(dataset_name_list, minute_lists, count_lists)

def geographical_density_map_graph(dataset_name_list, constructor_list, dataset_id_list):
    # Setup
    location_lists = []
    i = 0
    for constructor, dataset_id in zip(constructor_list, dataset_id_list):
        location_list = []
        version_configuration, modality, partition, dataset_type_, solution_type_ = parse_dataset_id(dataset_id)
        dataset = constructor.get_dataset_handle(version_configuration, modality, partition, dataset_type_=dataset_type_)
        source_metadata = constructor.get_source_metadata_handle(version_configuration, modality, partition, dataset_type_=dataset_type_)
        species_list = constructor.read_species_list(dataset)
        for species in species_list:
            metadata_list = constructor.read_metadata_list(dataset, source_metadata, species, dataset_type_=dataset_type_, solution_type_=solution_type_)
            for metadata in metadata_list:
                location = (metadata["longitude"], metadata["latitude"])
                location_list.append(location)
            i += 1
            print(i)
        location_lists.append(location_list)
    print(len(location_lists[0]))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    parameters = {
        "x_label": "Sorted Species",
        "y_label": "Number of Sightings",
        "y_scale": "log",
        "legend_location": "upper left",
        "x_min": 0,
        "x_max": 1000,
        "x_tick": 100,
        "y_max": 1000000,
        "y_min": 1,
        "line_colors": [colors[2], colors[0], colors[1]],
        "average_line_colors": [colors[2], colors[0], colors[1]]
    }

    # # Graph
    grapher = Grapher(parameters)
    grapher.geographical_density_map(location_lists)
