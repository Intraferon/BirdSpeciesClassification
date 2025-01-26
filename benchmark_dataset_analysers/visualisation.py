from matplotlib import pyplot as plt
from graphing import Grapher


dataset_name_translator = {"cub_200_2010": "CUB-200-\n2010",
                           "cub_200_2011": "CUB-200-\n2011",
                           "birdsnap": "Birdsnap",
                           "nabirds": "NABirds",
                           "inaturalist_2017": "iNaturalist\n2017",
                           "inaturalist_2018": "iNaturalist\n2018",
                           "clo_43sd": "CLO-43SD",
                           "birdclef_2014": "BirdCLEF\n2014",
                           "birdclef_2015": "BirdCLEF\n2015/2016",
                           "birdclef_2017": "BirdCLEF\n2017/2018",
                           "birdclef_2019": "BirdCLEF\n2019",
                           "birdclef_2020": "BirdCLEF\n2020"}

partition_name_translator = {"train": "Training Set",
                             "validation": "Validation Set",
                             "test": "Testing Set",
                             "monophone_train": "Training Set",
                             "monophone_test": "Testing Set",
                             "time_annotated_soundscape_validation": "Validation (Time-Annotated Soundscape)",
                             "record_annotated_soundscape_test": "Test (Record-Annotated Soundscape)",
                             "time_annotated_soundscape_test": "Test (Time-Annotated Soundscape)"}


def image_summary(dataset_name_list, summary_list, partition):
    print(f"\nimage_summary: {partition}\n")
    for i in range(len(dataset_name_list)):
        print(f"{dataset_name_list[i]}: {summary_list[i]}")


def image_count_graph(group_name_list, sub_bar_name_list, group_list):

    # Setup
    i = 0
    for group in group_list:
        if group_name_list[i] == "inaturalist_2017" or group_name_list[i] == "inaturalist_2018":
            j = 0
            for bar in group:
                for sub_bar_name in bar:
                    group_list[i][j][sub_bar_name] /= 10.0
                j += 1
        i += 1

    # Parameters
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    parameters = {
        "y_label": "Number of Images",
        "y_label_weight": "bold",
        "y_label_color": colors[0],
        "x_tick_label_colors": [colors[0]] * 4 + [colors[2]] * 2,
        "x_tick_label_weight": "bold",
        "y_min": 0,
        "y_max": 60000,
        "y_tick": 5000,
        "alternate_axis": "y",
        "alternate_y_label": "Number of Images",
        "alternate_y_label_weight": "bold",
        "alternate_y_label_color": colors[2],
        "alternate_y_min": 0,
        "alternate_y_max": 600000,
        "alternate_y_tick": 50000,
        "bar_width": 0.2,
        "legend_location": "upper left",
        "sub_bar_colors": [colors[4], colors[3], colors[1]],
        "group_name_translator": dataset_name_translator,
        "sub_bar_name_translator": partition_name_translator,
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.grouped_bar_graph(group_name_list, sub_bar_name_list, group_list)


def image_mean_graph(group_name_list, sub_bar_name_list, group_list):

    # [colors[4], colors[3], colors[1]]

    # Parameters
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    parameters = {
        "y_label": "Mean Class Size",
        "y_min": 0,
        "y_max": 120,
        "y_tick": 10,
        "bar_width": 0.2,
        "legend_location": "upper left",
        "sub_bar_colors": [colors[0], colors[1], colors[2]],
        "group_name_translator": dataset_name_translator,
        "sub_bar_name_translator": partition_name_translator,
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.grouped_bar_graph(group_name_list, sub_bar_name_list, group_list)

def image_frequency_distribution_graph(line_name_list, y_line_dictionary):

    # Parameters
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    parameters = {
        "x_multiplier": [1.0] * 3 + [0.1] * 2,
        "x_label": "Sorted Classes",
        "x_label_color": colors[0],
        "x_label_weight": "bold",
        "y_label": "Number of Images",
        "y_scale": "log",
        "x_min": 0,
        "x_max": 900,
        "x_tick": 100,
        "y_min": 1,
        "y_max": 10000,
        "alternate_axis": "x",
        "alternate_x_label": "Sorted Classes",
        "alternate_x_label_color": colors[2],
        "alternate_x_label_weight": "bold",
        "alternate_x_min": 0,
        "alternate_x_max": 9000,
        "alternate_x_tick": 1000,
        "legend_location": "upper left",
        "line_styles": ["-.", "-", "--", "-", "--"],
        "line_colors": [colors[0]] * 3 + [colors[2]] * 2,
        "line_name_translator": {dataset_name: dataset_name_translator[dataset_name].replace("\n", " ").replace("- ", "-") for dataset_name in dataset_name_translator}
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.frequency_distribution_line_graph(line_name_list, y_line_dictionary, include_average=False)


def audio_summary(dataset_name_list, summary_list, audio_type, partition):
    print(f"\naudio_summary: {audio_type}_{partition}\n")
    for i in range(len(dataset_name_list)):
        print(f"{dataset_name_list[i]}: {summary_list[i]}")


def audio_count_graph(group_name_list, sub_bar_name_list, group_list):

    # Setup
    i = 0
    for group in group_list:
        if group_name_list[i] == "birdclef_2019":
            j = 0
            for bar in group:
                for sub_bar_name in bar:
                    if sub_bar_name == "time_annotated_soundscape_test":
                        group_list[i][j][sub_bar_name] = 104021
                j += 1
        i += 1

    # Parameters
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    parameters = {
        "y_label": "Number of Audio Recordings",
        "y_min": 0,
        "y_max": 40000,
        "y_tick": 5000,
        "bar_width": 0.1,
        "legend_location": "upper left",
        "sub_bar_colors": [colors[1], colors[4]],
        "group_name_translator": dataset_name_translator,
        "sub_bar_name_translator": partition_name_translator,
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.grouped_bar_graph(group_name_list, sub_bar_name_list, group_list)


def audio_mean_graph(group_name_list, sub_bar_name_list, group_list):

    # Parameters
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    parameters = {
        "y_label": "Mean Class Size",
        "y_min": 0,
        "y_max": 120,
        "y_tick": 10,
        "bar_width": 0.1,
        "legend_location": "upper right",
        "sub_bar_colors": [colors[0], colors[2]],
        "group_name_translator": dataset_name_translator,
        "sub_bar_name_translator": partition_name_translator,
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.grouped_bar_graph(group_name_list, sub_bar_name_list, group_list)


def audio_frequency_distribution_graph(line_name_list, y_line_dictionary_list):

    # Setup
    # for species in y_line_dictionary_list[5]:
    #     if y_line_dictionary_list[5][species] == 100:
    #         y_line_dictionary_list[5][species] -= 4

    # Parameters
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    parameters = {
        "x_label": "Sorted Classes",
        "y_label": "Number of Audio Recordings",
        "y_scale": "log",
        "x_min": 0,
        "x_max": 1500,
        "x_tick": 100,
        "y_min": 1,
        "y_max": 10000,
        "legend_location": "upper right",
        "line_colors": [colors[0]] + colors[2:5] + [colors[6]],
        "line_name_translator": {dataset_name: dataset_name_translator[dataset_name].replace("\n", " ") for dataset_name in dataset_name_translator}
    }

    # Graph
    grapher = Grapher(parameters)
    grapher.frequency_distribution_line_graph(line_name_list, y_line_dictionary_list, include_average=False)


