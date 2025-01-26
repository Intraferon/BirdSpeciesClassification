from benchmark_dataset_analysers.cub_200_2010 import CUB2002010
from benchmark_dataset_analysers.cub_200_2011 import CUB2002011
from benchmark_dataset_analysers.birdsnap import Birdsnap
from benchmark_dataset_analysers.nabirds import NABirds
from benchmark_dataset_analysers.inaturalist_2017 import INaturalist2017
from benchmark_dataset_analysers.inaturalist_2018 import INaturalist2018
from benchmark_dataset_analysers.clo_43sd import CLO43SD
from benchmark_dataset_analysers.birdclef_2014 import BirdCLEF2014
from benchmark_dataset_analysers.birdclef_2015 import BirdCLEF2015
from benchmark_dataset_analysers.birdclef_2017 import BirdCLEF2017
from benchmark_dataset_analysers.birdclef_2019 import BirdCLEF2019
from benchmark_dataset_analysers.birdclef_2020 import BirdCLEF2020
from benchmark_dataset_analysers.visualisation import *


cub_200_2010 = CUB2002010()
cub_200_2011 = CUB2002011()
birdsnap = Birdsnap()
nabirds = NABirds()
inaturalist_2017 = INaturalist2017()
inaturalist_2018 = INaturalist2018()
clos_43sd = CLO43SD()
birdclef_2014 = BirdCLEF2014()
birdclef_2015 = BirdCLEF2015()
birdclef_2017 = BirdCLEF2017()
birdclef_2019 = BirdCLEF2019()
birdclef_2020 = BirdCLEF2020()


def prepare_image_statistics():
    dataset_list = [cub_200_2010, cub_200_2011, birdsnap, nabirds, inaturalist_2017, inaturalist_2018]
    partition_list = ["train", "validation", "test"]
    for dataset in dataset_list:
        for partition in partition_list:
            dataset.statisticise(partition)


def print_image_summary(partition):
    dataset_name_list = ["cub_200_2010", "cub_200_2011", "birdsnap", "nabirds", "inaturalist_2017", "inaturalist_2018"]
    dataset_list = [cub_200_2010, cub_200_2011, birdsnap, nabirds, inaturalist_2017, inaturalist_2018]
    summary_list = [dataset.get_statistics("summary", modality="image", partition=partition) for dataset in dataset_list]
    image_summary(dataset_name_list, summary_list, partition=partition)


def draw_image_count_graph():
    dataset_name_list = ["cub_200_2011", "birdsnap", "nabirds", "inaturalist_2017", "inaturalist_2018"]
    dataset_list = [cub_200_2011, birdsnap, nabirds, inaturalist_2017, inaturalist_2018]
    partition_list = ["train", "validation", "test"]
    count_list = [[{partition: int(dataset.get_statistics("summary", modality="image", partition=partition)["instance_count"])}
                   for partition in partition_list
                   if dataset.statistics_exists("summary", partition=partition)] for dataset in dataset_list]
    image_count_graph(dataset_name_list, partition_list, count_list)


def draw_image_mean_graph():
    dataset_name_list = ["cub_200_2011", "birdsnap", "nabirds", "inaturalist_2017", "inaturalist_2018"]
    dataset_list = [cub_200_2011, birdsnap, nabirds, inaturalist_2017, inaturalist_2018]
    partition_list = ["train", "validation", "test"]
    count_list = [[{partition: int(dataset.get_statistics("summary", modality="image", partition=partition)["instance_count"] / dataset.get_statistics("summary", modality="image", partition=partition)["class_count"])}
                   for partition in partition_list
                   if dataset.statistics_exists("summary", partition=partition)] for dataset in dataset_list]
    image_mean_graph(dataset_name_list, partition_list, count_list)


def draw_image_distribution_graph(partition):
    dataset_name_list = ["cub_200_2011", "birdsnap", "nabirds", "inaturalist_2017", "inaturalist_2018"]
    partition_list = [partition] * len(dataset_name_list)
    if partition == "test":
        partition_list[-1] = "validation"
        partition_list[-2] = "validation"
    dataset_list = [cub_200_2011, birdsnap, nabirds, inaturalist_2017, inaturalist_2018]
    taxon_frequency_dictionary_list = [dataset.get_statistics("distribution", modality="image", partition=partition_) for dataset, partition_ in zip(dataset_list, partition_list)]
    image_frequency_distribution_graph(dataset_name_list, taxon_frequency_dictionary_list)


def prepare_audio_statistics():
    dataset_list = [clos_43sd, birdclef_2014, birdclef_2015, birdclef_2017, birdclef_2019, birdclef_2020]
    partition_list = ["train", "validation", "test"]
    audio_type_list = ["monophone", "record_annotated_soundscape", "time_annotated_soundscape"]
    for dataset in dataset_list:
        for audio_type in audio_type_list:
            for partition in partition_list:
                dataset.statisticise(audio_type, partition)


def print_audio_summary(audio_type, partition):
    dataset_name_list = ["clo_43sd", "birdclef_2014", "birdclef_2015", "birdclef_2017", "birdclef_2019", "birdclef_2020"]
    dataset_list = [clos_43sd, birdclef_2014, birdclef_2015, birdclef_2017, birdclef_2019, birdclef_2020]
    dataset_rendition_list = [None, "birdclef_2014", "birdclef_2015", "birdclef_2017", None, None]
    summary_list = [dataset.get_statistics("summary", modality="audio", rendition=dataset_rendition, audio_type=audio_type, partition=partition) for dataset, dataset_rendition in zip(dataset_list, dataset_rendition_list)]
    audio_summary(dataset_name_list, summary_list, audio_type=audio_type, partition=partition)


def draw_audio_count_graph():
    dataset_name_list = ["birdclef_2014", "birdclef_2015", "birdclef_2017"]
    partition_list = ["monophone_train", "monophone_test"]
    dataset_list = [birdclef_2014, birdclef_2015, birdclef_2017]
    dataset_rendition_list = ["birdclef_2014", "birdclef_2015", "birdclef_2017"]
    set_list = [("monophone", "train"), ("monophone", "test")]
    count_list = [[{f"{set_[0]}_{set_[1]}": int(dataset.get_statistics("summary", modality="audio", rendition=dataset_rendition, audio_type=set_[0], partition=set_[1])["instance_count"])}
                   for set_ in set_list if dataset.statistics_exists("summary", rendition=dataset_rendition, audio_type=set_[0], partition=set_[1])]
                  for dataset, dataset_rendition in zip(dataset_list, dataset_rendition_list)]
    audio_count_graph(dataset_name_list, partition_list, count_list)


def draw_audio_mean_graph():
    dataset_name_list = ["birdclef_2014", "birdclef_2015", "birdclef_2017"]
    partition_list = ["monophone_train", "monophone_test"]
    dataset_list = [birdclef_2014, birdclef_2015, birdclef_2017]
    dataset_rendition_list = ["birdclef_2014", "birdclef_2015", "birdclef_2017"]
    set_list = [("monophone", "train"), ("monophone", "test")]
    count_list = [[{f"{set_[0]}_{set_[1]}": int(dataset.get_statistics("summary", modality="audio", rendition=dataset_rendition, audio_type=set_[0], partition=set_[1])["instance_count"] / dataset.get_statistics("summary", modality="audio", rendition=dataset_rendition, audio_type=set_[0], partition="train")["class_count"])}
                   for set_ in set_list if dataset.statistics_exists("summary", rendition=dataset_rendition, audio_type=set_[0], partition=set_[1])]
                  for dataset, dataset_rendition in zip(dataset_list, dataset_rendition_list)]
    audio_mean_graph(dataset_name_list, partition_list, count_list)


def draw_audio_distribution_graph():
    dataset_name_list = ["birdclef_2014", "birdclef_2015", "birdclef_2017"]
    dataset_list = [birdclef_2014, birdclef_2015, birdclef_2017]
    dataset_rendition_list = ["birdclef_2014", "birdclef_2015", "birdclef_2017"]
    taxon_frequency_dictionary_list = [dataset.get_statistics("distribution", modality="audio", rendition=dataset_rendition, audio_type="monophone", partition="train")
                                       for dataset, dataset_rendition in zip(dataset_list, dataset_rendition_list)]
    audio_frequency_distribution_graph(dataset_name_list, taxon_frequency_dictionary_list)


# print_image_summary("train")
# print_image_summary("validation")
# print_image_summary("test")
# draw_image_count_graph()
# draw_image_distribution_graph("train")
# draw_image_distribution_graph("test")
# print_audio_summary("monophone", "train")
# print_audio_summary("monophone", "test")
# print_audio_summary("time_annotated_soundscape", "validation")
# print_audio_summary("record_annotated_soundscape", "test")
# print_audio_summary("time_annotated_soundscape", "test")
# draw_audio_count_graph()
# draw_audio_distribution_graph()

draw_image_distribution_graph("test")

# draw_audio_distribution_graph()





