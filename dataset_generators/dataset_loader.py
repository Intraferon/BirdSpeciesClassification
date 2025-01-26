import math
from datetime import datetime
from dataset_generators.dataset_constructor import *
from partition_configurations import *
from pair_configurations import *
from dataset_generators.constants import *
import numpy as np


def load_dataset(constructor, configuration, partition, modality, context_parameters, augmentation_parameters):
    if modality == "image":
        dataset_kwargs = load_image_dataset(constructor, configuration, partition, context_parameters,  augmentation_parameters)
    if modality == "audio":
        dataset_kwargs = load_audio_dataset(constructor, configuration, partition, context_parameters, augmentation_parameters)
    if modality == "image-audio":
        dataset_kwargs = load_image_audio_dataset(constructor, configuration, partition, context_parameters, augmentation_parameters)
    return dataset_kwargs


def load_image_dataset(constructor, configuration, partition, context_parameters, augmentation_parameters):
    solution_type_ = determine_partition_solution_type(configuration)
    compiled_dataset_path = f"{constructor.compiled_path}{configuration}_image_{partition}_partition_{solution_type_}.json"
    dataset_kwargs = read_data_from_file_(compiled_dataset_path)
    if context_parameters is None:
        dataset_kwargs.pop("observation_context_list")
    else:
        dataset_kwargs["observation_context_list"] = paraphrase_context(dataset_kwargs["observation_context_list"], context_parameters)
        dataset_kwargs["observation_context_list"] = wrap_encode_context(dataset_kwargs["observation_context_list"])
    if augmentation_parameters is not None:
        dataset_kwargs["random_crop"] = augmentation_parameters["random_crop"]
        dataset_kwargs["center_crop"] = augmentation_parameters["center_crop"]
    return dataset_kwargs


def load_audio_dataset(constructor, configuration, partition, context_parameters, augmentation_parameters):
    solution_type_ = determine_partition_solution_type(configuration)
    compiled_dataset_path = f"{constructor.compiled_path}{configuration}_audio_{partition}_partition_{solution_type_}.json"
    dataset_kwargs = read_data_from_file_(compiled_dataset_path)
    if context_parameters is None:
        dataset_kwargs.pop("observation_context_list")
    else:
        dataset_kwargs["observation_context_list"] = paraphrase_context(dataset_kwargs["observation_context_list"], context_parameters)
        dataset_kwargs["observation_context_list"] = wrap_encode_context(dataset_kwargs["observation_context_list"])
    if augmentation_parameters is None or not augmentation_parameters["noise_addition"]:
        dataset_kwargs.pop("instance_noise_name_list", None)
        dataset_kwargs.pop("instance_noise_length_list", None)
    return dataset_kwargs


def load_image_audio_dataset(constructor, configuration, partition, context_parameters, augmentation_parameters):
    if configuration != "only_context":
        solution_type_ = determine_pair_solution_type(configuration[3])
        compiled_dataset_path = f"{constructor.compiled_path}{configuration[0]}_image-audio_{partition}_pair_{solution_type_}.json"
        dataset_kwargs = read_data_from_file_(compiled_dataset_path)
        if context_parameters is not None:
            dataset_kwargs["observation_image_context_list"] = paraphrase_context(dataset_kwargs["observation_image_context_list"], context_parameters)
            dataset_kwargs["observation_signal_context_list"] = paraphrase_context(dataset_kwargs["observation_signal_context_list"], context_parameters)
            dataset_kwargs["observation_image_context_list"] = wrap_encode_context(dataset_kwargs["observation_image_context_list"])
            dataset_kwargs["observation_signal_context_list"] = wrap_encode_context(dataset_kwargs["observation_signal_context_list"])
        if augmentation_parameters is None or not augmentation_parameters["noise_addition"]:
            dataset_kwargs.pop("instance_noise_name_list", None)
            dataset_kwargs.pop("instance_noise_length_list", None)
        if augmentation_parameters is not None:
            dataset_kwargs["random_crop"] = augmentation_parameters["random_crop"]
            dataset_kwargs["center_crop"] = augmentation_parameters["center_crop"]
    else:
        solution_type_ = determine_partition_solution_type(configuration)
        image_compiled_dataset_path = f"{constructor.compiled_path}only_context_image_{partition}_partition_{solution_type_}.json"
        image_dataset_kwargs = read_data_from_file_(image_compiled_dataset_path)
        audio_compiled_dataset_path = f"{constructor.compiled_path}only_context_audio_{partition}_partition_{solution_type_}.json"
        audio_dataset_kwargs = read_data_from_file_(audio_compiled_dataset_path)
        dataset_kwargs = {
            "observation_count": int(image_dataset_kwargs["observation_count"]) + int(audio_dataset_kwargs["observation_count"]),
            "observation_species_list": image_dataset_kwargs["observation_species_list"] + audio_dataset_kwargs["observation_species_list"],
            "observation_context_list": image_dataset_kwargs["observation_context_list"] + audio_dataset_kwargs["observation_context_list"]
        }
        dataset_kwargs["observation_context_list"] = paraphrase_context(dataset_kwargs["observation_context_list"], context_parameters)
        dataset_kwargs["observation_context_list"] = wrap_encode_context(dataset_kwargs["observation_context_list"])
    return dataset_kwargs


def paraphrase_context(observation_context_list, context_parameters):
    for i in range(len(observation_context_list)):
        latitude = observation_context_list[i][0]
        longitude = observation_context_list[i][1]
        date_ = observation_context_list[i][2]
        time_ = observation_context_list[i][3]
        context = []
        if context_parameters["include_location"]:
            latitude_norm = latitude / 90.0
            context.append(latitude_norm)
            longitude_norm = longitude / 180.0
            context.append(longitude_norm)
        if context_parameters["include_date"]:
            date_of_year_ = date_.split("-", 1)[1]
            if date_of_year_ == "02-29": date_of_year_ = "02-28"
            day_ = int(datetime.strptime(f"2019-{date_of_year_}", "%Y-%m-%d").timetuple().tm_yday) - 1
            day_norm_ = (day_ - 182) / 182
            context.append(day_norm_)
        if context_parameters["include_time"]:
            minute_ = (datetime.strptime(time_, "%H:%M") - datetime.strptime("00:00", "%H:%M")).total_seconds() / 60.0
            minute_norm_ = (minute_ - 719.5) / 719.5
            context.append(minute_norm_)
        observation_context_list[i] = context
    return observation_context_list


def wrap_encode_context(observation_context_list):
    wrap_encoded_observation_context_list = []
    for i in range(len(observation_context_list)):
        context = []
        for j in range(len(observation_context_list[i])):
            context.append(math.sin(math.pi * observation_context_list[i][j]))
            context.append(math.cos(math.pi * observation_context_list[i][j]))
        wrap_encoded_observation_context_list.append(context)
    return wrap_encoded_observation_context_list