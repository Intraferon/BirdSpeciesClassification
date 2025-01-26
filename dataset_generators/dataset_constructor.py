import audiofile
from utility import *
from dataset_generators.constants import *
from pair_configurations import *
from partition_configurations import *
from PIL import Image
from features.spectrogram import Spectrogram


def construct_general_dataset(subset, constructor, configuration, partition, modality, overwrite=True):
    if configuration != "only_context":
        if modality == "image":
            construct_image_dataset(subset, constructor, configuration, partition, overwrite=overwrite)
        if modality == "audio":
            construct_audio_dataset(subset, constructor, configuration, partition, overwrite=overwrite)
        if modality == "image-audio":
            construct_image_audio_dataset(subset, constructor, configuration, partition, overwrite=overwrite)
    else:
        if modality == "image" or modality == "audio":
            construct_context_dataset(constructor, configuration, modality, partition, overwrite=overwrite)
        else:
            construct_image_audio_context_dataset(constructor, configuration, modality, partition, overwrite=overwrite)

def construct_image_dataset(subset, constructor, configuration, partition, overwrite=True):

    solution_type_ = determine_partition_solution_type(configuration)

    compiled_dataset_path = f"{constructor.compiled_path}{configuration}_image_{partition}_partition_{solution_type_}.json"

    if overwrite or not os.path.exists(compiled_dataset_path):

        converted_list = set(read_data_from_file_(f"{subset.subset_path}/converted.txt"))

        observation_species_list = []
        observation_context_list = []

        observation_image_name_list = []
        observation_image_no_list = []

        dataset = constructor.get_dataset_handle(configuration, "image", partition, dataset_type_="partition")
        source_metadata = constructor.get_source_metadata_handle(configuration, "image", partition, dataset_type_="partition")
        species_list = constructor.read_species_list(dataset)

        i = 0

        observation_count = 0
        instance_image_count = 0

        for species in species_list:

            metadata_list = constructor.read_metadata_list(dataset, source_metadata, species, dataset_type_="partition", solution_type_=solution_type_)

            for metadata in metadata_list:

                observation_image_name = metadata["_id"]
                image_count = metadata["media_types"].count("Image")

                image_no_list = []
                for image_no in range(image_count):
                    image_id = f"{species}/{observation_image_name}_{image_no}"
                    if image_id in converted_list:
                        image_no_list.append(image_no)
                        instance_image_count += 1

                if image_no_list:

                    observation_species_list.append(species)
                    observation_context = create_context(metadata)
                    observation_context_list.append(observation_context)

                    observation_image_name_list.append(observation_image_name)
                    observation_image_no_list.append(image_no_list)
                    observation_count += 1

            print(f"Species: {i}")
            i += 1

        dataset_kwargs = {
            "observation_count": observation_count,
            "instance_image_count": instance_image_count,
            "observation_species_list": observation_species_list,
            "observation_context_list": observation_context_list,
            "observation_image_name_list": observation_image_name_list,
            "observation_image_no_list": observation_image_no_list
        }

        save_data_to_file_(compiled_dataset_path, dataset_kwargs)


def construct_audio_dataset(subset, constructor, configuration, partition, overwrite=True):

    spectrogram = Spectrogram()

    solution_type_ = determine_partition_solution_type(configuration)

    compiled_dataset_path = f"{constructor.compiled_path}{configuration}_audio_{partition}_partition_{solution_type_}.json"

    if overwrite or not os.path.exists(compiled_dataset_path):

        dataset = constructor.get_dataset_handle(configuration, "audio", partition, dataset_type_="partition")
        source_metadata = constructor.get_source_metadata_handle(configuration, "audio", partition, dataset_type_="partition")
        species_list = constructor.read_species_list(dataset)

        observation_species_list = []
        observation_context_list = []

        observation_signal_name_list = []
        observation_signal_no_list = []
        observation_signal_length_list = []

        if partition == "train":
            instance_noise_name_list = []
            instance_noise_length_list = []

        i = 0

        observation_count = 0
        instance_signal_count = 0

        for species in species_list:

            metadata_list = constructor.read_metadata_list(dataset, source_metadata, species, dataset_type_="partition", solution_type_=solution_type_)

            for metadata in metadata_list:

                observation_audio_name = metadata["_id"]
                audio_count = metadata["media_types"].count("Audio")

                signal_no_list = []
                signal_length_list = []

                for audio_no in range(audio_count):

                    audio_id = f"{species}/{observation_audio_name}_{audio_no}"
                    signal_path = f"{subset.aligned_signal_files_path}/{audio_id}.{TRAIN_AUDIO_FORMAT}"

                    if os.path.exists(signal_path):
                        signal = audiofile.read(signal_path)[0]
                        signal_length = signal.shape[0]
                        if signal_length > MINIMUM_SIGNAL_LENGTH and spectrogram.is_valid(signal):
                            signal_no_list.append(audio_no)
                            signal_length_list.append(signal_length)
                            instance_signal_count += max(min(int(signal_length / AUDIO_LENGTH), MAXIMUM_SPECTROGRAM_COUNT), 1)

                    if partition == "train":
                        noise_path = f"{subset.aligned_noise_files_path}/{audio_id}.{TRAIN_AUDIO_FORMAT}"
                        if os.path.exists(noise_path):
                            noise = audiofile.read(noise_path)[0]
                            noise_length = noise.shape[0]
                            if noise_length > AUDIO_LENGTH and spectrogram.is_valid(noise):
                                instance_noise_name_list.append(audio_id)
                                instance_noise_length_list.append(noise_length)

                if signal_no_list:

                    observation_species_list.append(species)
                    observation_context = create_context(metadata)
                    observation_context_list.append(observation_context)

                    observation_signal_name_list.append(observation_audio_name)
                    observation_signal_no_list.append(signal_no_list)
                    observation_signal_length_list.append(signal_length_list)
                    observation_count += 1

            print(f"Species: {i}")
            i += 1

        dataset_kwargs = {
            "observation_count": observation_count,
            "instance_signal_count": instance_signal_count,
            "observation_species_list": observation_species_list,
            "observation_context_list": observation_context_list,
            "observation_signal_name_list": observation_signal_name_list,
            "observation_signal_length_list": observation_signal_length_list,
            "observation_signal_no_list": observation_signal_no_list
        }

        if partition == "train":
            dataset_kwargs["instance_noise_name_list"] = instance_noise_name_list
            dataset_kwargs["instance_noise_length_list"] = instance_noise_length_list

        save_data_to_file_(compiled_dataset_path, dataset_kwargs)


def construct_image_audio_dataset(subset, constructor, configuration, partition, overwrite=True):

    spectrogram = Spectrogram()

    solution_type_ = determine_pair_solution_type(configuration[3])
    compiled_dataset_path = f"{constructor.compiled_path}{configuration[0]}_image-audio_{partition}_pair_{solution_type_}.json"

    image_solution_type_ = determine_partition_solution_type(configuration[1])
    image_compiled_dataset_path = f"{constructor.compiled_path}{configuration[1]}_image_{partition}_partition_{image_solution_type_}.json"
    image_dataset_kwargs = read_data_from_file_(image_compiled_dataset_path)
    source_observation_image_name_list = image_dataset_kwargs["observation_image_name_list"]
    source_observation_image_no_list = image_dataset_kwargs["observation_image_no_list"]
    source_observation_image_name_dictionary = {source_observation_image_name_list[i]: i for i in range(len(source_observation_image_name_list))}

    audio_solution_type_ = determine_partition_solution_type(configuration[2])
    audio_compiled_dataset_path = f"{constructor.compiled_path}{configuration[2]}_audio_{partition}_partition_{audio_solution_type_}.json"
    audio_dataset_kwargs = read_data_from_file_(audio_compiled_dataset_path)
    source_observation_signal_name_list = audio_dataset_kwargs["observation_signal_name_list"]
    source_observation_signal_no_list = audio_dataset_kwargs["observation_signal_no_list"]
    source_observation_signal_length_list = audio_dataset_kwargs["observation_signal_length_list"]
    source_observation_signal_name_dictionary = {source_observation_signal_name_list[i]: i for i in range(len(source_observation_signal_name_list))}

    if partition == "train":
        source_instance_noise_name_list = audio_dataset_kwargs["instance_noise_name_list"]
        source_instance_noise_length_list = audio_dataset_kwargs["instance_noise_length_list"]
        source_instance_noise_name_dictionary = {source_instance_noise_name_list[i]: i for i in range(len(source_instance_noise_name_list))}

    if overwrite or not os.path.exists(compiled_dataset_path):

        dataset = constructor.get_dataset_handle(configuration[0], "image-audio", partition, dataset_type_="pair")
        image_source_metadata = constructor.get_source_metadata_handle(configuration[0], "image", partition, dataset_type_="pair")
        audio_source_metadata = constructor.get_source_metadata_handle(configuration[0], "audio", partition, dataset_type_="pair")
        species_list = constructor.read_species_list(dataset)

        observation_species_list = []

        observation_image_context_list = []
        observation_image_name_list = []
        observation_image_no_list = []

        observation_signal_context_list = []
        observation_signal_name_list = []
        observation_signal_no_list = []
        observation_signal_length_list = []

        if partition == "train":
            instance_noise_name_list = []
            instance_noise_length_list = []

        i = 0

        observation_count = 0
        instance_image_count = 0
        instance_signal_count = 0

        for species in species_list:

            image_metadata_list, audio_metadata_list = constructor.read_joint_metadata_list(dataset, image_source_metadata, audio_source_metadata, species, dataset_type_="pair", solution_type_=solution_type_)

            metadata_count = len(image_metadata_list)

            for j in range(metadata_count):

                image_metadata = image_metadata_list[j]
                audio_metadata = audio_metadata_list[j]

                observation_image_name = image_metadata["_id"]

                image_no_list = []

                if observation_image_name in source_observation_image_name_dictionary:
                    k = source_observation_image_name_dictionary[observation_image_name]
                    for image_no in source_observation_image_no_list[k]:
                        image_no_list.append(image_no)
                        instance_image_count += 1

                observation_audio_name = audio_metadata["_id"]

                signal_no_list = []
                signal_length_list = []

                if observation_audio_name in source_observation_signal_name_dictionary:
                    k = source_observation_signal_name_dictionary[observation_audio_name]
                    for signal_no, signal_length in zip(source_observation_signal_no_list[k], source_observation_signal_length_list[k]):
                        signal_no_list.append(signal_no)
                        signal_length_list.append(signal_length)
                        instance_signal_count += 1

                if partition == "train":

                    audio_count = audio_metadata["media_types"].count("Audio")

                    for audio_no in range(audio_count):

                        audio_id = f"{species}/{observation_audio_name}_{audio_no}"

                        if audio_id in source_instance_noise_name_dictionary:
                            k = source_instance_noise_name_dictionary[audio_id]
                            noise_length = source_instance_noise_length_list[k]
                            instance_noise_name_list.append(audio_id)
                            instance_noise_length_list.append(noise_length)

                if image_no_list and signal_no_list:

                    observation_species_list.append(species)

                    observation_image_context = create_context(image_metadata)
                    observation_image_context_list.append(observation_image_context)
                    observation_image_name_list.append(observation_image_name)
                    observation_image_no_list.append(image_no_list)

                    observation_signal_context = create_context(audio_metadata)
                    observation_signal_context_list.append(observation_signal_context)
                    observation_signal_name_list.append(observation_audio_name)
                    observation_signal_no_list.append(signal_no_list)
                    observation_signal_length_list.append(signal_length_list)

                    observation_count += 1

            print(f"Species: {i}")
            i += 1

        observation_image_count = len(set(observation_image_name_list))
        observation_signal_count = len(set(observation_signal_name_list))

        dataset_kwargs = {
            "observation_count": observation_count,
            "observation_image_count": observation_image_count,
            "observation_signal_count": observation_signal_count,
            "instance_image_count": instance_image_count,
            "instance_signal_count": instance_signal_count,
            "observation_species_list": observation_species_list,
            "observation_image_context_list": observation_image_context_list,
            "observation_image_name_list": observation_image_name_list,
            "observation_image_no_list": observation_image_no_list,
            "observation_signal_context_list": observation_signal_context_list,
            "observation_signal_name_list": observation_signal_name_list,
            "observation_signal_length_list": observation_signal_length_list,
            "observation_signal_no_list": observation_signal_no_list
        }

        if partition == "train":
            dataset_kwargs["instance_noise_name_list"] = instance_noise_name_list
            dataset_kwargs["instance_noise_length_list"] = instance_noise_length_list

        save_data_to_file_(compiled_dataset_path, dataset_kwargs)


def construct_image_audio_context_dataset(constructor, configuration, modality, partition, overwrite=True):

    solution_type_ = determine_partition_solution_type(configuration)

    compiled_dataset_path = f"{constructor.compiled_path}{configuration}_{modality}_{partition}_partition_{solution_type_}.json"

    if overwrite or not os.path.exists(compiled_dataset_path):

        observation_species_list = []
        observation_context_list = []

        image_dataset = constructor.get_dataset_handle(configuration, "image", partition, dataset_type_="partition")
        image_source_metadata = constructor.get_source_metadata_handle(configuration, "image", partition, dataset_type_="partition")

        audio_dataset = constructor.get_dataset_handle(configuration, "audio", partition, dataset_type_="partition")
        audio_source_metadata = constructor.get_source_metadata_handle(configuration, "audio", partition, dataset_type_="partition")

        species_list = constructor.read_species_list(audio_dataset)

        i = 0

        observation_count = 0

        for species in species_list:

            image_metadata_list = constructor.read_metadata_list(image_dataset, image_source_metadata, species, dataset_type_="partition", solution_type_=solution_type_)

            for metadata in image_metadata_list:

                observation_species_list.append(species)
                observation_context = create_context(metadata)
                observation_context_list.append(observation_context)

                observation_count += 1

            audio_metadata_list = constructor.read_metadata_list(audio_dataset, audio_source_metadata, species, dataset_type_="partition", solution_type_=solution_type_)

            for metadata in audio_metadata_list:

                observation_species_list.append(species)
                observation_context = create_context(metadata)
                observation_context_list.append(observation_context)

                observation_count += 1

            print(f"Species: {i}")
            i += 1

        dataset_kwargs = {
            "observation_count": observation_count,
            "observation_species_list": observation_species_list,
            "observation_context_list": observation_context_list
        }

        save_data_to_file_(compiled_dataset_path, dataset_kwargs)

def construct_context_dataset(constructor, configuration, modality, partition, overwrite=True):

    solution_type_ = determine_partition_solution_type(configuration)

    compiled_dataset_path = f"{constructor.compiled_path}{configuration}_{modality}_{partition}_partition_{solution_type_}.json"

    if overwrite or not os.path.exists(compiled_dataset_path):

        observation_species_list = []
        observation_context_list = []

        dataset = constructor.get_dataset_handle(configuration, modality, partition, dataset_type_="partition")
        source_metadata = constructor.get_source_metadata_handle(configuration, modality, partition, dataset_type_="partition")
        species_list = constructor.read_species_list(dataset)

        i = 0

        observation_count = 0

        for species in species_list:

            metadata_list = constructor.read_metadata_list(dataset, source_metadata, species, dataset_type_="partition", solution_type_=solution_type_)

            for metadata in metadata_list:

                observation_species_list.append(species)
                observation_context = create_context(metadata)
                observation_context_list.append(observation_context)

                observation_count += 1

            print(f"Species: {i}")
            i += 1

        dataset_kwargs = {
            "observation_count": observation_count,
            "observation_species_list": observation_species_list,
            "observation_context_list": observation_context_list
        }

        save_data_to_file_(compiled_dataset_path, dataset_kwargs)


def create_context(metadata):
    latitude = metadata["latitude"]
    longitude = metadata["longitude"]
    date_ = metadata["date"]
    time_ = metadata["time"]
    context = [latitude, longitude, date_, time_]
    return context



