import pprint

from models.context_ann import ContextANN
from models.image_cnn import ImageCNN
from models.audio_cnn import AudioCNN
from models.image_context_cnn import ImageContextCNN
from models.audio_context_cnn import AudioContextCNN
from models.joint_cnn import JointCNN
from models.joint_context_cnn import JointContextCNN
from models.model import Model
from experiments.experiment import Experiment
from experiments.experiment_setup import setup
from dataset_generators.result_describer import *
from dataset_generators.visualisation import *
import warnings


# warnings.filterwarnings("ignore")


def get_experiment_dataset(experiment_id):

    modality, image_partition_configuration, audio_partition_configuration, pair_configuration, image_augmentation_configuration, audio_augmentation_configuration, context_dimensions, context = parse_experiment_id(experiment_id)

    if modality == "image":

        experiment_dataset = image_partition_configuration

    elif modality == "audio":

        experiment_dataset = audio_partition_configuration

    else:

        if pair_configuration != "only_context":

            experiment_dataset = (f"{image_partition_configuration}_{audio_partition_configuration}_{pair_configuration}",
                                  image_partition_configuration,
                                  audio_partition_configuration,
                                  pair_configuration)

        else:

            experiment_dataset = pair_configuration

    return experiment_dataset


# {modality}_{image_partition_configuration}_{audio_partition_configuration}_!{pair_configuration}_{augmentation_configuration}_!{context_dimensions}

def get_experiment(experiment_id):

    species_count = 1000

    modality, image_partition_configuration, audio_partition_configuration, pair_configuration, image_augmentation_configuration, audio_augmentation_configuration, context_dimensions, context = parse_experiment_id(experiment_id)

    if context_dimensions != 0:

        context_experiment_kwargs = setup(modality=modality, context_dimensions=context_dimensions, context=context)

        context_ann_kwargs = {"context_ann_input_dimensions": context_experiment_kwargs["context_ann_dimensions"]}
        context_ann = ContextANN(context_experiment_kwargs["species_count"], **context_ann_kwargs)

        if modality == "image":

            context_experiment_name = f"{species_count}_image_only_context_{context_dimensions}"
            context_model = Model(context_experiment_name, context_ann)

            if context:
                experiment = Experiment("image", context_model, context_experiment_kwargs)

        if modality == "audio":

            context_experiment_name = f"{species_count}_audio_only_context_{context_dimensions}"
            context_model = Model(context_experiment_name, context_ann)

            if context:
                experiment = Experiment("audio", context_model, context_experiment_kwargs)

        if modality == "joint":

            context_experiment_name = f"{species_count}_joint_only_context_{context_dimensions}"
            context_model = Model(context_experiment_name, context_ann)

            if context:
                experiment = Experiment("joint", context_model, context_experiment_kwargs)

    if not context:

        if modality == "image" or modality == "joint":

            image_experiment_name = f"{species_count}_image_{image_partition_configuration}_{image_augmentation_configuration}"
            image_experiment_kwargs = setup(modality="image", augmentation_configuration=image_augmentation_configuration)
            image_cnn = ImageCNN(image_experiment_kwargs["species_count"])
            image_model = Model(image_experiment_name, image_cnn)

            if modality == "image":

                if context_dimensions != 0:

                    image_context_experiment_name = f"{species_count}_image_{image_partition_configuration}_{image_augmentation_configuration}_{context_dimensions}"
                    image_context_experiment_kwargs = setup(modality="image", context_dimensions=context_dimensions, augmentation_configuration=image_augmentation_configuration)
                    image_context_cnn_kwargs = {"cnn_path": image_model.get_path(),
                                                "base_cnn_path": image_cnn.base_model_path,
                                                "context_ann_path": context_model.get_path(),
                                                "base_context_ann_path": context_ann.base_model_path}
                    image_context_cnn = ImageContextCNN(image_context_experiment_kwargs["species_count"], **image_context_cnn_kwargs)
                    image_context_model = Model(image_context_experiment_name, image_context_cnn)
                    experiment = Experiment("image", image_context_model, image_context_experiment_kwargs)

                else:

                    experiment = Experiment("image", image_model, image_experiment_kwargs)

        if modality == "audio" or modality == "joint":

            audio_experiment_name = f"{species_count}_audio_{audio_partition_configuration}_{audio_augmentation_configuration}"
            audio_experiment_kwargs = setup(modality="audio", augmentation_configuration=audio_augmentation_configuration)
            audio_cnn = AudioCNN(audio_experiment_kwargs["species_count"])
            audio_model = Model(audio_experiment_name, audio_cnn)

            if modality == "audio":

                if context_dimensions != 0:

                    audio_context_experiment_name = f"{species_count}_audio_{audio_partition_configuration}_{audio_augmentation_configuration}_{context_dimensions}"
                    audio_context_experiment_kwargs = setup(modality="audio", context_dimensions=context_dimensions, augmentation_configuration=audio_augmentation_configuration)
                    audio_context_cnn_kwargs = {"cnn_path": audio_model.get_path(),
                                                "base_cnn_path": audio_cnn.base_model_path,
                                                "context_ann_path": context_model.get_path(),
                                                "base_context_ann_path": context_ann.base_model_path}
                    audio_context_cnn = AudioContextCNN(audio_context_experiment_kwargs["species_count"], **audio_context_cnn_kwargs)
                    audio_context_model = Model(audio_context_experiment_name, audio_context_cnn)
                    experiment = Experiment("audio", audio_context_model, audio_context_experiment_kwargs)

                else:

                    experiment = Experiment("audio", audio_model, audio_experiment_kwargs)

        if modality == "joint":

            joint_experiment_name = f"{species_count}_joint_{image_partition_configuration}_{audio_partition_configuration}_{pair_configuration}"
            joint_experiment_kwargs = setup(modality="joint")
            joint_cnn_kwargs = {"image_cnn_path": image_model.get_path(), "audio_cnn_path": audio_model.get_path()}
            joint_cnn = JointCNN(joint_experiment_kwargs["species_count"], **joint_cnn_kwargs)
            joint_model = Model(joint_experiment_name, joint_cnn)

            if context_dimensions != 0:

                joint_context_experiment_name = f"{species_count}_joint_{image_partition_configuration}_{audio_partition_configuration}_{pair_configuration}_{context_dimensions}"
                joint_context_experiment_kwargs = setup(modality="joint", context_dimensions=context_dimensions)
                joint_context_cnn_kwargs = {"image_cnn_path": image_model.get_path(),
                                            "audio_cnn_path": audio_model.get_path(),
                                            "context_ann_path": context_model.get_path()}
                joint_context_cnn = JointContextCNN(joint_context_experiment_kwargs["species_count"], **joint_context_cnn_kwargs)
                joint_context_model = Model(joint_context_experiment_name, joint_context_cnn)
                experiment = Experiment("joint", joint_context_model, joint_context_experiment_kwargs)

            else:

                experiment = Experiment("joint", joint_model, joint_experiment_kwargs)

    return experiment


def parse_experiment_id(experiment_id):

    modality = experiment_id[0]
    image_partition_configuration = None
    audio_partition_configuration = None
    pair_configuration = None
    image_augmentation_configuration = None
    audio_augmentation_configuration = None
    context_dimensions = 0
    context = False

    if modality == "image":
        image_partition_configuration = experiment_id[1]
        if image_partition_configuration == "only_context":
            base_experiment_id_length = 2
            context = True
        else:
            image_augmentation_configuration = experiment_id[2]
            base_experiment_id_length = 3

    if modality == "audio":
        audio_partition_configuration = experiment_id[1]
        if audio_partition_configuration == "only_context":
            base_experiment_id_length = 2
            context = True
        else:
            audio_augmentation_configuration = experiment_id[2]
            base_experiment_id_length = 3

    if modality == "joint":
        pair_configuration = experiment_id[1]
        if pair_configuration == "only_context":
            base_experiment_id_length = 2
            context = True
        else:
            image_partition_configuration = experiment_id[1]
            audio_partition_configuration = experiment_id[2]
            pair_configuration = experiment_id[3]
            image_augmentation_configuration = 2
            audio_augmentation_configuration = 1
            base_experiment_id_length = 4

    if len(experiment_id) > base_experiment_id_length:
        context_dimensions = experiment_id[base_experiment_id_length]

    return modality, image_partition_configuration, audio_partition_configuration, pair_configuration, image_augmentation_configuration, audio_augmentation_configuration, context_dimensions, context


def train(experiment_id, epoch_step, overwrite_stage, overwrite_dataset):
    experiment_dataset = get_experiment_dataset(experiment_id)
    experiment = get_experiment(experiment_id)

    if overwrite_stage != None:
        experiment.train(experiment_dataset, epoch_step, overwrite_stage=overwrite_stage, overwrite_dataset=overwrite_dataset)
    else:
        experiment.train(experiment_dataset, epoch_step, overwrite_dataset=overwrite_dataset)


def test(train_experiment_id, test_experiment_id, experiment_dataset_i, model_stage, overwrite_dataset):
    experiment_dataset = get_experiment_dataset(test_experiment_id)
    experiment = get_experiment(train_experiment_id)

    experiment.test(experiment_dataset, experiment_dataset_i, model_stage, overwrite_dataset=overwrite_dataset)


def describe(experiment_id, experiment_dataset_i, model_stage, suffix = None):
    performance_metric_set = {"observation_accuracy", "observation_loss", "observation_top_k_accuracy", "observation_mean_reciprocal_rank", "observation_recall_dictionary"}

    experiment_dataset = get_experiment_dataset(experiment_id)
    experiment = get_experiment(experiment_id)

    if suffix is None:
        test_path = experiment.experiment_model.model_result_path
    else:
        test_path = f"{experiment.experiment_model.model_result_path.split('.')[0]}_{suffix}.json"
    test_name = f"{experiment.experiment_kwargs['constructor_dictionary']['test'][experiment_dataset_i].dataset_name} {experiment.experiment_kwargs['modality_dictionary']['test'].title()} ({experiment_dataset}) ({experiment.experiment_kwargs['experiment_test_partition'][experiment_dataset_i].title()}) (Stage: {model_stage})"

    result_dictionary = get_result(test_path, test_name, performance_metric_set)

    return result_dictionary


# experiment_id_ = ("joint", "all_0_3_0", "all_0_3_0", "all_0_0", 2)
# epoch_step_ = 4
# overwrite_stage_ = 1
# overwrite_dataset_ = False
# train(experiment_id_, epoch_step_, overwrite_stage_, overwrite_dataset_)

# train_experiment_id_ = ("joint", "all_0_3_0", "all_0_3_0", "all_0_0", 1)
# test_experiment_id_ =  ("joint", "all_0_3_0", "all_0_3_0", "all_0_0", 1)
# experiment_dataset_i_ = 1
# model_stage_ = 1
# overwrite_dataset_ = False
# test(train_experiment_id_, test_experiment_id_, experiment_dataset_i_, model_stage_, overwrite_dataset_)

# print("IMAGE")
# print("---------------------------------------------------------------------------------------")
# experiment_id_ = ("image", "all_0_3_0", 2)
# experiment_dataset_i_ = 1
# model_stage_ = 2
# result_dictionary = describe(experiment_id_, experiment_dataset_i_, model_stage_)
# print(f"Configuration: {experiment_id_} ({model_stage_})")
# pprint.pprint(result_dictionary)
# print("---------------------------------------------------------------------------------------")
#
#
print("AUDIO")
print("---------------------------------------------------------------------------------------")
experiment_id_ = ("audio", "all_0_3_0", 1)
experiment_dataset_i_ = 1
model_stage_ = 2
result_dictionary = describe(experiment_id_, experiment_dataset_i_, model_stage_)
print(f"Configuration: {experiment_id_} ({model_stage_})")
pprint.pprint(result_dictionary)
print("---------------------------------------------------------------------------------------")
experiment_id_ = ("audio", "all_0_3_0", 1, 1)
experiment_dataset_i_ = 1
model_stage_ = 2
result_dictionary = describe(experiment_id_, experiment_dataset_i_, model_stage_)
print(f"Configuration: {experiment_id_} ({model_stage_})")
pprint.pprint(result_dictionary)
print("---------------------------------------------------------------------------------------")
experiment_id_ = ("audio", "all_0_3_0", 1, 2)
experiment_dataset_i_ = 1
model_stage_ = 2
result_dictionary = describe(experiment_id_, experiment_dataset_i_, model_stage_)
print(f"Configuration: {experiment_id_} ({model_stage_})")
pprint.pprint(result_dictionary)
print("---------------------------------------------------------------------------------------")
experiment_id_ = ("audio", "all_0_3_0", 1, 3)
experiment_dataset_i_ = 1
model_stage_ = 2
result_dictionary = describe(experiment_id_, experiment_dataset_i_, model_stage_)
print(f"Configuration: {experiment_id_} ({model_stage_})")
pprint.pprint(result_dictionary)


# experiment_id_ = ("joint", "all_0_3_0", "all_0_3_0", "all_0_0")
# experiment_dataset_i_ = 0
# model_stage_ = 1
# n_result_dictionary = describe(experiment_id_, experiment_dataset_i_, model_stage_)
#
# experiment_id_ = ("joint", "all_0_3_0", "all_0_3_0", "all_0_0", 1)
# experiment_dataset_i_ = 1
# model_stage_ = 1
# l_result_dictionary = describe(experiment_id_, experiment_dataset_i_, model_stage_)
#
# experiment_id_ = ("joint", "all_0_3_0", "all_0_3_0", "all_0_0", 2)
# experiment_dataset_i_ = 1
# model_stage_ = 1
# ld_result_dictionary = describe(experiment_id_, experiment_dataset_i_, model_stage_)
#
# experiment_id_ = ("joint", "all_0_3_0", "all_0_3_0", "all_0_0", 3)
# experiment_dataset_i_ = 1
# model_stage_ = 1
# ldt_result_dictionary = describe(experiment_id_, experiment_dataset_i_, model_stage_)

# observation_recall_graph(["None", "Location", "Location / Date", "Location / Date / Time"],
#                          [n_result_dictionary["observation_species_size_list"], l_result_dictionary["observation_species_size_list"], ld_result_dictionary["observation_species_size_list"], ldt_result_dictionary["observation_species_size_list"]],
#                          [n_result_dictionary["observation_recall_list"], l_result_dictionary["observation_recall_list"], ld_result_dictionary["observation_recall_list"], ldt_result_dictionary["observation_recall_list"]])

