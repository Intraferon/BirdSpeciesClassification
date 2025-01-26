from datasets import *
from subsets import *
from dataset_generators.constants import *


def setup(modality=None, context_dimensions=None, augmentation_configuration=None, context=False):

    experiment_kwargs = {}

    experiment_kwargs["context_parameters"] = get_context_parameters(context_dimensions)

    experiment_kwargs["context_ann_dimensions"] = get_context_ann_dimensions(experiment_kwargs["context_parameters"])

    if not context:
        experiment_kwargs["augmentation_parameters"] = get_augmentation_parameters(modality, augmentation_configuration)
    else:
        experiment_kwargs["augmentation_parameters"] = None

    # ------------------------------------------------------------------------------------------------------------------

    if modality == "image":

        experiment_kwargs["modality_dictionary"] = {"train": "image",
                                                    "validation": "image",
                                                    "test": "image"}

    if modality == "audio":

        experiment_kwargs["modality_dictionary"] = {"train": "audio",
                                                    "validation": "audio",
                                                    "test": "audio"}

    if modality == "joint":

        experiment_kwargs["modality_dictionary"] = {"train": "image-audio",
                                                    "validation": "image-audio",
                                                    "test": "image-audio"}

    # ------------------------------------------------------------------------------------------------------------------

    experiment_kwargs["species_count"] = 1000

    experiment_kwargs["constructor_dictionary"] = {"train": get_inaturalist_and_xenocanto_1000_dataset_constructor(),
                                                   "validation": get_inaturalist_and_xenocanto_1000_dataset_constructor(),
                                                   "test": [get_inaturalist_and_xenocanto_1000_dataset_constructor(),
                                                            get_inaturalist_and_xenocanto_1000_dataset_constructor()]}

    experiment_kwargs["subset_dictionary"] = {"train": [get_inaturalist("Image"), get_xenocanto()],
                                              "validation": [get_inaturalist("Image"), get_xenocanto()],
                                              "test": [[get_inaturalist("Image"), get_xenocanto()],
                                                       [get_inaturalist("Image"), get_xenocanto()]]}

    experiment_kwargs["experiment_test_partition"] = ["validation", "test"]

    experiment_kwargs["epoch_cycle_length"] = 4

    if modality == "image":

        if not context:

            if context_dimensions == 0 or context_dimensions is None:
                experiment_kwargs["maximum_epoch_count"] = get_maximum_epoch_count("image")
                experiment_kwargs["model_kwargs"] = get_model_kwargs("image")

            else:
                experiment_kwargs["maximum_epoch_count"] = get_maximum_epoch_count("image-context")
                experiment_kwargs["model_kwargs"] = get_model_kwargs("image-context")

        else:

            experiment_kwargs["maximum_epoch_count"] = get_maximum_epoch_count("context")
            experiment_kwargs["model_kwargs"] = get_model_kwargs("context")

    if modality == "audio":

        if not context:

            if context_dimensions == 0 or context_dimensions is None:
                experiment_kwargs["maximum_epoch_count"] = get_maximum_epoch_count("audio")
                experiment_kwargs["model_kwargs"] = get_model_kwargs("audio")

            else:
                experiment_kwargs["maximum_epoch_count"] = get_maximum_epoch_count("audio-context")
                experiment_kwargs["model_kwargs"] = get_model_kwargs("audio-context")

        else:

            experiment_kwargs["maximum_epoch_count"] = get_maximum_epoch_count("context")
            experiment_kwargs["model_kwargs"] = get_model_kwargs("context")

    if modality == "joint":

        if not context:

            if context_dimensions == 0 or context_dimensions is None:
                experiment_kwargs["maximum_epoch_count"] = get_maximum_epoch_count("image-audio")
                experiment_kwargs["model_kwargs"] = get_model_kwargs("image-audio")

            else:
                experiment_kwargs["maximum_epoch_count"] = get_maximum_epoch_count("image-audio-context")
                experiment_kwargs["model_kwargs"] = get_model_kwargs("image-audio-context")

        else:

            experiment_kwargs["maximum_epoch_count"] = get_maximum_epoch_count("context")
            experiment_kwargs["model_kwargs"] = get_model_kwargs("context")

    return experiment_kwargs


def get_maximum_epoch_count(type_):

    if type_ == "image" or type_ == "audio":

        maximum_epoch_count = [0, 12, 100]

    elif type_ == "image-audio" or type_ == "image-context" or type_ == "audio-context" or type_ == "image-audio-context":

        maximum_epoch_count = [0, 100, 100]

    elif type_ == "context":

        maximum_epoch_count = [100, 0, 0]

    else:

        maximum_epoch_count = None

    return maximum_epoch_count


def get_model_kwargs(type_):

    if type_ == "image" or type_ == "audio":

        model_kwargs = {

            "learning_rate_schedule":

                {1: {"monitor": "val_categorical_accuracy",
                     "initial_learning_rate": 1.0e-4,
                     "end_learning_rate": 1.5625e-5,
                     "decay_rate": 0.5,
                     "min_delta": 0.01,
                     "patience": 0},
                 2: {"monitor": "val_categorical_accuracy",
                     "initial_learning_rate": 1.0e-5,
                     "end_learning_rate": 1.5625e-7,
                     "decay_rate": 0.5,
                     "min_delta": 0.01,
                     "patience": 2}},

            "early_stopping":

                {1: {"monitor": "val_categorical_accuracy",
                     "min_delta": 0.005,
                     "patience": 3},
                 2: {"monitor": "val_categorical_accuracy",
                     "min_delta": 0.005,
                     "patience": 4}}
        }

    elif type_ == "image-audio" or type_ == "image-audio-context":

        model_kwargs = {

            "learning_rate_schedule":

                {1: {"monitor": "val_categorical_accuracy",
                     "initial_learning_rate": 1.0e-7,
                     "end_learning_rate": 1.0e-8,
                     "decay_rate": 0.5,
                     "min_delta": 0.002,
                     "patience": 0},
                 2: {"monitor": "val_categorical_accuracy",
                     "initial_learning_rate": 2.5e-7,
                     "end_learning_rate": 1.1e-8,
                     "decay_rate": 0.8,
                     "min_delta": 0.002,
                     "patience": 0},
                 },

            "early_stopping":

                {1: {"monitor": "val_categorical_accuracy",
                     "min_delta": 0.001,
                     "patience": 4},
                 2: {"monitor": "val_categorical_accuracy",
                     "min_delta": 0.001,
                     "patience": 4}
                 }
        }

    elif type_ == "image-context" or type_ == "audio-context":

        model_kwargs = {

            "learning_rate_schedule":

                {1: {"monitor": "val_categorical_accuracy",
                     "initial_learning_rate": 2.5e-7,
                     "end_learning_rate": 1.1e-9,
                     "decay_rate": 0.5,
                     "min_delta": 0.002,
                     "patience": 0},
                 2: {"monitor": "val_categorical_accuracy",
                     "initial_learning_rate": 2.5e-7,
                     "end_learning_rate": 1.1e-8,
                     "decay_rate": 0.8,
                     "min_delta": 0.002,
                     "patience": 0},
                 },

            "early_stopping":

                {1: {"monitor": "val_categorical_accuracy",
                     "min_delta": 0.001,
                     "patience": 4},
                 2: {"monitor": "val_categorical_accuracy",
                     "min_delta": 0.001,
                     "patience": 4}
                 }
        }

    elif type_ == "context":

        model_kwargs = {

            "learning_rate_schedule":

                {0: {"monitor": "val_categorical_accuracy",
                     "initial_learning_rate": 1.0e-3,
                     "end_learning_rate": 1.5625e-7,
                     "decay_rate": 0.5,
                     "min_delta": 0.001,
                     "patience": 1}},

            "early_stopping":

                {0: {"monitor": "val_categorical_accuracy",
                     "min_delta": 0.0001,
                     "patience": 6}}
        }

    else:

        model_kwargs = None

    return model_kwargs


def get_context_parameters(context_dimensions):
    if context_dimensions == 0:
        context_parameters = None
    elif context_dimensions == 1:
        context_parameters = {"include_location": True,
                              "include_date": False,
                              "include_time": False}
    elif context_dimensions == 2:
        context_parameters = {"include_location": True,
                              "include_date": True,
                              "include_time": False}
    elif context_dimensions == 3:
        context_parameters = {"include_location": True,
                              "include_date": True,
                              "include_time": True}
    else:
        context_parameters = None
    return context_parameters


def get_context_ann_dimensions(context_parameters):
    context_dimensions = 0
    if context_parameters is not None:
        if context_parameters["include_location"]:
            context_dimensions += 4
        if context_parameters["include_date"]:
            context_dimensions += 2
        if context_parameters["include_time"]:
            context_dimensions += 2
    return context_dimensions


def get_augmentation_parameters(modality, augmentation_configuration):

    if modality == "image":

        if augmentation_configuration == 0:
            augmentation_parameters = {"random_crop": False,
                                       "center_crop": False}
        elif augmentation_configuration == 1:
            augmentation_parameters = {"random_crop": True,
                                       "center_crop": False}
        elif augmentation_configuration == 2:
            augmentation_parameters = {"random_crop": True,
                                       "center_crop": True}
        else:
            augmentation_parameters = {"random_crop": True,
                                       "center_crop": True}

    if modality == "audio":

        if augmentation_configuration == 0:
            augmentation_parameters = {"noise_addition": False}
        elif augmentation_configuration == 1:
            augmentation_parameters = {"noise_addition": True}
        else:
            augmentation_parameters = {"noise_addition": True}

    if modality == "joint":

        augmentation_parameters = {"random_crop": True,
                                   "center_crop": True,
                                   "noise_addition": True}

    return augmentation_parameters
