import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from instance_generators.instance_generator import InstanceGenerator
from dataset_generators.dataset_loader import *
from models.model import Model
from experiments.experiment_setup import *
import pprint


class Experiment:

    def __init__(self, modality, experiment_model, experiment_kwargs):
        self.modality = modality
        self.experiment_model = experiment_model
        self.experiment_kwargs = experiment_kwargs

    def train(self, experiment_dataset, epoch_step, overwrite_stage=None, overwrite_dataset=False):

        partition_list = ["train", "validation"]

        print("Constructing datasets...")

        for partition in partition_list:
            construct_general_dataset(self.get_subset_dictionary(partition),
                                      self.experiment_kwargs["constructor_dictionary"][partition],
                                      experiment_dataset,
                                      partition,
                                      self.experiment_kwargs["modality_dictionary"][partition],
                                      overwrite=overwrite_dataset)

        print("Loading datasets...")

        dataset_kwargs_dictionary = {partition: load_dataset(self.experiment_kwargs["constructor_dictionary"][partition],
                                                             experiment_dataset,
                                                             partition,
                                                             self.experiment_kwargs["modality_dictionary"][partition],
                                                             self.experiment_kwargs["context_parameters"],
                                                             self.experiment_kwargs["augmentation_parameters"])
                                     for partition in partition_list}

        print("Setting up model...")

        self.experiment_model.setup(self.experiment_kwargs["maximum_epoch_count"], overwrite_stage=overwrite_stage, train=True, **self.experiment_kwargs["model_kwargs"])

        epoch_count = self.experiment_model.get_progress()[1]

        print("Creating instance generators...")

        instance_generator_dictionary = {partition: InstanceGenerator(self.get_subset_dictionary(partition),
                                                                      self.experiment_kwargs["modality_dictionary"][partition],
                                                                      partition,
                                                                      self.experiment_kwargs["species_count"],
                                                                      epoch_count,
                                                                      self.experiment_kwargs["epoch_cycle_length"],
                                                                      **dataset_kwargs_dictionary[partition])
                                         for partition in partition_list}

        print("Training model...")

        self.experiment_model.train(epoch_step, instance_generator_dictionary["train"], instance_generator_dictionary["validation"])


    def test(self, experiment_dataset, experiment_dataset_i, model_stage, overwrite_dataset=False):

        partition = "test"

        print("Constructing dataset...")

        construct_general_dataset(self.get_subset_dictionary(partition, experiment_dataset_i),
                                  self.experiment_kwargs["constructor_dictionary"][partition][experiment_dataset_i],
                                  experiment_dataset,
                                  self.experiment_kwargs["experiment_test_partition"][experiment_dataset_i],
                                  self.experiment_kwargs["modality_dictionary"][partition],
                                  overwrite=overwrite_dataset)

        print("Loading dataset...")

        dataset_kwargs = load_dataset(self.experiment_kwargs["constructor_dictionary"][partition][experiment_dataset_i],
                                      experiment_dataset,
                                      self.experiment_kwargs["experiment_test_partition"][experiment_dataset_i],
                                      self.experiment_kwargs["modality_dictionary"][partition],
                                      self.experiment_kwargs["context_parameters"],
                                      self.experiment_kwargs["augmentation_parameters"])

        print("Creating instance generator...")

        instance_generator = InstanceGenerator(self.get_subset_dictionary(partition, experiment_dataset_i),
                                               self.experiment_kwargs["modality_dictionary"][partition],
                                               partition,
                                               self.experiment_kwargs["species_count"],
                                               None,
                                               None,
                                               **dataset_kwargs)

        self.experiment_model.setup()

        test_name = f"{self.experiment_kwargs['constructor_dictionary'][partition][experiment_dataset_i].dataset_name} {self.experiment_kwargs['modality_dictionary'][partition].title()} ({experiment_dataset}) ({self.experiment_kwargs['experiment_test_partition'][experiment_dataset_i].title()}) (Stage: {model_stage})"

        self.experiment_model.test(test_name, model_stage, instance_generator)


    def get_subset_dictionary(self, partition, experiment_dataset_i=None):

        if partition != "test":

            if self.modality == "image" or self.modality == "image-context":
                subset_dictionary = self.experiment_kwargs["subset_dictionary"][partition][0]
            if self.modality == "audio" or self.modality == "audio-context":
                subset_dictionary = self.experiment_kwargs["subset_dictionary"][partition][1]
            if self.modality == "joint" or self.modality == "joint-context":
                subset_dictionary = self.experiment_kwargs["subset_dictionary"][partition]

        else:

            if self.modality == "image" or self.modality == "image-context":
                subset_dictionary = self.experiment_kwargs["subset_dictionary"][partition][experiment_dataset_i][0]
            if self.modality == "audio" or self.modality == "audio-context":
                subset_dictionary = self.experiment_kwargs["subset_dictionary"][partition][experiment_dataset_i][1]
            if self.modality == "joint" or self.modality == "joint-context":
                subset_dictionary = self.experiment_kwargs["subset_dictionary"][partition][experiment_dataset_i]

        return subset_dictionary
