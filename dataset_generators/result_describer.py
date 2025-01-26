from dataset_generators.dataset_constructor import *
from partition_configurations import *
from pair_configurations import *
import matplotlib.pyplot as plt
import numpy as np


def get_result(test_path, test_name, performance_metric_set):

    if os.path.exists(test_path):

        model_result_dictionary = read_data_from_file_(test_path)

        if test_name in model_result_dictionary:

            result_dictionary = {}

            if "observation_accuracy" in performance_metric_set:
                result_dictionary["instance_accuracy"] = format(model_result_dictionary[test_name]["instance_accuracy"], ".4f")
                result_dictionary["observation_accuracy"] = format(model_result_dictionary[test_name]["observation_accuracy"], ".4f")

            if "observation_top_k_accuracy" in performance_metric_set:
                result_dictionary["instance_top_k_accuracy"] = format(model_result_dictionary[test_name]["instance_top_k_accuracy"], ".4f")
                result_dictionary["observation_top_k_accuracy"] = format(model_result_dictionary[test_name]["observation_top_k_accuracy"], ".4f")

            if "observation_mean_reciprocal_rank" in performance_metric_set:
                result_dictionary["instance_mean_reciprocal_rank"] = format(model_result_dictionary[test_name]["instance_mean_reciprocal_rank"], ".4f")
                result_dictionary["observation_mean_reciprocal_rank"] = format(model_result_dictionary[test_name]["observation_mean_reciprocal_rank"], ".4f")

            if "observation_loss" in performance_metric_set:
                result_dictionary["instance_loss"] = format(model_result_dictionary[test_name]["instance_loss"], ".4f")
                result_dictionary["observation_loss"] = format(model_result_dictionary[test_name]["observation_loss"], ".4f")

            if "observation_recall" in performance_metric_set or "observation_recall_dictionary" in performance_metric_set:

                observation_confusion_matrix = model_result_dictionary[test_name]["observation_confusion_matrix"]

                observation_recall_dictionary = {}
                observation_species_size_dictionary = {}
                observation_recall_list = []
                observation_species_size_list = []

                for species_label in observation_confusion_matrix:
                    observation_species_size_dictionary[species_label] = 0
                    for species_prediction in observation_confusion_matrix[species_label]:
                        observation_species_size_dictionary[species_label] += observation_confusion_matrix[species_label][species_prediction]
                    observation_recall_dictionary[species_label] = float(observation_confusion_matrix[species_label][species_label] / observation_species_size_dictionary[species_label])
                    observation_species_size_list.append(observation_species_size_dictionary[species_label])
                    observation_recall_list.append(observation_recall_dictionary[species_label])

                # if "observation_recall_dictionary" in performance_metric_set:
                #     result_dictionary["observation_recall_dictionary"] = observation_recall_dictionary
                #     result_dictionary["observation_species_size_dictionary"] = observation_species_size_dictionary
                #     result_dictionary["observation_recall_list"] = observation_recall_list
                #     result_dictionary["observation_species_size_list"] = observation_species_size_list


            return result_dictionary

        else:

            return 1;

    else:

        return 0



