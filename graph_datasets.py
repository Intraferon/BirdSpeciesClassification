from datasets import *
from dataset_generators.visualisation import *

dataset_name_list = ["Image", "Audio"]

constructor_list = [get_inaturalist_and_xenocanto_1000_dataset_constructor(),
                    get_inaturalist_and_xenocanto_1000_dataset_constructor()]

dataset_id_list = [("all_0_3_0", "image", "train-validation-test", "partition", "best"),
                   ("all_0_3_0", "audio", "train-validation-test", "partition", "best")]

# source_metadata_id_list = [(3, "image", "train-validation-test"),
#                            (3, "audio", "train-validation-test")]

observation_total_frequency_distribution_graph(dataset_name_list, constructor_list, dataset_id_list)

# observation_total_frequency_distribution_graph(dataset_name_list, constructor_list, dataset_id_list)

# time_density_graph(dataset_name_list, constructor_list, dataset_id_list)
# attribute_distribution_graph(dataset_name_list, constructor_list, dataset_id_list)