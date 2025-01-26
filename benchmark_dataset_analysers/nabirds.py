from benchmark_dataset_analysers.benchmark_dataset import Dataset
from utility import *


class NABirds(Dataset):

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self):

        # Paths
        dataset_path = "D:/Masters/Resources/Benchmark Datasets/NABirds"

        # Parent Constructor
        super(NABirds, self).__init__(dataset_path)

    # ******************************************************************************************************************

    def statisticise(self, partition):
        summary_statistics = self.create_statistics_template("summary", modality="image")
        distribution_statistics = self.create_statistics_template("distribution", modality="image")
        if partition == "train" or partition == "test":
            frequency_dictionary = {}
            partition_file_path = f"{self.downloaded_files_path}/train_test_split.txt"
            instance_file_path = f"{self.downloaded_files_path}/images.txt"
            partition_code = 1 if partition == "train" else 0
            instance_partition_list = read_data_from_file_(partition_file_path)
            instance_partition_dictionary = {row.split()[0]: int(row.split()[1]) for row in instance_partition_list}
            instance_file_path_list = read_data_from_file_(instance_file_path)
            instance_count = 0
            for row in instance_file_path_list:
                instance_id = row.split()[0]
                if instance_partition_dictionary[instance_id] == partition_code:
                    instance_path = row.split()[1]
                    instance_class = instance_path.split("/")[0]
                    update_frequency_dictionary_(instance_class, 1, frequency_dictionary)
                    instance_count += 1
            class_count = len(frequency_dictionary)
            alternate_frequency_dictionary = {}
            hierarchy_file_path = f"{self.downloaded_files_path}/hierarchy.txt"
            class_file_path = f"{self.downloaded_files_path}/classes.txt"
            instance_hierarchy_dictionary = {int(row.split()[0]): int(row.split()[1]) for row in read_data_from_file_(hierarchy_file_path)}
            class_dictionary = {int(row.split(None, 1)[0]): row.split(None, 1)[1] for row in read_data_from_file_(class_file_path)}
            alternate_instance_count = 0
            for row in instance_file_path_list:
                instance_id = row.split()[0]
                if instance_partition_dictionary[instance_id] == partition_code:
                    instance_path = row.split()[1]
                    instance_class = int(instance_path.split("/")[0])
                    if "(" in class_dictionary[instance_class]: instance_class = instance_hierarchy_dictionary[instance_class]
                    update_frequency_dictionary_(instance_class, 1, alternate_frequency_dictionary)
                    alternate_instance_count += 1
            alternate_class_count = len(alternate_frequency_dictionary)
            summary_statistics["instance_count"] = instance_count
            summary_statistics["alternate_instance_count"] = alternate_instance_count
            summary_statistics["species_count"] = class_count
            summary_statistics["alternate_class_count"] = alternate_class_count
            distribution_statistics = frequency_dictionary
        self.update_statistics(summary_statistics, "summary", modality="image", partition=partition)
        self.update_statistics(distribution_statistics, "distribution", modality="image", partition=partition)
