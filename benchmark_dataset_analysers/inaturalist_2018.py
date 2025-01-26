from benchmark_dataset_analysers.benchmark_dataset import Dataset
from utility import *


class INaturalist2018(Dataset):

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self):

        # Paths
        dataset_path = "D:/Masters/Resources/Benchmark Datasets/iNaturalist 2018"

        # Parent Constructor
        super(INaturalist2018, self).__init__(dataset_path)

    # ******************************************************************************************************************

    def statisticise(self, partition):
        summary_statistics = self.create_statistics_template("summary", modality="image")
        distribution_statistics = self.create_statistics_template("distribution", modality="image")
        if partition == "train" or partition == "validation":
            frequency_dictionary = {}
            instance_file_name = "train2018.json" if partition == "train" else "val2018.json"
            instance_file_path = f"{self.downloaded_files_path}/{instance_file_name}"
            instance_data_list = read_data_from_file_(instance_file_path)["images"]
            instance_count = 0
            for instance_data in instance_data_list:
                instance_path = instance_data["file_name"]
                instance_class = instance_path.split("/")[2]
                update_frequency_dictionary_(instance_class, 1, frequency_dictionary)
                instance_count += 1
            class_count = len(frequency_dictionary)
            alternate_frequency_dictionary = {}
            alternate_instance_count = 0
            for instance_data in instance_data_list:
                instance_path = instance_data["file_name"]
                instance_parent_class = instance_path.split("/")[1]
                if instance_parent_class == "Aves":
                    instance_class = instance_path.split("/")[2]
                    update_frequency_dictionary_(instance_class, 1, alternate_frequency_dictionary)
                    alternate_instance_count += 1
            alternate_class_count = len(alternate_frequency_dictionary)
            summary_statistics["instance_count"] = instance_count
            summary_statistics["alternate_instance_count"] = alternate_instance_count
            summary_statistics["species_count"] = class_count
            summary_statistics["alternate_class_count"] = alternate_class_count
            distribution_statistics = frequency_dictionary
        if partition == "test":
            summary_statistics["instance_count"] = 149394
            summary_statistics["species_count"] = 8142
            summary_statistics["alternate_class_count"] = 1258
        self.update_statistics(summary_statistics, "summary", modality="image", partition=partition)
        self.update_statistics(distribution_statistics, "distribution", modality="image", partition=partition)
