from benchmark_dataset_analysers.benchmark_dataset import Dataset
from utility import *


class CUB2002010(Dataset):

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self):

        # Paths
        dataset_path = "D:/Masters/Resources/Benchmark Datasets/CUB-200-2010"

        # Parent Constructor
        super(CUB2002010, self).__init__(dataset_path)

    # ******************************************************************************************************************

    def statisticise(self, partition):
        summary_statistics = self.create_statistics_template("summary", modality="image")
        distribution_statistics = self.create_statistics_template("distribution", modality="image")
        if partition == "train" or partition == "test":
            frequency_dictionary = {}
            instance_file_path = f"{self.downloaded_files_path}/{partition}.txt"
            instance_file_path_list = read_data_from_file_(instance_file_path)
            instance_count = 0
            for instance_path in instance_file_path_list:
                instance_class = instance_path.split("/")[0]
                update_frequency_dictionary_(instance_class, 1, frequency_dictionary)
                instance_count += 1
            class_count = len(frequency_dictionary)
            summary_statistics["instance_count"] = instance_count
            summary_statistics["species_count"] = class_count
            distribution_statistics = frequency_dictionary
        self.update_statistics(summary_statistics, "summary", modality="image", partition=partition)
        self.update_statistics(distribution_statistics, "distribution", modality="image", partition=partition)
