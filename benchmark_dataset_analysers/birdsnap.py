from benchmark_dataset_analysers.benchmark_dataset import Dataset
from utility import *


class Birdsnap(Dataset):

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self):

        # Paths
        dataset_path = "D:/Masters/Resources/Benchmark Datasets/Birdsnap"

        # Parent Constructor
        super(Birdsnap, self).__init__(dataset_path)

    # ******************************************************************************************************************

    def statisticise(self, partition):
        summary_statistics = self.create_statistics_template("summary", modality="image")
        distribution_statistics = self.create_statistics_template("distribution", modality="image")
        if partition == "train" or partition == "test":
            frequency_dictionary = {}
            instance_file_path = f"{self.downloaded_files_path}/all-ims.txt"
            test_instance_file_path = f"{self.downloaded_files_path}/test_images.txt"
            instance_file_path_list = read_data_from_file_(instance_file_path)
            test_instance_file_path_set = set(read_data_from_file_(test_instance_file_path)[1:])
            instance_count = 0
            for instance_file_path in instance_file_path_list:
                instance_class = instance_file_path.split("/")[0]
                if instance_file_path in test_instance_file_path_set:
                    if partition == "test":
                        update_frequency_dictionary_(instance_class, 1, frequency_dictionary)
                        instance_count += 1
                else:
                    if partition == "train":
                        update_frequency_dictionary_(instance_class, 1, frequency_dictionary)
                        instance_count += 1
            class_count = len(frequency_dictionary)
            summary_statistics["instance_count"] = instance_count
            summary_statistics["species_count"] = class_count
            distribution_statistics = frequency_dictionary
        self.update_statistics(summary_statistics, "summary", modality="image", partition=partition)
        self.update_statistics(distribution_statistics, "distribution", modality="image", partition=partition)

