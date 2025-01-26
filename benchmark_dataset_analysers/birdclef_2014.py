from benchmark_dataset_analysers.benchmark_dataset import Dataset
from utility import *
from xml.etree import ElementTree


class BirdCLEF2014(Dataset):

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self):

        # Paths
        dataset_path = "D:/Masters/Resources/Benchmark Datasets/BirdCLEF 2014 - 2018"

        # Parent Constructor
        super(BirdCLEF2014, self).__init__(dataset_path)

    # ******************************************************************************************************************

    def statisticise(self, audio_type, partition):
        summary_statistics = self.create_statistics_template("summary", modality="audio", audio_type=audio_type, partition=partition)
        distribution_statistics = self.create_statistics_template("distribution", audio_type=audio_type, partition=partition)
        if audio_type == "monophone" and partition == "train":
            summary_statistics["instance_count"], summary_statistics["species_count"], distribution_statistics = self.statisticise_monophone_train()
        if audio_type == "monophone" and partition == "test":
            summary_statistics["instance_count"] = self.statisticise_monophone_test()
        self.update_statistics(summary_statistics, "summary", modality="audio", rendition="birdclef_2014", audio_type=audio_type, partition=partition)
        self.update_statistics(distribution_statistics, "distribution", modality="audio", rendition="birdclef_2014", audio_type=audio_type, partition=partition)

    def statisticise_monophone_train(self):
        frequency_dictionary = {}
        data_path = f"{self.downloaded_files_path}/monophone_train/"
        instance_count = 0
        for instance_file_name in os.listdir(data_path):
            rendition = instance_file_name.split("_")[0]
            if rendition == "LIFECLEF2014":
                instance_file_path = f"{data_path}{instance_file_name}"
                with open(instance_file_path, "r", encoding="utf8") as instance_file:
                    instance_data_tree = ElementTree.parse(instance_file)
                    instance_data_root = instance_data_tree.getroot()
                    for instance_data_child in instance_data_root:
                        if instance_data_child.tag == "VernacularNames":
                            instance_class_name = instance_data_child.text
                    frequency_dictionary = update_frequency_dictionary_(instance_class_name, 1, frequency_dictionary)
                    instance_count += 1
        class_count = len(frequency_dictionary)
        return instance_count, class_count, frequency_dictionary

    def statisticise_monophone_test(self):
        data_path = f"{self.downloaded_files_path}/monophone_test/"
        instance_count = 0
        for instance_file_name in os.listdir(data_path):
            if instance_file_name.split(".")[1] == "xml":
                rendition = instance_file_name.split("_")[0]
                if rendition == "LIFECLEF2014":
                    instance_count += 1
        return instance_count


