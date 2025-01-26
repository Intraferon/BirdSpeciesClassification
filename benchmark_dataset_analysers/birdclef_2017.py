from benchmark_dataset_analysers.benchmark_dataset import Dataset
import audiofile
from utility import *
from xml.etree import ElementTree


class BirdCLEF2017(Dataset):

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self):

        # Paths
        dataset_path = "D:/Masters/Resources/Benchmark Datasets/BirdCLEF 2014 - 2018"

        # Parent Constructor
        super(BirdCLEF2017, self).__init__(dataset_path)

    # ******************************************************************************************************************

    def statisticise(self, audio_type, partition):
        summary_statistics = self.create_statistics_template("summary", modality="audio", audio_type=audio_type, partition=partition)
        distribution_statistics = self.create_statistics_template("distribution", audio_type=audio_type, partition=partition)
        if audio_type == "monophone" and partition == "train":
            summary_statistics["instance_count"], summary_statistics["species_count"], distribution_statistics = self.statisticise_monophone_train()
        if audio_type == "monophone" and partition == "test":
            summary_statistics["instance_count"] = self.statisticise_monophone_test()
        if audio_type == "record_annotated_soundscape" and partition == "test":
            summary_statistics["instance_count"], summary_statistics["soundscape_count"], summary_statistics["instance_duration"], summary_statistics["total_duration"] = self.statisticise_record_annotated_soundscape_test()
        if audio_type == "time_annotated_soundscape" and partition == "validation":
            summary_statistics["instance_count"], summary_statistics["soundscape_count"], summary_statistics["species_count"], summary_statistics["instance_duration"], summary_statistics["total_duration"] = self.statisticise_time_annotated_soundscape_validation()
        if audio_type == "time_annotated_soundscape" and partition == "test":
            summary_statistics["instance_count"], summary_statistics["soundscape_count"], summary_statistics["instance_duration"], summary_statistics["total_duration"] = self.statisticise_time_annotated_soundscape_test()
        self.update_statistics(summary_statistics, "summary", modality="audio", rendition="birdclef_2017", audio_type=audio_type, partition=partition)
        self.update_statistics(distribution_statistics, "distribution", modality="audio", rendition="birdclef_2017", audio_type=audio_type, partition=partition)

    def statisticise_monophone_train(self):
        frequency_dictionary = {}
        data_path = f"{self.downloaded_files_path}/monophone_train/"
        instance_count = 0
        for instance_file_name in os.listdir(data_path):
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
                instance_count += 1
        return instance_count

    @staticmethod
    def statisticise_record_annotated_soundscape_test():
        instance_count = 925
        soundscape_count = 925
        instance_duration = "About 00:10:00"
        total_duration = "About 154:10:00"
        return instance_count, soundscape_count, instance_duration, total_duration

    def statisticise_time_annotated_soundscape_validation(self):
        data_path = f"{self.downloaded_files_path}/soundscape_validation/"
        soundscape_count = 0
        instance_duration = "00:00:05"
        total_duration = 0
        for soundscape_file in os.listdir(data_path):
            if soundscape_file.split(".")[1] != "xml":
                soundscape_file_path = f"{data_path}{soundscape_file}"
                soundscape_duration = audiofile.duration(soundscape_file_path)
                total_duration += soundscape_duration
                soundscape_count += 1
        instance_count = int(total_duration / 5.0)
        total_minutes, total_seconds = divmod(total_duration, 60)
        total_hours, total_minutes = divmod(total_minutes, 60)
        total_duration = f"{str(int(total_hours)).zfill(2)}:{str(int(total_minutes)).zfill(2)}:{str(int(total_seconds)).zfill(2)}"
        class_name_set = set()
        for soundscape_file in os.listdir(data_path):
            if soundscape_file.split(".")[1] == "xml":
                soundscape_file_path = f"{data_path}{soundscape_file}"
                soundscape_data_tree = ElementTree.parse(soundscape_file_path)
                soundscape_data_root = soundscape_data_tree.getroot()
                for soundscape_data_child in soundscape_data_root:
                    if soundscape_data_child.tag == "ClassIds":
                        for soundscape_class_child in soundscape_data_child:
                            for soundscape_class_data_child in soundscape_class_child:
                                if soundscape_class_data_child.tag == "ClassId":
                                    class_name = soundscape_class_data_child.text
                                    class_name_set.add(class_name)
        class_count = len(class_name_set)
        return instance_count, soundscape_count, class_count, instance_duration, total_duration

    def statisticise_time_annotated_soundscape_test(self):
        data_path = f"{self.downloaded_files_path}/soundscape_test/"
        soundscape_count = 0
        instance_duration = "00:00:05"
        total_duration = 0
        for soundscape_file in os.listdir(data_path):
            if soundscape_file.split(".")[1] != "xml":
                soundscape_file_path = f"{data_path}{soundscape_file}"
                soundscape_duration = audiofile.duration(soundscape_file_path)
                total_duration += soundscape_duration
                soundscape_count += 1
        instance_count = int(total_duration / 5.0)
        total_minutes, total_seconds = divmod(total_duration, 60)
        total_hours, total_minutes = divmod(total_minutes, 60)
        total_duration = f"{str(int(total_hours)).zfill(2)}:{str(int(total_minutes)).zfill(2)}:{str(int(total_seconds)).zfill(2)}"
        return instance_count, soundscape_count, instance_duration, total_duration
