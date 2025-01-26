from benchmark_dataset_analysers.benchmark_dataset import Dataset
import audiofile
from utility import *


class BirdCLEF2019(Dataset):

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self):

        # Paths
        dataset_path = "D:/Masters/Resources/Benchmark Datasets/BirdCLEF 2019"

        # Parent Constructor
        super(BirdCLEF2019, self).__init__(dataset_path)

    # ******************************************************************************************************************

    def statisticise(self, audio_type, partition):
        summary_statistics = self.create_statistics_template("summary", modality="audio", audio_type=audio_type, partition=partition)
        distribution_statistics = self.create_statistics_template("distribution", audio_type=audio_type, partition=partition)
        if audio_type == "monophone" and partition == "train":
            summary_statistics["instance_count"], summary_statistics["species_count"], distribution_statistics = self.statisticise_monophone_train()
        if audio_type == "time_annotated_soundscape" and partition == "validation":
            summary_statistics["instance_count"], summary_statistics["soundscape_count"], summary_statistics["species_count"], summary_statistics["instance_duration"], summary_statistics["total_duration"] = self.statisticise_time_annotated_soundscape_validation()
        if audio_type == "time_annotated_soundscape" and partition == "test":
            summary_statistics["instance_count"], summary_statistics["soundscape_count"], summary_statistics["instance_duration"], summary_statistics["total_duration"] = self.statisticise_time_annotated_soundscape_test()
        self.update_statistics(summary_statistics, "summary", modality="audio", audio_type=audio_type, partition=partition)
        self.update_statistics(distribution_statistics, "distribution", modality="audio", audio_type=audio_type, partition=partition)

    def statisticise_monophone_train(self):
        frequency_dictionary = {}
        data_path = f"{self.downloaded_files_path}/monophone_train/data/"
        instance_count = 0
        for class_folder in os.listdir(data_path):
            class_path = f"{data_path}/{class_folder}"
            class_instance_count = len(os.listdir(class_path))
            frequency_dictionary = update_frequency_dictionary_(class_folder, class_instance_count, frequency_dictionary)
            instance_count += class_instance_count
        class_count = len(frequency_dictionary)
        return instance_count, class_count, frequency_dictionary

    def statisticise_time_annotated_soundscape_validation(self):
        data_path = f"{self.downloaded_files_path}/soundscape_validation/data/"
        metadata_path = f"{self.downloaded_files_path}/soundscape_validation/metadata/"
        soundscape_count = 0
        instance_duration = "00:00:05"
        total_duration = 0
        for soundscape_file in os.listdir(data_path):
            soundscape_file_path = f"{data_path}{soundscape_file}"
            soundscape_duration = audiofile.duration(soundscape_file_path)
            total_duration += soundscape_duration
            soundscape_count += 1
        instance_count = int(total_duration / 5.0)
        total_minutes, total_seconds = divmod(total_duration, 60)
        total_hours, total_minutes = divmod(total_minutes, 60)
        total_duration = f"{str(int(total_hours)).zfill(2)}:{str(int(total_minutes)).zfill(2)}:{str(int(total_seconds)).zfill(2)}"
        class_name_set = set()
        for soundscape_file in os.listdir(metadata_path):
            soundscape_file_path = f"{metadata_path}{soundscape_file}"
            soundscape_data = read_data_from_file_(soundscape_file_path)
            if "ClassIds" in soundscape_data:
                for soundscape_class_data in soundscape_data["ClassIds"]:
                    class_name = soundscape_class_data["ClassId"]
                    class_name_set.add(class_name)
        class_count = len(class_name_set)
        return instance_count, soundscape_count, class_count, instance_duration, total_duration

    def statisticise_time_annotated_soundscape_test(self):
        data_path = f"{self.downloaded_files_path}/soundscape_test/data/"
        soundscape_count = 0
        instance_duration = "00:00:05"
        total_duration = 0
        for soundscape_file in os.listdir(data_path):
            soundscape_file_path = f"{data_path}{soundscape_file}"
            soundscape_duration = audiofile.duration(soundscape_file_path)
            total_duration += soundscape_duration
            soundscape_count += 1
        instance_count = int(total_duration / 5.0)
        total_minutes, total_seconds = divmod(total_duration, 60)
        total_hours, total_minutes = divmod(total_minutes, 60)
        total_duration = f"{str(int(total_hours)).zfill(2)}:{str(int(total_minutes)).zfill(2)}:{str(int(total_seconds)).zfill(2)}"
        return instance_count, soundscape_count, instance_duration, total_duration

