from utility import *


class Dataset:

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.downloaded_files_path = f"{self.dataset_path}/Downloaded/"
        self.statistics_files_path = f"{self.dataset_path}/Statistics/"

    @staticmethod
    def create_statistics_template(statistics_type, modality=None, audio_type=None, partition=None):
        statistics_template = {}
        if modality == "image":
            if statistics_type == "summary":
                statistics_template = {"instance_count": 0, "alternate_instance_count": -1, "species_count": 0, "alternate_class_count": 0}
            if statistics_type == "distribution":
                statistics_template = {}
        if modality == "audio":
            if statistics_type == "summary":
                if audio_type == "monophone" and partition == "train":
                    statistics_template = {"instance_count": 0, "species_count": 0}
                if audio_type == "monophone" and partition == "test":
                    statistics_template = {"instance_count": 0}
                if audio_type == "record_annotated_soundscape" and partition == "test":
                    statistics_template = {"instance_count": 0, "soundscape_count": 0, "species_count": -1, "instance_duration": "00:00:00", "total_duration": "00:00:00"}
                if audio_type == "time_annotated_soundscape" and partition == "validation":
                    statistics_template = {"instance_count": 0, "soundscape_count": 0, "species_count": 0, "instance_duration": "00:00:00", "total_duration": "00:00:00"}
                if audio_type == "time_annotated_soundscape" and partition == "test":
                    statistics_template = {"instance_count": 0, "soundscape_count": 0, "species_count": -1, "instance_duration": "00:00:00", "total_duration": "00:00:00"}
            if statistics_type == "distribution":
                statistics_template = {}
        return statistics_template

    def update_statistics(self, statistics, statistics_type, modality=None, rendition=None, audio_type=None, partition=None):
        statistics_template = self.create_statistics_template(statistics_type, modality=modality, audio_type=audio_type, partition=partition)
        if statistics != statistics_template:
            statistics_file_code = "-".join([x for x in [rendition, audio_type, partition] if x is not None])
            statistics_file_path = f"{self.statistics_files_path}/{statistics_type} ({statistics_file_code}).json"
            save_data_to_file_(statistics_file_path, statistics)

    def get_statistics(self, statistics_type, modality=None, rendition=None, audio_type=None, partition=None):
        statistics_file_code = "-".join([x for x in [rendition, audio_type, partition] if x is not None])
        statistics_file_path = f"{self.statistics_files_path}/{statistics_type} ({statistics_file_code}).json"
        if os.path.exists(statistics_file_path):
            statistics = read_data_from_file_(statistics_file_path)
        else:
            statistics = self.create_statistics_template(statistics_type, modality=modality, audio_type=audio_type, partition=partition)
        return statistics

    def statistics_exists(self, statistics_type, rendition=None, audio_type=None, partition=None):
        statistics_file_code = "-".join([x for x in [rendition, audio_type, partition] if x is not None])
        statistics_file_path = f"{self.statistics_files_path}/{statistics_type} ({statistics_file_code}).json"
        if os.path.exists(statistics_file_path):
            exists = True
        else:
            exists = False
        return exists


