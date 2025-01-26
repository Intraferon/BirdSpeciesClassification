from dataset_generators.pruner import Pruner
from dataset_generators.partitioner import Partitioner
from dataset_generators.pairer import Pairer
from dataset_generators.constants import *
from pymongo import MongoClient
import shutil
from utility import *


class Constructor:

    partitioner_control_logs_path = f"{RESOURCE_PATH}dataset_generators/partitioner_control_logs/"
    partitioner_experiment_logs_path = f"{RESOURCE_PATH}dataset_generators/partitioner_experiment_logs/"
    partitioner_experiment_saves_path = f"{RESOURCE_PATH}dataset_generators/partitioner_experiment_saves/"
    pairer_control_logs_path = f"{RESOURCE_PATH}dataset_generators/pairer_control_logs/"
    pairer_experiment_logs_path = f"{RESOURCE_PATH}dataset_generators/pairer_experiment_logs/"

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self, dataset_name, database_name):

        self.dataset_name = dataset_name
        self.dataset_path = f"C:/Users/sanam/Documents/Masters/Resources/Datasets/{dataset_name}/"

        self.compiled_path = f"{self.dataset_path}compiled/"

        self.database_name = database_name
        self.partition_experiment_database_name = f"part_{self.database_name}"
        self.pair_experiment_database_name = f"pair_{self.database_name}"

        self.version_info_file_path = f"{self.dataset_path}version_info.json"
        self.latest_version_file_path = f"{self.dataset_path}latest_version.txt"

        self.master_database_structure_file_path = f"{self.dataset_path}master_database_structure.json"
        self.partition_experiment_master_database_structure_file_path = f"{self.dataset_path}partition_experiment_master_database_structure.json"
        self.pair_experiment_master_database_structure_file_path = f"{self.dataset_path}pair_experiment_master_database_structure.json"

        self.parameters_file_path = f"{self.dataset_path}parameters.json"
        self.partition_experiment_configurations_file_path = f"{self.dataset_path}partition_experiment_configurations.json"
        self.pair_experiment_configurations_file_path = f"{self.dataset_path}pair_experiment_configurations.json"

        self.statistics_file_path = f"{self.dataset_path}statistics.json"
        self.partition_experiment_statistics_file_path = f"{self.dataset_path}partition_experiment_statistics.json"
        self.pair_experiment_statistics_file_path = f"{self.dataset_path}pair_experiment_statistics.json"

        self.sorted_species_file_path = f"{self.dataset_path}sorted_species.txt"

    # ************************************************ MAJOR METHODS ***************************************************

    def configure_database(self, master_database_structure, parameters=None, version_info="Source", source_dataset_structure=None):
        print("Configuring Database...")
        version = 0
        create_folder_(self.dataset_path)
        create_folder_(self.compiled_path)
        self.save_latest_version(version)
        self.save_version_info(version, version_info)
        self.save_master_database_structure(version, master_database_structure)
        self.save_parameters(version, parameters)
        if source_dataset_structure is None:
            self.create_database_from_source()
        else:
            self.create_database_from_dataset(source_dataset_structure)
        statistics = self.statisticise_database(version)
        self.save_statistics(version, statistics)

    def create_database_from_source(self):
        print("Creating Database...")
        version = 0
        master_database_structure = self.get_master_database_structure(version)
        for modality in master_database_structure:
            for partition in master_database_structure[modality]:
                source_dataset = self.get_source_dataset_handle(version, modality, partition)
                dataset = self.get_dataset_handle(version, modality, partition)
                for row in source_dataset.find():
                    dataset.insert_one(row)

    def create_database_from_dataset(self, source_dataset_structure):
        print("Creating Database from Dataset...")
        version = 0
        master_database_structure = self.get_master_database_structure(version)
        for modality in master_database_structure:
            for partition in master_database_structure[modality]:
                source_dataset = source_dataset_structure[modality][partition]
                dataset = self.get_dataset_handle(version, modality, partition)
                for row in source_dataset.find():
                    dataset.insert_one(row)

    def update_database(self, parameters, version_info):
        print("Updating Database...")
        old_version = self.get_latest_version()
        new_version = old_version + 1
        self.construct_dataset(old_version, new_version, parameters)
        self.save_latest_version(new_version)
        self.save_version_info(new_version, version_info)
        self.save_parameters(new_version, parameters)
        statistics = self.statisticise_database(new_version)
        self.save_statistics(new_version, statistics)

    def construct_dataset(self, old_version, new_version, parameters):
        pruner = Pruner(self, old_version, new_version, parameters["pruner"])
        pruner.run()

    def sort_dataset(self):
        version = self.get_latest_version()
        parameters = self.get_parameters(version)
        pruner = Pruner(self, version, version, parameters["pruner"])
        sorted_species_list = pruner.sort_by_detail()
        save_data_to_file_(self.sorted_species_file_path, sorted_species_list)

    def create_partition_experiment_control(self, subset, configuration, modality, partition):
        print("Creating Partition Experiment Control...")
        version = self.get_latest_version()
        configuration_parameters = self.get_configuration(configuration, "partition")
        log_path = f"{self.partitioner_control_logs_path}{self.database_name}#{modality}#{configuration}.txt"
        partitioner = Partitioner(self, configuration_parameters["partitioner"])
        partitioner.control(subset, configuration, version, modality, partition, log_path=log_path)

    def conduct_partition_experiment(self, subset, configuration, modality, partition):
        print("Conducting Partition Experiment...")
        version = self.get_latest_version()
        configuration_parameters = self.get_configuration(configuration, "partition")
        base_log_path = f"{self.partitioner_experiment_logs_path}{self.database_name}"
        base_save_path = f"{self.partitioner_experiment_saves_path}{self.database_name}"
        partitioner = Partitioner(self, configuration_parameters["partitioner"])
        partitioner.experiment(subset, configuration, version, modality, partition, base_log_path, base_save_path, create_save=True, load_save=True)

    def create_partition_experiment_context(self, configuration, modality, partition):
        print("Creating Partition Experiment Context...")
        version = self.get_latest_version()
        configuration_parameters = self.get_configuration(configuration, "partition")
        log_path = f"{self.partitioner_control_logs_path}{self.database_name}#{modality}#{configuration}.txt"
        partitioner = Partitioner(self, configuration_parameters["partitioner"])
        partitioner.context(version, modality, partition, log_path=log_path)

    def evaluate_partition_experiment(self, configuration, modality, partition):
        print("Evaluating Partition Experiment...")
        version = self.get_latest_version()
        base_log_path = f"{self.partitioner_experiment_logs_path}{self.database_name}"
        partitioner = Partitioner(self)
        partitioner.evaluate_all(configuration, version, modality, partition, base_log_path)

    def compare_partition_experiment(self, target_configuration, reference_configuration, modality, partition):
        print("Comparing Partition Experiment...")
        version = self.get_latest_version()
        base_log_path = f"{self.partitioner_experiment_logs_path}{self.database_name}"
        partitioner = Partitioner(self)
        partitioner.compare_all(target_configuration, reference_configuration, version, modality, partition, base_log_path)

    def create_pair_experiment_source(self, partition_configuration, partition_solution_type, partition):
        print("Creating Pair Experiment Source...")
        configuration_id = f"{partition_configuration}_source"
        pairer = Pairer(self, None)
        pairer.source(configuration_id, partition_configuration, partition_solution_type, partition)

    def create_pair_experiment_control(self, configuration, image_partition_configuration, audio_partition_configuration, partition_solution_type, partition):
        print("Creating Pair Experiment Control...")
        configuration_id = f"{image_partition_configuration}_{audio_partition_configuration}_{configuration}"
        configuration_parameters = self.get_configuration(configuration, "pair")
        log_path = f"{self.pairer_control_logs_path}{self.database_name}#{partition}#{configuration_id}.txt"
        pairer = Pairer(self, configuration_parameters["pairer"])
        pairer.control(configuration_id, image_partition_configuration, audio_partition_configuration, partition_solution_type, partition, log_path)

    def conduct_pair_experiment(self, configuration, image_partition_configuration, audio_partition_configuration, partition_solution_type, partition):
        print("Conducting Pair Experiment...")
        configuration_id = f"{image_partition_configuration}_{audio_partition_configuration}_{configuration}"
        configuration_parameters = self.get_configuration(configuration, "pair")
        log_path = f"{self.pairer_experiment_logs_path}{self.database_name}#{partition}#{configuration_id}.json"
        pairer = Pairer(self, configuration_parameters["pairer"])
        pairer.experiment(configuration_id, image_partition_configuration, audio_partition_configuration, partition_solution_type, partition, log_path)


    def evaluate_pair_experiment(self, configuration, image_partition_configuration, audio_partition_configuration):
        print("Evaluating Pair Experiment...")
        configuration_id = f"{image_partition_configuration}_{audio_partition_configuration}_{configuration}"
        configuration_parameters = self.get_configuration(configuration, "pair")
        base_log_path = f"{self.pairer_experiment_logs_path}{self.database_name}"
        pairer = Pairer(self, configuration_parameters["pairer"])
        pairer.evaluate_all(configuration_id, base_log_path)

    # ************************************************ READ METHODS ***************************************************

    @staticmethod
    def read_species_list(dataset):
        species_list = [species["_id"] for species in dataset.find()]
        return species_list

    @staticmethod
    def read_observation_list(dataset, species, dataset_type_=None, solution_type_=""):
        if dataset_type_ is None:
            observation_list = dataset.find({"_id": species})[0]["observation_ids"]
        else:
            observation_list = dataset.find({"_id": species})[0][f"{solution_type_}_observation_ids"]
        return observation_list

    def read_observation_count(self, dataset, species, dataset_type_=None, solution_type_=""):
        observation_count = len(self.read_observation_list(dataset, species, dataset_type_, solution_type_))
        return observation_count

    def read_attribute_list(self, dataset, source_metadata, species, attribute_name, dataset_type_=None, solution_type_=""):
        observation_list = self.read_observation_list(dataset, species, dataset_type_=dataset_type_, solution_type_=solution_type_)
        source_attribute_list = [observation[attribute_name] for observation in source_metadata.find({"_id": {"$in": observation_list}})]
        return source_attribute_list

    def read_compound_attribute_list(self, dataset, source_metadata, species, attribute_name_list, dataset_type_=None, solution_type_=""):
        observation_list = self.read_observation_list(dataset, species, dataset_type_=dataset_type_, solution_type_=solution_type_)
        metadata_list = source_metadata.find({"_id": {"$in": observation_list}})
        compound_attribute_list = []
        for metadata in metadata_list:
            compound_attribute = {}
            for attribute_name in attribute_name_list:
                if attribute_name in metadata:
                    compound_attribute[attribute_name] = metadata[attribute_name]
            compound_attribute_list.append(compound_attribute)
        return compound_attribute_list

    def read_joint_compound_attribute_list(self, dataset, source_metadata, species, attribute_name_list, dataset_type_=None, solution_type_=""):
        observation_list = self.read_observation_list(dataset, species, dataset_type_=dataset_type_, solution_type_=solution_type_)
        image_observation_list = [observation[0] for observation in observation_list]
        audio_observation_list = [observation[1] for observation in observation_list]
        image_compound_attribute_list = self.extract_compound_attribute_list(image_observation_list)
        audio_compound_attribute_list = self.extract_compound_attribute_list(audio_observation_list)
        return image_compound_attribute_list, audio_compound_attribute_list

    def extract_compound_attribute_list(self, observation_list):
        observation_order_dictionary = {observation_list[i]: i for i in range(len(observation_list))}
        metadata_list = source_metadata.find({"_id": {"$in": observation_list}})
        unordered_compound_attribute_list = []
        for metadata in metadata_list:
            compound_attribute = {"_id": metadata["_id"]}
            for attribute_name in attribute_name_list:
                if attribute_name in metadata:
                    compound_attribute[attribute_name] = metadata[attribute_name]
            unordered_compound_attribute_list.append(compound_attribute)
        compound_attribute_list = [None] * len(unordered_compound_attribute_list)
        for compound_attribute in unordered_compound_attribute_list:
            i = observation_order_dictionary[compound_attribute["_id"]]
            compound_attribute_list[i] = compound_attribute
        return compound_attribute_list

    @staticmethod
    def read_metadata(source_metadata, observation):
        metadata = source_metadata.find({"_id": observation})[0]
        return metadata

    def read_metadata_list(self, dataset, source_metadata, species, dataset_type_=None, solution_type_=""):
        observation_list = self.read_observation_list(dataset, species, dataset_type_=dataset_type_, solution_type_=solution_type_)
        metadata_list = [_ for _ in source_metadata.find({"_id": {"$in": observation_list}})]
        return metadata_list

    def read_joint_metadata_list(self, dataset, image_source_metadata, audio_source_metadata, species, dataset_type_=None, solution_type_=""):
        observation_list = self.read_observation_list(dataset, species, dataset_type_=dataset_type_, solution_type_=solution_type_)
        image_observation_list = [observation[0] for observation in observation_list]
        audio_observation_list = [observation[1] for observation in observation_list]
        image_metadata_list = self.extract_metadata_list(image_observation_list, image_source_metadata)
        audio_metadata_list = self.extract_metadata_list(audio_observation_list, audio_source_metadata)
        return image_metadata_list, audio_metadata_list

    def extract_metadata_list(self, observation_list, source_metadata):
        metadata_list = []
        for observation in observation_list:
            metadata = self.read_metadata(source_metadata, observation)
            metadata_list.append(metadata)
        return metadata_list

    # ************************************************ UPDATE METHODS ***************************************************

    @staticmethod
    def update_observation_list(dataset, species, observation_list, dataset_type_=None, solution_type_=""):
        if dataset_type_ is None:
            dataset.find_one_and_update({"_id": species}, {"$set": {"observation_ids": observation_list}}, upsert=True)
        else:
            dataset.find_one_and_update({"_id": species}, {"$set": {f"{solution_type_}_observation_ids": observation_list}}, upsert=True)

    # ************************************************ SAVE METHODS ***************************************************

    def save_master_database_structure(self, version_configuration, master_database_structure, dataset_type_=None):
        if dataset_type_ == "partition":
            master_database_structure_file_path = self.partition_experiment_master_database_structure_file_path
        elif dataset_type_ == "pair":
            master_database_structure_file_path = self.pair_experiment_master_database_structure_file_path
        else:
            master_database_structure_file_path = self.master_database_structure_file_path
        master_database_structure_dictionary = read_data_from_file_(master_database_structure_file_path) if os.path.exists(master_database_structure_file_path) else {}
        master_database_structure_dictionary[str(version_configuration)] = master_database_structure
        save_data_to_file_(master_database_structure_file_path, master_database_structure_dictionary)

    def save_version_info(self, version, version_info):
        version_info_dictionary = read_data_from_file_(self.version_info_file_path) if os.path.exists(self.version_info_file_path) else {}
        version_info_dictionary[str(version)] = version_info
        save_data_to_file_(self.version_info_file_path, version_info_dictionary)

    def save_parameters(self, version, parameters):
        parameter_dictionary = read_data_from_file_(self.parameters_file_path) if os.path.exists(self.parameters_file_path) else {}
        parameter_dictionary[str(version)] = parameters
        save_data_to_file_(self.parameters_file_path, parameter_dictionary)

    def save_statistics(self, version_configuration, statistics, dataset_type_=None):

        if dataset_type_ == "partition":
            statistics_file_path = self.partition_experiment_statistics_file_path
        elif dataset_type_ == "pair":
            statistics_file_path = self.pair_experiment_statistics_file_path
        else:
            statistics_file_path = self.statistics_file_path

        statistics_dictionary = read_data_from_file_(statistics_file_path) if os.path.exists(statistics_file_path) else {}
        statistics_dictionary[str(version_configuration)] = statistics
        save_data_to_file_(statistics_file_path, statistics_dictionary)

    def save_configuration(self, configuration, parameters, dataset_type_):
        configurations_file_path = self.partition_experiment_configurations_file_path if dataset_type_ == "partition" else self.pair_experiment_configurations_file_path
        configuration_dictionary = read_data_from_file_(configurations_file_path) if os.path.exists(configurations_file_path) else {}
        configuration_dictionary[configuration] = parameters
        save_data_to_file_(configurations_file_path, configuration_dictionary)

    def save_latest_version(self, version):
        save_data_to_file_(self.latest_version_file_path, [str(version)])

    # ************************************************ GETTER METHODS ***************************************************

    def get_log_path(self, configuration, modality, species, dataset_type_):
        if dataset_type_ == "partition":
            logs_path = self.partitioner_experiment_logs_path
        else:
            logs_path = self.pairer_experiment_logs_path
        log_path = f"{logs_path}{self.database_name}#{modality}#{species.lower().replace(' ', '_')}#{configuration}.json"
        return log_path

    def get_master_database_structure(self, version_configuration, dataset_type_=None):

        if dataset_type_ == "partition":
            master_database_structure_file_path = self.partition_experiment_master_database_structure_file_path
        elif dataset_type_ == "pair":
            master_database_structure_file_path = self.pair_experiment_master_database_structure_file_path
        else:
            master_database_structure_file_path = self.master_database_structure_file_path

        master_database_structure = {}
        if os.path.exists(master_database_structure_file_path):
            master_database_structure_dictionary = read_data_from_file_(master_database_structure_file_path)
            if str(version_configuration) in master_database_structure_dictionary:
                master_database_structure = master_database_structure_dictionary[str(version_configuration)]

        return master_database_structure

    def get_statistics(self, version_configuration, dataset_type_=None):

        if dataset_type_ == "partition":
            statistics_file_path = self.partition_experiment_statistics_file_path
        elif dataset_type_ == "pair":
            statistics_file_path = self.pair_experiment_statistics_file_path
        else:
            statistics_file_path = self.statistics_file_path

        statistics = {}
        if os.path.exists(statistics_file_path):
            statistics_dictionary = read_data_from_file_(statistics_file_path)
            if str(version_configuration) in statistics_dictionary:
                statistics = statistics_dictionary[str(version_configuration)]

        return statistics

    def get_parameters(self, version):
        parameters = read_data_from_file_(self.parameters_file_path)[str(version)]
        return parameters

    def get_sorted_species(self):
        sorted_species_list = read_data_from_file_(self.sorted_species_file_path)
        return sorted_species_list

    def get_configuration(self, configuration, dataset_type_):
        configurations_file_path = self.partition_experiment_configurations_file_path if dataset_type_ == "partition" else self.pair_experiment_configurations_file_path
        configuration = read_data_from_file_(configurations_file_path)[configuration]
        return configuration

    def get_configuration_list(self, dataset_type_):
        configurations_file_path = self.partition_experiment_configurations_file_path if dataset_type_ == "partition" else self.pair_experiment_configurations_file_path
        configuration_dictionary = read_data_from_file_(configurations_file_path)
        configuration_list = list(configuration_dictionary.keys())
        return configuration_list

    def get_latest_version(self):
        latest_version = int(read_data_from_file_(self.latest_version_file_path)[0])
        return latest_version

    def get_dataset_handle(self, version_configuration, modality, partition, dataset_type_=None):
        if dataset_type_ == "partition":
            database_name = self.partition_experiment_database_name
        elif dataset_type_ == "pair":
            database_name = self.pair_experiment_database_name
        else:
            database_name = self.database_name
        client = MongoClient()
        database = client[database_name]
        dataset_name = f"{str(version_configuration)}_{modality}_{partition}"
        dataset = database[dataset_name]
        return dataset

    def get_source_database_handle(self, version_configuration, modality, partition, dataset_type_=None):
        client = MongoClient()
        master_database_structure = self.get_master_database_structure(version_configuration, dataset_type_=dataset_type_)
        if dataset_type_ == "pair":
            source_database = client[master_database_structure["image-audio"][partition][0 if modality == "image" else 1]]
        else:
            source_database = client[master_database_structure[modality][partition]]
        return source_database

    def get_source_dataset_handle(self, version_configuration, modality, partition, dataset_type_=None):
        source_database = self.get_source_database_handle(version_configuration, modality, partition, dataset_type_=dataset_type_)
        source_dataset = source_database.species
        return source_dataset

    def get_source_metadata_handle(self, version_configuration, modality, partition, dataset_type_=None):
        source_database = self.get_source_database_handle(version_configuration, modality, partition, dataset_type_=dataset_type_)
        source_metadata = source_database.observation
        return source_metadata

    # ************************************************ DELETE METHODS ***************************************************

    def delete(self, dataset_type_=None):
        if dataset_type_ == "partition":
            database_name = self.partition_experiment_database_name
        elif dataset_type_ == "pair":
            database_name = self.pair_experiment_database_name
        else:
            database_name = self.database_name
            if os.path.exists(self.dataset_path): shutil.rmtree(self.dataset_path)
        client = MongoClient()
        client.drop_database(database_name)

    @staticmethod
    def delete_dataset(dataset):
        dataset.drop()

    @staticmethod
    def delete_species(dataset, species):
        dataset.delete_one({"_id": species})




