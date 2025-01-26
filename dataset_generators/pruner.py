from dataset_generators.parameters import Parameters
from taxonomy_generators.sensitive_species import SensitiveSpecies
import copy
from utility import *


class Pruner:

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self, constructor, old_version, new_version, parameters):
        self.constructor = constructor
        self.old_version = old_version
        self.new_version = new_version
        self.parameters = parameters
        self.master_database_structure = self.constructor.get_master_database_structure(old_version)

    # ************************************************ MAJOR METHODS ***************************************************

    def run(self):
        self.create_base()
        if self.parameters["only_common_species"]:
            species_sets = [set(self.constructor.read_species_list(self.constructor.get_dataset_handle(self.new_version, m, p)))
                            for m in self.master_database_structure for p in self.master_database_structure[m]]
            common_species_set = self.determine_common_species(species_sets)
            self.by_species(common_species_set)
        if self.parameters["remove_sensitive_species"]:
            self.by_sensitive_species()
        if self.parameters["species"]:
            self.by_species(set(self.parameters["species"]))
        self.by_observer()
        self.by_observation()
        self.by_joint_media()
        self.by_duration()
        self.by_location_uncertainty()
        self.by_observation_count()
        self.by_observer_count()
        self.by_detail()

    def create_base(self):
        self.constructor.save_master_database_structure(self.new_version, self.master_database_structure)
        i = 0
        for modality in self.master_database_structure:
            for partition in self.master_database_structure[modality]:
                old_dataset = self.constructor.get_dataset_handle(self.old_version, modality, partition)
                new_dataset = self.constructor.get_dataset_handle(self.new_version, modality, partition)
                species_list = self.constructor.read_species_list(old_dataset)
                for species in species_list:
                    observation_list = self.constructor.read_observation_list(old_dataset, species)
                    self.constructor.update_observation_list(new_dataset, species, observation_list)
                    if i % 1000 == 0:
                        print(f"Base: {i}")
                    i += 1

    def by_species(self, valid_species_set):
        for modality in self.master_database_structure:
            for partition in self.master_database_structure[modality]:
                i = 0
                dataset = self.constructor.get_dataset_handle(self.new_version, modality, partition)
                species_list = self.constructor.read_species_list(dataset)
                for species in species_list:
                    if species not in valid_species_set:
                        self.constructor.delete_species(dataset, species)
                    if i % 1000 == 0:
                        print(f"By Species: {i}")
                    i += 1

    @staticmethod
    def determine_common_species(species_sets):
        common_species_set = set.intersection(*species_sets)
        return common_species_set

    def by_sensitive_species(self):
        sensitive_species = SensitiveSpecies()
        sensitive_species_set = sensitive_species.get()
        valid_species_sets = []
        for modality in self.master_database_structure:
            for partition in self.master_database_structure[modality]:
                dataset = self.constructor.get_dataset_handle(self.new_version, modality, partition)
                species_list = self.constructor.read_species_list(dataset)
                i = 0
                valid_species_set = set()
                for species in species_list:
                    if species not in sensitive_species_set:
                        valid_species_set.add(species)
                    if i % 1000 == 0:
                        print(f"By Sensitive Species: {i}")
                    i += 1
                valid_species_sets.append(valid_species_set)
        if valid_species_sets:
            valid_common_species_set = self.determine_common_species(valid_species_sets)
            self.by_species(valid_common_species_set)

    def by_joint_media(self):
        only_joint_media = self.parameters["only_joint_media"]
        for modality in self.master_database_structure:
            for partition in self.master_database_structure[modality]:
                if only_joint_media[modality][partition]:
                    i = 0
                    dataset = self.constructor.get_dataset_handle(self.new_version, modality, partition)
                    source_metadata = self.constructor.get_source_metadata_handle(self.new_version, modality, partition)
                    species_list = self.constructor.read_species_list(dataset)
                    for species in species_list:
                        metadata_list = self.constructor.read_metadata_list(dataset, source_metadata, species)
                        pruned_metadata_list = []
                        for metadata in metadata_list:
                            media_types = metadata["media_types"]
                            if "Image" in media_types and "Audio" in media_types:
                                pruned_metadata_list.append(metadata)
                        pruned_observation_list = [metadata["_id"] for metadata in pruned_metadata_list]
                        self.constructor.update_observation_list(dataset, species, pruned_observation_list)
                        if i % 1000 == 0:
                            print(f"By Joint Media: {i}")
                        i += 1

    def by_observation(self):
        for modality in self.master_database_structure:
            for partition in self.master_database_structure[modality]:
                observation_list = self.parameters["observations_for_removal"][modality][partition]
                if observation_list:
                    i = 0
                    observation_set = set(observation_list)
                    dataset = self.constructor.get_dataset_handle(self.new_version, modality, partition)
                    source_metadata = self.constructor.get_source_metadata_handle(self.new_version, modality, partition)
                    species_list = self.constructor.read_species_list(dataset)
                    for species in species_list:
                        metadata_list = self.constructor.read_metadata_list(dataset, source_metadata, species)
                        pruned_metadata_list = []
                        for metadata in metadata_list:
                            if metadata["gbif_id"] not in observation_set:
                                pruned_metadata_list.append(metadata)
                        pruned_observation_list = [metadata["_id"] for metadata in pruned_metadata_list]
                        self.constructor.update_observation_list(dataset, species, pruned_observation_list)
                        if i % 1000 == 0:
                            print(f"By Observation: {i}")
                        i += 1

    def by_observer(self):
        for modality in self.master_database_structure:
            for partition in self.master_database_structure[modality]:
                observer_dictionary = self.parameters["observers_for_removal"][modality][partition]
                if observer_dictionary:
                    i = 0
                    dataset = self.constructor.get_dataset_handle(self.new_version, modality, partition)
                    source_metadata = self.constructor.get_source_metadata_handle(self.new_version, modality, partition)
                    species_list = self.constructor.read_species_list(dataset)
                    for species in species_list:
                        if species in observer_dictionary:
                            observer_set = set(observer_dictionary[species])
                            metadata_list = self.constructor.read_metadata_list(dataset, source_metadata, species)
                            pruned_metadata_list = []
                            for metadata in metadata_list:
                                if metadata["observer"] not in observer_set:
                                    pruned_metadata_list.append(metadata)
                            pruned_observation_list = [metadata["_id"] for metadata in pruned_metadata_list]
                            self.constructor.update_observation_list(dataset, species, pruned_observation_list)
                        if i % 1000 == 0:
                            print(f"By Observer: {i}")
                        i += 1

    def by_duration(self):
        duration_thresholds = self.parameters["duration_threshold"]
        for modality in self.master_database_structure:
            for partition in self.master_database_structure[modality]:
                if duration_thresholds[modality][partition] != Parameters.parameter_type_defaults["duration_threshold"]:
                    i = 0
                    dataset = self.constructor.get_dataset_handle(self.new_version, modality, partition)
                    source_metadata = self.constructor.get_source_metadata_handle(self.new_version, modality, partition)
                    species_list = self.constructor.read_species_list(dataset)
                    for species in species_list:
                        metadata_list = self.constructor.read_metadata_list(dataset, source_metadata, species)
                        pruned_metadata_list = []
                        for metadata in metadata_list:
                            duration_threshold_seconds = int(duration_thresholds[modality][partition].split(":")[0]) * 60 + int(duration_thresholds[modality][partition].split(":")[1])
                            metadata_duration_seconds = int(metadata["duration"].split(":")[0]) * 60 + int(metadata["duration"].split(":")[1])
                            if metadata_duration_seconds <= duration_threshold_seconds:
                                pruned_metadata_list.append(metadata)
                        pruned_observation_list = [metadata["_id"] for metadata in pruned_metadata_list]
                        self.constructor.update_observation_list(dataset, species, pruned_observation_list)
                        if i % 1000 == 0:
                            print(f"By Duration: {i}")
                        i += 1

    def by_location_uncertainty(self):
        location_uncertainty_thresholds = self.parameters["location_uncertainty_threshold"]
        for modality in self.master_database_structure:
            for partition in self.master_database_structure[modality]:
                if location_uncertainty_thresholds[modality][partition] != Parameters.parameter_type_defaults["location_uncertainty_threshold"]:
                    i = 0
                    dataset = self.constructor.get_dataset_handle(self.new_version, modality, partition)
                    source_metadata = self.constructor.get_source_metadata_handle(self.new_version, modality, partition)
                    species_list = self.constructor.read_species_list(dataset)
                    for species in species_list:
                        metadata_list = self.constructor.read_metadata_list(dataset, source_metadata, species)
                        pruned_metadata_list = []
                        for metadata in metadata_list:
                            location_uncertainty_threshold_metres = location_uncertainty_thresholds[modality][partition] * 1000
                            location_uncertainty_metres = metadata["location_uncertainty"]
                            if location_uncertainty_metres <= location_uncertainty_threshold_metres:
                                pruned_metadata_list.append(metadata)
                        pruned_observation_list = [metadata["_id"] for metadata in pruned_metadata_list]
                        self.constructor.update_observation_list(dataset, species, pruned_observation_list)
                        if i % 1000 == 0:
                            print(f"By Location Uncertainty: {i}")
                        i += 1

    def by_observation_count(self):
        valid_species_sets = []
        observation_minimum_thresholds = self.parameters["observation_minimum_threshold"]
        observation_maximum_thresholds = self.parameters["observation_maximum_threshold"]
        for modality in self.master_database_structure:
            for partition in self.master_database_structure[modality]:
                dataset = self.constructor.get_dataset_handle(self.new_version, modality, partition)
                source_metadata = self.constructor.get_source_metadata_handle(self.new_version, modality, partition)
                species_list = self.constructor.read_species_list(dataset)
                if observation_minimum_thresholds[modality][partition] != Parameters.parameter_type_defaults["observation_minimum_threshold"] or observation_maximum_thresholds[modality][partition] != Parameters.parameter_type_defaults["observation_maximum_threshold"]:
                    i = 0
                    valid_species_set = set()
                    for species in species_list:
                        metadata_list = self.constructor.read_metadata_list(dataset, source_metadata, species)
                        observation_count = len(metadata_list)
                        if observation_minimum_thresholds[modality][partition] <= observation_count <= observation_maximum_thresholds[modality][partition]:
                            valid_species_set.add(species)
                        if i % 1000 == 0:
                            print(f"By Observation Count: {i}")
                        i += 1
                else:
                    valid_species_set = set(species_list)
                valid_species_sets.append(valid_species_set)
        if valid_species_sets:
            valid_common_species_set = self.determine_common_species(valid_species_sets)
            self.by_species(valid_common_species_set)

    def by_observer_count(self):
        valid_species_sets = []
        observer_minimum_thresholds = self.parameters["observer_minimum_threshold"]
        observer_maximum_thresholds = self.parameters["observer_maximum_threshold"]
        for modality in self.master_database_structure:
            for partition in self.master_database_structure[modality]:
                dataset = self.constructor.get_dataset_handle(self.new_version, modality, partition)
                source_metadata = self.constructor.get_source_metadata_handle(self.new_version, modality, partition)
                species_list = self.constructor.read_species_list(dataset)
                if observer_minimum_thresholds[modality][partition] != Parameters.parameter_type_defaults["observer_minimum_threshold"] or observer_maximum_thresholds[modality][partition] != Parameters.parameter_type_defaults["observer_maximum_threshold"]:
                    i = 0
                    valid_species_set = set()
                    for species in species_list:
                        metadata_list = self.constructor.read_metadata_list(dataset, source_metadata, species)
                        observers = set()
                        for metadata in metadata_list:
                            if "observer" in metadata:
                                observers.add(metadata["observer"])
                        observer_count = len(observers)
                        if observer_minimum_thresholds[modality][partition] <= observer_count <= observer_maximum_thresholds[modality][partition]:
                            valid_species_set.add(species)
                        if i % 1000 == 0:
                            print(f"By Observer Count: {i}")
                        i += 1
                else:
                    valid_species_set = set(species_list)
                valid_species_sets.append(valid_species_set)
        if valid_species_sets:
            valid_common_species_set = self.determine_common_species(valid_species_sets)
            self.by_species(valid_common_species_set)

    def determine_species_dictionary(self):
        i = 0
        species_dictionary = {}
        for modality in self.master_database_structure:
            for partition in self.master_database_structure[modality]:
                dataset = self.constructor.get_dataset_handle(self.new_version, modality, partition)
                species_list = self.constructor.read_species_list(dataset)
                for species in species_list:
                    if species not in species_dictionary:
                        species_dictionary[species] = i
                        i += 1
        return species_dictionary

    def determine_detail_count_dictionary(self, species_dictionary):
        detail_count_dictionary = {}
        if self.parameters["require_sex_detail"]:
            detail_count_dictionary["male"] = [0.0] * len(species_dictionary)
            detail_count_dictionary["female"] = [0.0] * len(species_dictionary)
        if self.parameters["require_age_detail"]:
            detail_count_dictionary["adult"] = [0.0] * len(species_dictionary)
            detail_count_dictionary["young"] = [0.0] * len(species_dictionary)
        if self.parameters["require_subspecies_detail"]:
            detail_count_dictionary["subspecies"] = [0.0] * len(species_dictionary)
            detail_count_dictionary["uncertain_subspecies"] = [0.0] * len(species_dictionary)
        return detail_count_dictionary

    def determine_detail_presence_dictionary(self):
        detail_presence_dictionary = {}
        if self.parameters["require_sex_detail"]:
            detail_presence_dictionary["male"] = False
            detail_presence_dictionary["female"] = False
        if self.parameters["require_age_detail"]:
            detail_presence_dictionary["adult"] = False
            detail_presence_dictionary["young"] = False
        if self.parameters["require_subspecies_detail"]:
            detail_presence_dictionary["subspecies"] = False
            detail_presence_dictionary["uncertain_subspecies"] = False
        return detail_presence_dictionary

    def by_detail(self):
        if self.parameters["maximum_species_count"] != Parameters.parameter_type_defaults["maximum_species_count"]:
            sorted_species_list = self.sort_by_detail()
            valid_species_set = set(sorted_species_list[:self.parameters["maximum_species_count"]])
            self.by_species(valid_species_set)

    def sort_by_detail(self):
        detail_count_dictionary_list = []
        species_dictionary = self.determine_species_dictionary()
        detail_count_dictionary_template = self.determine_detail_count_dictionary(species_dictionary)
        attribute_name_list = ["sex", "age", "general_vocalisation", "specific_vocalisation", "subspecies", "uncertain_subspecies", "group", "uncertain_group"]
        partition_weight_list = []
        if self.parameters["maximum_species_count"] != Parameters.parameter_type_defaults["maximum_species_count"]:
            for modality in self.master_database_structure:
                for partition in self.master_database_structure[modality]:
                    detail_count_dictionary = copy.deepcopy(detail_count_dictionary_template)
                    observation_frequency_list = [0] * len(species_dictionary)
                    dataset = self.constructor.get_dataset_handle(self.new_version, modality, partition)
                    source_metadata = self.constructor.get_source_metadata_handle(self.new_version, modality, partition)
                    species_list = self.constructor.read_species_list(dataset)
                    i = 0
                    for species in species_list:
                        species_i = species_dictionary[species]
                        compound_attribute_list = self.constructor.read_compound_attribute_list(dataset, source_metadata, species, attribute_name_list)
                        observation_frequency_list[species_i] = len(compound_attribute_list)
                        for compound_attribute in compound_attribute_list:
                            detail_presence_dictionary = self.determine_detail_presence_dictionary()
                            if self.parameters["require_sex_detail"]:
                                for sex in compound_attribute["sex"]:
                                    if sex != "Unknown":
                                        if sex == "Male":
                                            detail_presence_dictionary["male"] = True
                                        if sex == "Female":
                                            detail_presence_dictionary["female"] = True
                            if self.parameters["require_age_detail"]:
                                for age in compound_attribute["age"]:
                                    if age != "Unknown":
                                        if age == "Adult":
                                            detail_presence_dictionary["adult"] = True
                                        else:
                                            detail_presence_dictionary["young"] = True
                            if self.parameters["require_subspecies_detail"]:
                                if compound_attribute["subspecies"] != "" or compound_attribute["group"] != "":
                                    detail_presence_dictionary["subspecies"] = True
                                if ("uncertain_subspecies" in compound_attribute and compound_attribute["uncertain_subspecies"] != "") or ("uncertain_group" in compound_attribute and compound_attribute["uncertain_group"] != ""):
                                    detail_presence_dictionary["uncertain_subspecies"] = True
                            for detail in detail_presence_dictionary:
                                if detail_presence_dictionary[detail]:
                                    detail_count_dictionary[detail][species_i] += 1
                        if i % 100 == 0:
                            print(f"By Detail: {i}")
                        i += 1
                    for detail in detail_count_dictionary:
                        for species_i in range(len(species_dictionary)):
                            detail_count_dictionary[detail][species_i] = (detail_count_dictionary[detail][species_i] / float(observation_frequency_list[species_i])) * 100
                    detail_count_dictionary_list.append(detail_count_dictionary)
                    partition_weight_list.append(self.parameters["partition_weight"][modality][partition])
            print("Sorting By Detail...")
            species_list = [None] * len(species_dictionary)
            for species in species_dictionary:
                species_list[species_dictionary[species]] = species
            tier_count = max([self.parameters["detail_tier"][detail] for detail in self.parameters["detail_tier"]]) + 1
            tiered_weighted_count_list = [[0.0 for _ in range(len(species_dictionary))] for _ in range(tier_count)]
            tiered_penalty_list = [[0 for _ in range(len(species_dictionary))] for _ in range(tier_count)]
            for species_i in range(len(species_list)):
                for detail in self.parameters["detail_tier"]:
                    tier = self.parameters["detail_tier"][detail]
                    i = 0
                    for detail_count_dictionary in detail_count_dictionary_list:
                        if detail_count_dictionary[detail][species_i] == 0:
                            tiered_penalty_list[tier][species_i] += self.parameters["detail_penalty"][detail] * partition_weight_list[i]
                        tiered_weighted_count_list[tier][species_i] += detail_count_dictionary[detail][species_i] * self.parameters["detail_weight"][detail] * partition_weight_list[i]
                        i += 1
            for tier in range(tier_count):
                for species_i in range(len(species_list)):
                    tiered_weighted_count_list[tier][species_i] = tiered_weighted_count_list[tier][species_i] - tiered_weighted_count_list[tier][species_i] * (tiered_penalty_list[tier][species_i] / 100.0)
            sorted_species_list = [list(x) for x in zip(*sorted(zip(*tiered_weighted_count_list, species_list), reverse=True))][-1]
            return sorted_species_list

    # ******************************************************************************************************************
