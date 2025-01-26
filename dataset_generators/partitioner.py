from dataset_generators.parameters import Parameters
from dataset_generators.genetic_algorithm import GeneticAlgorithm
from utility import *
import pprint
import random
import copy
import geopy.distance
from datetime import datetime
import time
import numpy as np
import shutil
from subset_generators.subset import Subset
from dataset_generators.visualisation import *


class Partitioner:

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self, constructor, parameters=None):

        self.constructor = constructor
        self.parameters = parameters

        self.focused_parameters = None

        self.attribute_pool_list = None
        self.extended_attribute_pool_list = None
        self.compound_attribute_list = None
        self.compound_attribute_dictionary = None
        self.compound_attribute_weight_list = None
        self.compound_attribute_sparsity_list = None
        self.compound_attribute_class_list = None
        self.compound_attribute_class_dictionary = None
        self.compound_attribute_class_reference = None
        self.compound_attribute_class_count_table = None
        self.compound_attribute_class_sparsity_list = None
        self.m_max = None
        self.m_ideal = None
        self.a_max = None
        self.a_ideal = None
        self.w = None

        self.observer_list = None
        self.observer_dictionary = None
        self.observer_count_table = None
        self.observer_frequency_table_template = None

    # ************************************************ MAJOR METHODS ***************************************************

    def control(self, subset, configuration, version, modality, partition, log_path):

        if os.path.exists(log_path):
            logs = set(read_data_from_file_(log_path))
        else:
            logs = set()

        species_limit = 1000

        old_master_database_structure = self.constructor.get_master_database_structure(version)
        new_master_database_structure = self.constructor.get_master_database_structure(configuration, dataset_type_="partition")
        if modality not in new_master_database_structure: new_master_database_structure[modality] = {}

        new_dataset_dictionary = {}
        subpartitions = partition.split("-")
        for subpartition in subpartitions:
            new_dataset_dictionary[subpartition] = self.constructor.get_dataset_handle(configuration, modality, subpartition, dataset_type_="partition")
            new_master_database_structure[modality][subpartition] = old_master_database_structure[modality][partition]

        self.constructor.save_master_database_structure(configuration, new_master_database_structure, dataset_type_="partition")

        self.focused_parameters = self.focus_parameters(modality, partition)

        old_dataset = self.constructor.get_dataset_handle(version, modality, partition)
        source_metadata = self.constructor.get_source_metadata_handle(version, modality, partition)
        species_list = self.constructor.read_species_list(old_dataset)

        downloaded_list = set(["_".join(data_id.split("/")[1].split("_")[:-1]) for data_id in read_data_from_file_(f"{subset.subset_path}/downloaded.txt")])

        i = 0

        for species in species_list:

            print(f"species: {species} ({i})")

            if species not in logs:

                source_metadata_list = self.constructor.read_metadata_list(old_dataset, source_metadata, species)

                metadata_list = [metadata for metadata in source_metadata_list if metadata["_id"] in downloaded_list]

                metadata_count = len(metadata_list) if len(metadata_list) <= self.focused_parameters["maximum_observation_count"] else len(source_metadata_list)
                subpartition_dimensions = self.determine_partition_counts(metadata_count, species)

                id_metadata_list = [metadata["_id"] for metadata in metadata_list]
                random.shuffle(id_metadata_list)

                subpartition_offsets = np.append(np.array([0]), np.cumsum(subpartition_dimensions)[:-1])

                for j in range(len(subpartitions)):
                    subpartition = subpartitions[j]
                    subpartition_start = subpartition_offsets[j]
                    subpartition_end = subpartition_start + subpartition_dimensions[j]
                    id_metadata_list_subpartition = id_metadata_list[subpartition_start:subpartition_end]
                    self.constructor.update_observation_list(new_dataset_dictionary[subpartition], species, id_metadata_list_subpartition, dataset_type_="partition", solution_type_="control")

                logs.add(species)

                if i > species_limit:
                    break

                i += 1

        save_data_to_file_(log_path, logs)

    def experiment(self, subset, configuration, version, modality, partition, base_log_path, base_save_path, create_save=True, load_save=True):

        species_limit = 1000

        old_master_database_structure = self.constructor.get_master_database_structure(version)
        new_master_database_structure = self.constructor.get_master_database_structure(configuration, dataset_type_="partition")
        if modality not in new_master_database_structure: new_master_database_structure[modality] = {}

        new_dataset_dictionary = {}
        subpartitions = partition.split("-")
        for subpartition in subpartitions:
            new_dataset_dictionary[subpartition] = self.constructor.get_dataset_handle(configuration, modality, subpartition, dataset_type_="partition")
            new_master_database_structure[modality][subpartition] = old_master_database_structure[modality][partition]

        self.constructor.save_master_database_structure(configuration, new_master_database_structure, dataset_type_="partition")

        self.focused_parameters = self.focus_parameters(modality, partition)
        self.attribute_pool_list, self.extended_attribute_pool_list = self.determine_attribute_pool_list()
        self.compound_attribute_list, self.compound_attribute_dictionary, self.compound_attribute_weight_list, self.compound_attribute_sparsity_list = self.determine_compound_attribute_structures()

        old_dataset = self.constructor.get_dataset_handle(version, modality, partition)
        source_metadata = self.constructor.get_source_metadata_handle(version, modality, partition)
        species_list = self.constructor.read_species_list(old_dataset)

        downloaded_list = set(["_".join(data_id.split("/")[1].split("_")[:-1]) for data_id in read_data_from_file_(f"{subset.subset_path}/downloaded.txt")])

        i = 0

        for species in species_list:

            print(f"species: {species} ({i})")

            log_path = f"{base_log_path}#{modality}#{species.lower().replace(' ', '_')}#{configuration}.json"
            save_path = f"{base_save_path}#{modality}#{species.lower().replace(' ', '_')}#{configuration}.npy"

            has_converged = self.read_status(log_path, load_save)

            if not has_converged:

                source_metadata_list = self.constructor.read_metadata_list(old_dataset, source_metadata, species)

                metadata_list = [metadata for metadata in source_metadata_list if metadata["_id"] in downloaded_list]

                metadata_count = len(metadata_list) if len(metadata_list) <= self.focused_parameters["maximum_observation_count"] else len(source_metadata_list)

                print(f"size: {metadata_count}")

                if metadata_count < 3000000:

                    best_solution, absolute_best_solution, id_metadata_list, subpartition_dimensions = self.determine_metadata_partition_list(metadata_list, metadata_count, species, subpartitions, log_path, save_path, create_save, load_save)
                    subpartition_offsets = np.append(np.array([0]), np.cumsum(subpartition_dimensions)[:-1])

                    for j in range(len(subpartitions)):

                        subpartition = subpartitions[j]
                        subpartition_start = subpartition_offsets[j]
                        subpartition_end = subpartition_start + subpartition_dimensions[j]

                        if best_solution is not None:
                            best_solution_subpartition = best_solution[subpartition_start:subpartition_end]
                            best_observation_ids_subpartition = [id_metadata_list[best_solution_subpartition[k]] for k in range(best_solution_subpartition.shape[0])]
                            self.constructor.update_observation_list(new_dataset_dictionary[subpartition], species, best_observation_ids_subpartition, dataset_type_="partition", solution_type_="best")

                        if absolute_best_solution is not None:
                            absolute_best_solution_subpartition = absolute_best_solution[subpartition_start:subpartition_end]
                            absolute_best_observation_ids_subpartition = [id_metadata_list[absolute_best_solution_subpartition[k]] for k in range(absolute_best_solution_subpartition.shape[0])]
                            self.constructor.update_observation_list(new_dataset_dictionary[subpartition], species, absolute_best_observation_ids_subpartition, dataset_type_="partition", solution_type_="absolute_best")

                    if i > species_limit:
                        break

                    i += 1


    def context(self, version, modality, partition, log_path):

        configuration = "only_context"

        if os.path.exists(log_path):
            logs = set(read_data_from_file_(log_path))
        else:
            logs = set()

        species_limit = 1000

        old_master_database_structure = self.constructor.get_master_database_structure(version)
        new_master_database_structure = self.constructor.get_master_database_structure(configuration, dataset_type_="partition")
        if modality not in new_master_database_structure: new_master_database_structure[modality] = {}

        self.focused_parameters = self.focus_parameters(modality, partition)

        new_dataset_dictionary = {}
        subpartitions = partition.split("-")
        for subpartition in subpartitions:
            new_dataset_dictionary[subpartition] = self.constructor.get_dataset_handle(configuration, modality, subpartition, dataset_type_="partition")
            new_master_database_structure[modality][subpartition] = old_master_database_structure[modality][partition]

        self.constructor.save_master_database_structure(configuration, new_master_database_structure, dataset_type_="partition")

        old_dataset = self.constructor.get_dataset_handle(version, modality, partition)
        species_list = self.constructor.read_species_list(old_dataset)

        i = 0

        for species in species_list:

            print(f"species: {species} ({i})")

            if species not in logs:

                observation_list = self.constructor.read_observation_list(old_dataset, species)
                observation_count = len(observation_list)

                random.shuffle(observation_list)

                subpartition_dimensions = self.determine_partition_counts(observation_count, species)
                subpartition_offsets = np.append(np.array([0]), np.cumsum(subpartition_dimensions)[:-1])

                for j in range(len(subpartitions)):
                    subpartition = subpartitions[j]
                    subpartition_start = subpartition_offsets[j]
                    subpartition_end = subpartition_start + subpartition_dimensions[j]
                    observation_list_subpartition = observation_list[subpartition_start:subpartition_end]
                    self.constructor.update_observation_list(new_dataset_dictionary[subpartition], species, observation_list_subpartition, dataset_type_="partition", solution_type_="context")

                logs.add(species)

                if i > species_limit:
                    break

                i += 1

        save_data_to_file_(log_path, logs)


    def evaluate_all(self, configuration, version, modality, partition, base_log_path):

        dataset = self.constructor.get_dataset_handle(version, modality, partition)
        species_list = self.constructor.read_species_list(dataset)

        i = 0

        MBFI = 0

        BFI_list = []
        observation_count_list = []

        for species in species_list:
            log_path = f"{base_log_path}#{modality}#{species.lower().replace(' ', '_')}#{configuration}.json"
            best_fitness_list = read_data_from_file_(log_path)["best_fitness"]
            initial_best_f = best_fitness_list[0]
            final_best_f = best_fitness_list[-1]
            BFI = (final_best_f - initial_best_f)
            MBFI += BFI
            observation_count = self.constructor.read_observation_count(dataset, species)
            BFI_list.append(BFI)
            observation_count_list.append(observation_count)

        MBFI /= len(species_list)

        print(f"MBFI: {MBFI}")

        best_fitness_improvement_distribution_graph(["iNaturalist"], [BFI_list], [observation_count_list])


    def compare_all(self, target_configuration, reference_configuration, version, modality, partition, base_log_path):

        dataset = self.constructor.get_dataset_handle(version, modality, partition)
        species_list = self.constructor.read_species_list(dataset)

        i = 0

        MBFD = 0

        for species in species_list:
            target_log_path = f"{base_log_path}#{modality}#{species.lower().replace(' ', '_')}#{target_configuration}.json"
            target_best_f = read_data_from_file_(target_log_path)["best_fitness"][-1]

            reference_log_path = f"{base_log_path}#{modality}#{species.lower().replace(' ', '_')}#{reference_configuration}.json"
            reference_best_f = read_data_from_file_(reference_log_path)["best_fitness"][-1]

            MBFD += (target_best_f - reference_best_f)

        MBFD /= len(species_list)

        print(f"MBFD: {MBFD}")


    @staticmethod
    def read_status(log_path, load_save):
        has_converged = False
        if load_save and os.path.exists(log_path):
            log_dictionary = read_data_from_file_(log_path)
            has_converged = log_dictionary["has_converged"]
        return has_converged

    def determine_metadata_partition_list(self, metadata_list, metadata_count, species, partitions, log_path, save_path, create_save, load_save, partition_counts=None):
        metadata_list = self.rephrase_metadata(metadata_list)
        if partition_counts is None:
            partition_counts = self.determine_partition_counts(metadata_count, species)
        priority_partition_i = None
        if self.focused_parameters["priority_partition"] != Parameters.parameter_type_defaults["priority_partition"]:
            priority_partition_i = partitions.index(self.focused_parameters["priority_partition"])
        semi_priority_partition_i = None
        if self.focused_parameters["semi_priority_partition"] != Parameters.parameter_type_defaults["semi_priority_partition"]:
            semi_priority_partition_i = partitions.index(self.focused_parameters["semi_priority_partition"])
        self.compound_attribute_class_list, self.compound_attribute_class_dictionary, self.compound_attribute_class_reference, self.compound_attribute_class_count_table, self.compound_attribute_class_sparsity_list, self.m_max, self.m_ideal, self.a_max, self.a_ideal = self.determine_compound_attribute_class_structures(metadata_list, partition_counts, priority_partition_i, semi_priority_partition_i)
        id_metadata_list, attribute_metadata_list = self.simplify_metadata(metadata_list)
        genetic_algorithm = GeneticAlgorithm(self.focused_parameters,
                                             self.compound_attribute_weight_list,
                                             self.compound_attribute_class_reference, self.compound_attribute_class_count_table, self.compound_attribute_class_sparsity_list,
                                             self.m_max, self.m_ideal, self.a_max, self.a_ideal,
                                             len(self.compound_attribute_list), len(self.compound_attribute_class_list),
                                             partition_counts, priority_partition_i, semi_priority_partition_i,
                                             attribute_metadata_list,
                                             log_path, save_path,
                                             create_save=create_save, load_save=load_save)
        best_solution, absolute_best_solution = genetic_algorithm.run()
        return best_solution, absolute_best_solution, id_metadata_list, partition_counts

    # ************************************************ MINOR METHODS ***************************************************

    def focus_parameters(self, modality, partition):
        focus_parameters = {}
        for parameter in self.parameters:
            if modality in self.parameters[parameter]:
                if partition in self.parameters[parameter][modality]:
                    focus_parameters[parameter] = self.parameters[parameter][modality][partition]
                else:
                    focus_parameters[parameter] = self.parameters[parameter][modality]
            else:
                focus_parameters[parameter] = self.parameters[parameter]
        return focus_parameters

    def determine_partition_counts(self, metadata_count, species):
        maximum_count = self.focused_parameters["maximum_observation_count"]
        if self.focused_parameters["distribution"] is not None:
            ideal_ratio = self.focused_parameters["distribution"][species]
            observation_count = min(ideal_ratio * maximum_count, metadata_count)
            observation_count = max(20, observation_count)
        else:
            observation_count = metadata_count
        observation_count = min(observation_count, maximum_count)
        partition_counts = [0] * len(self.focused_parameters["partition_ratio"])
        for i in range(0, len(partition_counts)):
            partition_counts[i] = int(self.focused_parameters["partition_ratio"][i] * observation_count)
        partition_counts[self.focused_parameters["remainder_partition"]] += (int(observation_count * sum(self.focused_parameters["partition_ratio"])) - sum(partition_counts))
        return np.array(partition_counts)

    def determine_attribute_pool_list(self):
        attribute_pool_list = []
        extended_attribute_pool_list = []
        if self.focused_parameters["balance_sex"]:
            attribute_pool_list.append("sex")
            extended_attribute_pool_list.append("sex")
        if self.focused_parameters["balance_age"]:
            attribute_pool_list.append("age")
            extended_attribute_pool_list.append("age")
        if self.focused_parameters["balance_vocalisation"]:
            attribute_pool_list.append("vocalisation")
            for k in range(len(self.focused_parameters["vocalisation_zoom"])):
                extended_attribute_pool_list.append(f"vocalisation_{k}")
        if self.focused_parameters["balance_subspecies"]:
            attribute_pool_list.append("subspecies")
            extended_attribute_pool_list.append("subspecies")
        if self.focused_parameters["balance_location"]:
            attribute_pool_list.append("location")
            for k in range(len(self.focused_parameters["location_zoom"])):
                extended_attribute_pool_list.append(f"location_{k}")
        if self.focused_parameters["balance_date"]:
            attribute_pool_list.append("date")
            for k in range(len(self.focused_parameters["date_zoom"])):
                extended_attribute_pool_list.append(f"date_{k}")
        if self.focused_parameters["balance_time"]:
            attribute_pool_list.append("time")
            for k in range(len(self.focused_parameters["time_zoom"])):
                extended_attribute_pool_list.append(f"time_{k}")
        return attribute_pool_list, extended_attribute_pool_list

    def determine_compound_attribute_structures(self):

        def create():
            compound_attribute_powerset_ = determine_powerset(self.attribute_pool_list)
            for attribute_set in compound_attribute_powerset_[:]:
                if "time" in attribute_set and ("date" not in attribute_set or "location" not in attribute_set):
                    compound_attribute_powerset_.remove(attribute_set)
                elif "date" in attribute_set and ("location" not in attribute_set):
                    compound_attribute_powerset_.remove(attribute_set)
            for attribute_set in compound_attribute_powerset_[:]:
                if "time" in attribute_set:
                    attribute_size = len(attribute_set) - 2
                elif "date" in attribute_set:
                    attribute_size = len(attribute_set) - 1
                else:
                    attribute_size = len(attribute_set)
                if attribute_size <= self.focused_parameters["maximum_attribute_size"]:
                    zoom_attribute_set_list = []
                    for attribute in attribute_set:
                        if f"{attribute}_zoom" in self.focused_parameters:
                            zoom_attribute_set_list.append(list(range(len(self.focused_parameters[f"{attribute}_zoom"]))))
                    if zoom_attribute_set_list:
                        zoom_attribute_set_product = list(itertools.product(*zoom_attribute_set_list))
                        zoom_attribute_set_list = [attribute_set for _ in range(len(zoom_attribute_set_product))]
                        for i in range(len(zoom_attribute_set_product)):
                            zoom_attribute_set = []
                            j = 0
                            for attribute in zoom_attribute_set_list[i]:
                                if f"{attribute}_zoom" in self.focused_parameters:
                                    zoom_attribute_set.append(f"{attribute}_{zoom_attribute_set_product[i][j]}")
                                    j += 1
                                else:
                                    zoom_attribute_set.append(attribute)
                            zoom_attribute_set_list[i] = tuple(zoom_attribute_set)
                        compound_attribute_powerset_.extend(zoom_attribute_set_list)
                        compound_attribute_powerset_.remove(attribute_set)
                else:
                    compound_attribute_powerset_.remove(attribute_set)
            return compound_attribute_powerset_

        def weight():
            compound_attribute_weight_list_ = []
            for compound_attribute in compound_attribute_list:
                if "time" in compound_attribute:
                    attribute_size = len(compound_attribute.split("-")) - 2
                elif "date" in compound_attribute:
                    attribute_size = len(compound_attribute.split("-")) - 1
                else:
                    attribute_size = len(compound_attribute.split("-"))
                attribute_set = compound_attribute.split("-")
                compound_attribute_weight = 0
                for attribute in attribute_set:
                    if f"{attribute}_weight" in self.focused_parameters:
                        compound_attribute_weight += self.focused_parameters[f"{attribute}_weight"]
                    else:
                        base_attribute = attribute.split('_')[0]
                        if base_attribute == "location" and ("date" in compound_attribute or "time" in compound_attribute):
                            pass
                        elif base_attribute == "date" and "time" in compound_attribute:
                            pass
                        else:
                            compound_attribute_weight += self.focused_parameters[f"{base_attribute}_weight"][int(attribute.split('_')[-1])]
                compound_attribute_weight /= (2 ** (attribute_size - 1))
                compound_attribute_weight_list_.append(round(compound_attribute_weight, 4))
            return np.array(compound_attribute_weight_list_)

        compound_attribute_list = ["-".join(attribute_set) for attribute_set in create()]
        compound_attribute_dictionary = {compound_attribute_list[i]: i for i in range(len(compound_attribute_list))}
        compound_attribute_weight_list = weight()
        compound_attribute_sparsity_list = np.array([any([attribute.split("_")[0] in ["sex", "age", "subspecies", "vocalisation"] for attribute in compound_attribute.split("-")]) for compound_attribute in compound_attribute_list])
        return compound_attribute_list, compound_attribute_dictionary, compound_attribute_weight_list, compound_attribute_sparsity_list

    def rephrase_metadata(self, metadata_list):

        def rephrase_attributes(metadata_, subset_, self_):
            attribute_dictionary_ = {}
            if "sex" in self_.attribute_pool_list:
                individual_count_ = len(metadata_["sex"])
            elif "age" in self_.attribute_pool_list:
                individual_count_ = len(metadata_["age"])
            elif "vocalisation" in self_.attribute_pool_list:
                individual_count_ = len(metadata_["general_vocalisation"])
            else:
                individual_count_ = 1
            attribute_dictionary_["individual_count"] = individual_count_
            if "subspecies" in self_.attribute_pool_list:
                subspecies = metadata_["subspecies"]
                group = metadata_["group"]
                uncertain_subspecies = metadata_.get("uncertain_subspecies", "")
                uncertain_group = metadata_.get("uncertain_group", "")
                subspecies_list = [x for x in [subspecies, group, uncertain_subspecies, uncertain_group] if x != ""]
                if subspecies_list:
                    attribute_dictionary_["subspecies"] = subspecies_list[0]
                else:
                    attribute_dictionary_["subspecies"] = "Unknown"
            if "location" in self_.attribute_pool_list:
                for i_ in range(len(self_.focused_parameters["location_zoom"])):
                    zoom_ = self_.focused_parameters["location_zoom"][i_]
                    offset_ = self_.focused_parameters["location_offset"][i_]
                    bin_ = subset_.determine_location_bin(metadata_["latitude"], metadata_["longitude"], bin_size=zoom_, bin_start_offset=offset_)
                    attribute_dictionary_[f"location_{i_}"] = bin_
            if "date" in self_.attribute_pool_list:
                for i_ in range(len(self_.focused_parameters["date_zoom"])):
                    zoom_ = self_.focused_parameters["date_zoom"][i_]
                    offset_ = self_.focused_parameters["date_offset"][i_]
                    bin_ = subset_.determine_date_bin(metadata_["date"], bin_size=zoom_, bin_start_offset=offset_)
                    attribute_dictionary_[f"date_{i_}"] = bin_
            if "time" in self_.attribute_pool_list:
                for i_ in range(len(self_.focused_parameters["time_zoom"])):
                    zoom_ = self_.focused_parameters["time_zoom"][i_]
                    offset_ = self_.focused_parameters["time_offset"][i_]
                    bin_ = subset_.determine_time_bin(metadata_["time"], bin_size=zoom_, bin_start_offset=offset_)
                    attribute_dictionary_[f"time_{i_}"] = bin_
            if "vocalisation" in self_.attribute_pool_list:
                for i_ in range(len(self_.focused_parameters["vocalisation_zoom"])):
                    attribute_dictionary_[f"vocalisation_{i_}"] = metadata_["general_vocalisation"] + metadata_["specific_vocalisation"]
            return attribute_dictionary_

        def remove_attributes(metadata_, self_):
            general_metadata_attribute_pool_ = ["general_vocalisation", "specific_vocalisation",
                                                "subspecies", "group", "uncertain_subspecies", "uncertain_group",
                                                "latitude", "longitude", "date", "time",
                                                "rating", "has_background_species"]
            if not self_.focused_parameters["balance_sex"]:
                general_metadata_attribute_pool_.append("sex")
            if not self_.focused_parameters["balance_age"]:
                general_metadata_attribute_pool_.append("age")
            for attribute_ in general_metadata_attribute_pool_:
                if attribute_ in metadata_:
                    metadata_.pop(attribute_)
            return metadata_

        for i in range(0, len(metadata_list)):
            metadata = metadata_list[i]
            attribute_dictionary = rephrase_attributes(metadata, Subset, self)
            metadata = remove_attributes(metadata, self)
            metadata_list[i] = {**metadata, **attribute_dictionary}

        return metadata_list

    def determine_compound_attribute_class_structures(self, metadata_list, partition_counts, priority_partition_i, semi_priority_partition_i):

        compound_attribute_class_list = []
        compound_attribute_class_dictionary = {}

        pruned_compound_attribute_class_list = []
        pruned_compound_attribute_class_dictionary = {}
        pruned_compound_attribute_class_reference = []
        pruned_compound_attribute_class_count_table = []
        pruned_compound_attribute_class_sparsity_list = []
        pruned_m_max = []
        pruned_m_ideal = []
        pruned_a_max = []
        pruned_a_ideal = []

        def survey():

            compound_attribute_class_reference = []
            compound_attribute_class_count_table = []
            compound_attribute_class_sparsity_list = []

            for i in range(len(metadata_list)):
                metadata = metadata_list[i]
                individual_count = metadata["individual_count"]
                compound_attribute_class_id_encounter_set = set()
                for j in range(individual_count):
                    compound_attribute_subset = self.determine_compound_attribute_subset(metadata, j)
                    for compound_attribute in compound_attribute_subset:
                        compound_attribute_i = self.compound_attribute_dictionary[compound_attribute]
                        compound_attribute_class = self.determine_compound_attribute_class(metadata, compound_attribute, j)
                        compound_attribute_class_sparsity = self.compound_attribute_sparsity_list[compound_attribute_i]
                        compound_attribute_class_id = f"{compound_attribute}#{compound_attribute_class}"
                        if compound_attribute_class_id not in compound_attribute_class_dictionary:
                            compound_attribute_class_i = len(compound_attribute_class_reference)
                            compound_attribute_class_list.append(compound_attribute_class_id)
                            compound_attribute_class_dictionary[compound_attribute_class_id] = compound_attribute_class_i
                            compound_attribute_class_reference.append(compound_attribute_i)
                            compound_attribute_class_count_table.append(0)
                            compound_attribute_class_sparsity_list.append(compound_attribute_class_sparsity)
                        else:
                            compound_attribute_class_i = compound_attribute_class_dictionary[compound_attribute_class_id]
                        if compound_attribute_class_id not in compound_attribute_class_id_encounter_set:
                            compound_attribute_class_count_table[compound_attribute_class_i] += 1
                            compound_attribute_class_id_encounter_set.add(compound_attribute_class_id)

            return np.array(compound_attribute_class_reference), np.array(compound_attribute_class_count_table), np.array(compound_attribute_class_sparsity_list)

        def analyse(compound_attribute_class_reference, compound_attribute_class_count_table, compound_attribute_class_sparsity_list):

            compound_attribute_count = len(self.compound_attribute_list)
            compound_attribute_class_count = len(compound_attribute_class_list)
            is_sparse = self.compound_attribute_sparsity_list
            is_dense = ~is_sparse
            is_class_sparse = compound_attribute_class_sparsity_list
            is_class_dense = ~is_class_sparse
            individual_size = np.sum(partition_counts)
            gene_pool_size = len(metadata_list)
            chromosome_count = partition_counts.shape[0]

            M = compound_attribute_class_count_table
            r = compound_attribute_class_reference
            p = partition_counts[np.newaxis]
            n = individual_size
            N = gene_pool_size

            m_max = np.minimum(n, M)

            m_ideal = np.empty(compound_attribute_class_count, dtype=np.float64)
            m_ideal[is_class_sparse] = m_max[is_class_sparse]
            m_ideal[is_class_dense] = M[is_class_dense] * (n / N)

            m_ideal_round = np.round(m_ideal)
            m_ideal_error = np.zeros(compound_attribute_count, dtype=np.float64)
            m_ideal_error[is_dense] = n - np.bincount(r, weights=m_ideal_round, minlength=compound_attribute_count)[is_dense]
            m_ideal_remainder = np.zeros(compound_attribute_class_count, dtype=np.float64)

            for i in range(compound_attribute_count):
                subset_i = np.where(r == i)[0]
                c = subset_i.shape[0]
                if c > 0:
                    m_ideal_remainder[subset_i] = m_ideal[subset_i] - m_ideal_round[subset_i]
                    sorted_subset_i = np.argsort(m_ideal_remainder[subset_i])
                    j = c - 1
                    while m_ideal_error[i] > 0.0:
                        k = sorted_subset_i[j]
                        if m_ideal_round[subset_i[k]] < n:
                            m_ideal_round[subset_i[k]] += 1
                            m_ideal_error[i] -= 1
                        j -= 1
                        if j == -1:
                            m_ideal_remainder[subset_i] = m_ideal[subset_i] - m_ideal_round[subset_i]
                            sorted_subset_i = np.argsort(m_ideal_remainder[subset_i])
                            j = c - 1
                    j = 0
                    while m_ideal_error[i] < 0.0:
                        k = sorted_subset_i[j]
                        if m_ideal_round[subset_i[k]] > 0:
                            m_ideal_round[subset_i[k]] -= 1
                            m_ideal_error[i] += 1
                        j += 1
                        if j == c:
                            m_ideal_remainder[subset_i] = m_ideal[subset_i] - m_ideal_round[subset_i]
                            sorted_subset_i = np.argsort(m_ideal_remainder[subset_i])
                            j = 0

            a_max = np.minimum(p, m_max[np.newaxis].T)

            a_ideal = np.matmul(m_ideal_round[np.newaxis].T, p) / n
            a_ideal_round = np.round(a_ideal)
            a_ideal_round[np.tile(is_class_sparse.reshape(is_class_sparse.shape[0], 1), (1, chromosome_count)) & (a_ideal_round == 0.0)] = 1.0
            a_ideal_m_ideal_is_1 = np.zeros(chromosome_count, dtype=np.float64)
            if priority_partition_i is not None:
                a_ideal_m_ideal_is_1[priority_partition_i] = 1.0
            a_ideal_m_ideal_is_2 = np.zeros(chromosome_count, dtype=np.float64)
            if priority_partition_i is not None:
                a_ideal_m_ideal_is_2[priority_partition_i] = 1.0
            if semi_priority_partition_i is not None:
                a_ideal_m_ideal_is_2[semi_priority_partition_i] = 1.0
            a_ideal_round[is_class_sparse & (m_ideal_round == 1)] = a_ideal_m_ideal_is_1
            a_ideal_round[is_class_sparse & (m_ideal_round == 2)] = a_ideal_m_ideal_is_2
            a_ideal_error = m_ideal_round - np.sum(a_ideal_round, axis=1)
            a_ideal_remainder = a_ideal - a_ideal_round
            sorted_i = np.argsort(a_ideal_remainder, axis=1)

            for i in range(compound_attribute_class_count):
                j = chromosome_count - 1
                while a_ideal_error[i] > 0.0:
                    k = sorted_i[i][j]
                    if a_ideal_round[i][k] < a_max[i][j]:
                        a_ideal_round[i][k] += 1
                        a_ideal_error[i] -= 1
                    j -= 1
                    if j == -1:
                        a_ideal_remainder = a_ideal - a_ideal_round
                        sorted_i = np.argsort(a_ideal_remainder, axis=1)
                        j = chromosome_count - 1
                j = 0
                while a_ideal_error[i] < 0.0:
                    k = sorted_i[i][j]
                    if (is_class_sparse[i] and a_ideal_round[i][k] > 1) or (is_class_dense[i] and a_ideal_round[i][k] > 0):
                        a_ideal_round[i][k] -= 1
                        a_ideal_error[i] += 1
                    j += 1
                    if j == chromosome_count:
                        a_ideal_remainder = a_ideal - a_ideal_round
                        sorted_i = np.argsort(a_ideal_remainder, axis=1)
                        j = 0

            return m_max, m_ideal_round, a_max, a_ideal_round

        def edit(m_max, m_ideal, a_max, a_ideal, compound_attribute_class_reference, compound_attribute_class_count_table, compound_attribute_class_sparsity_list):

            new_compound_attribute_class_i = 0

            for compound_attribute_class_i in range(len(compound_attribute_class_list)):

                compound_attribute_class_id = compound_attribute_class_list[compound_attribute_class_i]

                compound_attribute = compound_attribute_class_id.split("#")[0]
                attribute_size = len(compound_attribute.split("-"))

                if m_ideal[compound_attribute_class_i] > (attribute_size - 1):
                    compound_attribute_i = compound_attribute_class_reference[compound_attribute_class_i]
                    compound_attribute_class_count = compound_attribute_class_count_table[compound_attribute_class_i]
                    compound_attribute_class_sparsity = compound_attribute_class_sparsity_list[compound_attribute_class_i]

                    pruned_compound_attribute_class_list.append(compound_attribute_class_id)
                    pruned_compound_attribute_class_dictionary[compound_attribute_class_id] = new_compound_attribute_class_i
                    pruned_compound_attribute_class_reference.append(compound_attribute_i)
                    pruned_compound_attribute_class_count_table.append(compound_attribute_class_count)
                    pruned_compound_attribute_class_sparsity_list.append(compound_attribute_class_sparsity)
                    pruned_m_max.append(m_max[compound_attribute_class_i])
                    pruned_m_ideal.append(m_ideal[compound_attribute_class_i])
                    pruned_a_max.append(a_max[compound_attribute_class_i])
                    pruned_a_ideal.append(a_ideal[compound_attribute_class_i])

                    new_compound_attribute_class_i += 1

        if self.attribute_pool_list:
            compound_attribute_class_reference_, compound_attribute_class_count_table_, compound_attribute_class_sparsity_list_ = survey()
            if compound_attribute_class_reference_.shape[0] != 0:
                m_max_, m_ideal_, a_max_, a_ideal_round_ = analyse(compound_attribute_class_reference_, compound_attribute_class_count_table_, compound_attribute_class_sparsity_list_)
                edit(m_max_, m_ideal_, a_max_, a_ideal_round_, compound_attribute_class_reference_, compound_attribute_class_count_table_, compound_attribute_class_sparsity_list_)

        return pruned_compound_attribute_class_list, pruned_compound_attribute_class_dictionary, np.array(pruned_compound_attribute_class_reference), np.array(pruned_compound_attribute_class_count_table), np.array(pruned_compound_attribute_class_sparsity_list), np.array(pruned_m_max), np.array(pruned_m_ideal), np.array(pruned_a_max), np.array(pruned_a_ideal)

    def simplify_metadata(self, metadata_list):

        id_metadata_list = []
        attribute_metadata_list = []

        def simplify_attributes(metadata_):
            attribute_metadata_ = []
            individual_count_ = metadata_["individual_count"]
            compound_attribute_class_id_encounter_set_ = set()
            for j_ in range(individual_count_):
                compound_attribute_subset_ = self.determine_compound_attribute_subset(metadata_, j_)
                for compound_attribute_ in compound_attribute_subset_:
                    compound_attribute_class_ = self.determine_compound_attribute_class(metadata_, compound_attribute_, j_)
                    compound_attribute_class_id_ = f"{compound_attribute_}#{compound_attribute_class_}"
                    if compound_attribute_class_id_ in self.compound_attribute_class_dictionary:
                        if compound_attribute_class_id_ not in compound_attribute_class_id_encounter_set_:
                            compound_attribute_class_i_ = self.compound_attribute_class_dictionary[compound_attribute_class_id_]
                            attribute_metadata_.append(compound_attribute_class_i_)
                            compound_attribute_class_id_encounter_set_.add(compound_attribute_class_id_)
            return np.array(attribute_metadata_, dtype=np.int32)

        for i in range(len(metadata_list)):
            metadata = metadata_list[i]
            id_metadata_list.append(metadata["_id"])
            if self.attribute_pool_list:
                attribute_metadata = simplify_attributes(metadata)
                attribute_metadata_list.append(attribute_metadata)

        return id_metadata_list, attribute_metadata_list

    # ************************************************ UTILITY METHODS ***************************************************

    def determine_compound_attribute_subset(self, metadata, individual_i):
        extended_attribute_pool_subset = set()
        for attribute in self.extended_attribute_pool_list:
            if isinstance(metadata[attribute], list):
                if metadata[attribute][individual_i] != "Unknown":
                    extended_attribute_pool_subset.add(attribute)
            else:
                if metadata[attribute] != "Unknown":
                    extended_attribute_pool_subset.add(attribute)
        compound_attribute_subset = [compound_attribute for compound_attribute in self.compound_attribute_list if set(compound_attribute.split("-")).issubset(extended_attribute_pool_subset)]
        return compound_attribute_subset

    @staticmethod
    def determine_compound_attribute_class(metadata, compound_attribute, individual_i):
        compound_attribute_class = []
        for attribute in compound_attribute.split("-"):
            if isinstance(metadata[attribute], list):
                compound_attribute_class.append(str(metadata[attribute][individual_i]))
            else:
                compound_attribute_class.append(str(metadata[attribute]))
        compound_attribute_class = "-".join(compound_attribute_class)
        return compound_attribute_class
