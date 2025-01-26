import geopy.distance
import re
import random
import math
from datetime import datetime

from dataset_generators.visualisation import best_fitness_improvement_distribution_graph, agreement_improvement_distribution_graph
from utility import *
import pprint


class Pairer:

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self, constructor, parameters):

        self.constructor = constructor
        self.parameters = parameters
        self.focused_parameters = None

    # ************************************************ MAJOR METHODS ***************************************************

    def source(self, configuration_id, partition_configuration, partition_solution_type, partition):

        old_master_database_structure = self.constructor.get_master_database_structure(partition_configuration, dataset_type_="partition")
        new_master_database_structure = self.constructor.get_master_database_structure(configuration_id, dataset_type_="pair")

        if "image-audio" not in new_master_database_structure:
            new_master_database_structure["image-audio"] = {}
        if partition not in new_master_database_structure["image-audio"]:
            new_master_database_structure["image-audio"][partition] = [old_master_database_structure["image-audio"][partition], old_master_database_structure["image-audio"][partition]]

        self.constructor.save_master_database_structure(configuration_id, new_master_database_structure, dataset_type_="pair")

        pair_dataset = self.constructor.get_dataset_handle(configuration_id, "image-audio", partition, dataset_type_="pair")
        partition_dataset = self.constructor.get_dataset_handle(partition_configuration, "image-audio", partition, dataset_type_="partition")
        species_list = self.constructor.read_species_list(partition_dataset)

        i = 0

        for species in species_list:

            observation_list = self.constructor.read_observation_list(partition_dataset, species, dataset_type_="partition", solution_type_=partition_solution_type)
            paired_observation_list = [[observation, observation] for observation in observation_list]

            self.constructor.update_observation_list(pair_dataset, species, paired_observation_list, dataset_type_="pair", solution_type_="source")

            print(f"Species: {i}")
            i += 1

    def control(self, configuration_id, image_partition_configuration, audio_partition_configuration, partition_solution_type, partition, log_path):

        species_limit = 1000

        old_image_master_database_structure = self.constructor.get_master_database_structure(image_partition_configuration, dataset_type_="partition")
        old_audio_master_database_structure = self.constructor.get_master_database_structure(audio_partition_configuration, dataset_type_="partition")
        new_master_database_structure = self.constructor.get_master_database_structure(configuration_id, dataset_type_="pair")

        if "image-audio" not in new_master_database_structure:
            new_master_database_structure["image-audio"] = {}
        if partition not in new_master_database_structure["image-audio"]:
            new_master_database_structure["image-audio"][partition] = [old_image_master_database_structure["image"][partition], old_audio_master_database_structure["audio"][partition]]

        self.constructor.save_master_database_structure(configuration_id, new_master_database_structure, dataset_type_="pair")

        pair_experiment_dataset = self.constructor.get_dataset_handle(configuration_id, "image-audio", partition, dataset_type_="pair")

        image_dataset = self.constructor.get_dataset_handle(image_partition_configuration, "image", partition, dataset_type_="partition")
        audio_dataset = self.constructor.get_dataset_handle(audio_partition_configuration, "audio", partition, dataset_type_="partition")

        species_list = self.constructor.read_species_list(audio_dataset)

        if os.path.exists(log_path):
            logs = set(read_data_from_file_(log_path))
        else:
            logs = set()

        i = 0

        for species in species_list:

            if species not in logs:

                image_observation_list = self.constructor.read_observation_list(image_dataset, species, dataset_type_="partition", solution_type_=partition_solution_type)
                audio_observation_list = self.constructor.read_observation_list(audio_dataset, species, dataset_type_="partition", solution_type_=partition_solution_type)

                paired_observation_list = self.determine_control_paired_observation_list(image_observation_list, audio_observation_list, partition)

                self.constructor.update_observation_list(pair_experiment_dataset, species, paired_observation_list, dataset_type_="pair", solution_type_="control")

                logs.add(species)

                if i > species_limit:
                    break

                print(f"Species: {i}")
                i += 1

        save_data_to_file_(log_path, logs)

    def experiment(self, configuration_id, image_partition_configuration, audio_partition_configuration, partition_solution_type, partition, log_path):

        species_limit = 1000

        old_image_master_database_structure = self.constructor.get_master_database_structure(image_partition_configuration, dataset_type_="partition")
        old_audio_master_database_structure = self.constructor.get_master_database_structure(audio_partition_configuration, dataset_type_="partition")
        new_master_database_structure = self.constructor.get_master_database_structure(configuration_id, dataset_type_="pair")

        if "image-audio" not in new_master_database_structure:
            new_master_database_structure["image-audio"] = {}
        if partition not in new_master_database_structure["image-audio"]:
            new_master_database_structure["image-audio"][partition] = [old_image_master_database_structure["image"][partition], old_audio_master_database_structure["audio"][partition]]

        self.constructor.save_master_database_structure(configuration_id, new_master_database_structure, dataset_type_="pair")

        pair_experiment_dataset = self.constructor.get_dataset_handle(configuration_id, "image-audio", partition, dataset_type_="pair")

        image_dataset = self.constructor.get_dataset_handle(image_partition_configuration, "image", partition, dataset_type_="partition")
        audio_dataset = self.constructor.get_dataset_handle(audio_partition_configuration, "audio", partition, dataset_type_="partition")
        image_source_metadata = self.constructor.get_source_metadata_handle(image_partition_configuration, "image", partition, dataset_type_="partition")
        audio_source_metadata = self.constructor.get_source_metadata_handle(audio_partition_configuration, "audio", partition, dataset_type_="partition")

        species_list = self.constructor.read_species_list(audio_dataset)

        if os.path.exists(log_path):
            logs = read_data_from_file_(log_path)
        else:
            logs = {}

        i = 0

        for species in species_list:

            if species not in logs:

                image_metadata_list = self.constructor.read_metadata_list(image_dataset, image_source_metadata, species, dataset_type_="partition", solution_type_=partition_solution_type)
                audio_metadata_list = self.constructor.read_metadata_list(audio_dataset, audio_source_metadata, species, dataset_type_="partition", solution_type_=partition_solution_type)

                # if len(image_metadata_list) < 300 and len(audio_metadata_list) < 300:

                if True:

                    print(f"Species {i}: ({len(image_metadata_list)}, {len(audio_metadata_list)})")

                    paired_observation_list, initial_mean_agreement, final_mean_agreement = self.determine_experiment_paired_metadata_list(image_metadata_list, audio_metadata_list, partition)

                    self.constructor.update_observation_list(pair_experiment_dataset, species, paired_observation_list, dataset_type_="pair", solution_type_="best")

                    logs[species] = {"initial_mean_agreement": initial_mean_agreement, "final_mean_agreement": final_mean_agreement}

                    if i > species_limit:
                        break

                    print()
                    i += 1

        save_data_to_file_(log_path, logs)


    def evaluate_all(self, configuration_id, base_log_path):

        MAI_lists = []
        observation_count_lists = []

        for partition in ["train", "validation", "test"]:

            log_path = f"{base_log_path}#{partition}#{configuration_id}.json"

            dataset = self.constructor.get_dataset_handle(configuration_id, "image-audio", partition, dataset_type_="pair")
            species_list = self.constructor.read_species_list(dataset)

            if os.path.exists(log_path):
                logs = read_data_from_file_(log_path)
            else:
                logs = {}

            MAI_list = []
            observation_count_list = []

            for species in species_list:
                FA = logs[species]["final_mean_agreement"]
                IA = logs[species]["initial_mean_agreement"]
                MAI = (FA - IA)
                observation_count = self.constructor.read_observation_count(dataset, species, dataset_type_="pair", solution_type_="best")
                MAI_list.append(MAI)
                observation_count_list.append(observation_count)

            MAI_lists.append(MAI_list)
            observation_count_lists.append(observation_count_list)

        agreement_improvement_distribution_graph(["Training Set", "Validation Set", "Testing Set"], MAI_lists, observation_count_lists)


    # ************************************************ MINOR METHODS **************************************************

    def determine_control_paired_observation_list(self, image_observation_list, audio_observation_list, partition):

        self.focused_parameters = self.focus_parameters("image-audio", partition)

        random.shuffle(image_observation_list)
        random.shuffle(audio_observation_list)

        pair_list = []
        for i in range(len(image_observation_list)):
            for j in range(len(audio_observation_list)):
                pair = (i, j)
                pair_list.append(pair)

        random.shuffle(pair_list)

        pruned_pair_list = self.prune_pair_list(pair_list, [1 for i in range(len(pair_list))], len(image_observation_list), len(audio_observation_list))[0]

        paired_observation_list = [[image_observation_list[pair[0]], audio_observation_list[pair[1]]] for pair in pruned_pair_list]

        return paired_observation_list

    def determine_experiment_paired_metadata_list(self, image_metadata_list, audio_metadata_list, partition):

        self.focused_parameters = self.extend_parameters(self.focus_parameters("image-audio", partition))

        print("Rephrasing Metadata...")

        image_metadata_list = self.rephrase_metadata(image_metadata_list)
        audio_metadata_list = self.rephrase_metadata(audio_metadata_list)

        print("Creating Pairs...")

        pair_list = []
        individual_agreement_list = []
        context_agreement_list = []
        agreement_list = []

        initial_agreement = 0.0

        for i in range(len(image_metadata_list)):
            for j in range(len(audio_metadata_list)):
                agreement, individual_agreement, context_agreement = self.determine_status(image_metadata_list[i], audio_metadata_list[j])
                pair = (i, j)
                pair_list.append(pair)
                agreement_list.append(agreement)
                individual_agreement_list.append(individual_agreement)
                context_agreement_list.append(context_agreement)
                initial_agreement += agreement

        initial_agreement /= len(agreement_list)

        print("Sorting Pairs...")

        context_agreement_list, individual_agreement_list, agreement_list, pair_list = zip(*sorted(zip(context_agreement_list, individual_agreement_list, agreement_list, pair_list), reverse=True))

        print("Pruning Pairs...")

        pruned_pair_list, pruned_agreement_list = self.prune_pair_list(pair_list, agreement_list, len(image_metadata_list), len(audio_metadata_list))

        final_agreement = sum(pruned_agreement_list) / len(pruned_agreement_list)

        paired_observation_list = [[image_metadata_list[pair[0]]["_id"], audio_metadata_list[pair[1]]["_id"]] for pair in pruned_pair_list]

        return paired_observation_list, initial_agreement, final_agreement

    def prune_pair_list(self, pair_list, agreement_list, image_observation_count, audio_observation_count):

        lower_modality_k, upper_modality_k, lower_threshold, lower_threshold_count, upper_threshold, upper_threshold_count = self.determine_threshold_parameters(image_observation_count, audio_observation_count)

        def prune(pruned_pair_list_, pruned_agreement_list_, i_, lower_modality_encounter_dictionary_, upper_modality_encounter_dictionary_, threshold_, threshold_count_):

            while threshold_count_ > 0 and i_ < len(pair_list):

                lower_modality_i = pair_list[i_][lower_modality_k]
                upper_modality_i = pair_list[i_][upper_modality_k]

                lower_modality_encounter_dictionary_ = update_frequency_dictionary_(lower_modality_i, 0, lower_modality_encounter_dictionary_)
                upper_modality_encounter_dictionary_ = update_frequency_dictionary_(upper_modality_i, 0, upper_modality_encounter_dictionary_)

                if lower_modality_encounter_dictionary_[lower_modality_i] < threshold_ and upper_modality_encounter_dictionary_[upper_modality_i] < 1:

                    if pair_list[i_] not in pruned_pair_list_:

                        pruned_pair_list_.append(pair_list[i_])
                        pruned_agreement_list_.append(agreement_list[i_])

                        lower_modality_encounter_dictionary_ = update_frequency_dictionary_(lower_modality_i, 1, lower_modality_encounter_dictionary_)
                        upper_modality_encounter_dictionary_ = update_frequency_dictionary_(upper_modality_i, 1, upper_modality_encounter_dictionary_)

                        if lower_modality_encounter_dictionary_[lower_modality_i] == threshold_:
                            threshold_count_ -= 1

                i_ += 1

            return pruned_pair_list_, pruned_agreement_list_, i_, lower_modality_encounter_dictionary_, upper_modality_encounter_dictionary_, threshold_count_

        pruned_pair_list, pruned_agreement_list, i, lower_modality_encounter_dictionary, upper_modality_encounter_dictionary, upper_threshold_count = prune([], [],0, {}, {}, upper_threshold, upper_threshold_count)
        pruned_pair_list, pruned_agreement_list, i, lower_modality_encounter_dictionary, upper_modality_encounter_dictionary, lower_threshold_count = prune(pruned_pair_list, pruned_agreement_list, i, lower_modality_encounter_dictionary, upper_modality_encounter_dictionary, lower_threshold, lower_threshold_count)

        return pruned_pair_list, pruned_agreement_list

    def determine_threshold_parameters(self, image_observation_count, audio_observation_count):

        def calculate(upper_observation_count, lower_observation_count):
            lower_threshold_ = int(upper_observation_count / lower_observation_count)
            upper_threshold_ = lower_threshold_ + 1
            upper_threshold_count_ = upper_observation_count - (lower_threshold_ * lower_observation_count)
            lower_threshold_count_ = lower_observation_count - upper_threshold_count_
            return lower_threshold_, lower_threshold_count_, upper_threshold_, upper_threshold_count_

        if image_observation_count >= audio_observation_count:
            upper_modality_i = 0
            lower_modality_i = 1
            lower_threshold, lower_threshold_count, upper_threshold, upper_threshold_count = calculate(image_observation_count, audio_observation_count)
        else:
            upper_modality_i = 1
            lower_modality_i = 0
            lower_threshold, lower_threshold_count, upper_threshold, upper_threshold_count = calculate(audio_observation_count, image_observation_count)

        return lower_modality_i, upper_modality_i, lower_threshold, lower_threshold_count, upper_threshold, upper_threshold_count

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

    def extend_parameters(self, focus_parameters):

        focus_parameters["individual_weight"] = 0
        focus_parameters["context_weight"] = 0

        if focus_parameters["pair_sex"]:
            focus_parameters["Male_weight"] = 1.0
            focus_parameters["Female_weight"] = 4.0
            focus_parameters["sex_weight_bound"] = focus_parameters["Male_weight"] + focus_parameters["Female_weight"]
            focus_parameters["individual_weight"] += focus_parameters["sex_weight"]

        if focus_parameters["pair_age"]:
            focus_parameters["Adult_weight"] = 1.0
            focus_parameters["Ambiguous_weight"] = 4.0
            focus_parameters["Juvenile_weight"] = 4.0
            focus_parameters["Nestling_weight"] = 4.0
            focus_parameters["age_weight_bound"] = focus_parameters["Adult_weight"] + focus_parameters["Ambiguous_weight"] + focus_parameters["Juvenile_weight"] + focus_parameters["Nestling_weight"]
            focus_parameters["individual_weight"] += focus_parameters["age_weight"]

        if focus_parameters["pair_subspecies"]:
            focus_parameters["subspecies_class_weight"] = 1.0
            focus_parameters["subspecies_weight_bound"] = focus_parameters["subspecies_class_weight"] * 3.0
            focus_parameters["individual_weight"] += focus_parameters["subspecies_weight"]

        if focus_parameters["pair_location"]:
            focus_parameters["context_weight"] += focus_parameters["location_weight"]

        if focus_parameters["pair_date"]:
            focus_parameters["context_weight"] += focus_parameters["date_weight"]

        if focus_parameters["pair_time"]:
            focus_parameters["context_weight"] += focus_parameters["time_weight"]

        focus_parameters["weight"] = focus_parameters["individual_weight"] + focus_parameters["context_weight"]

        return focus_parameters

    def rephrase_metadata(self, metadata_list):

        def rephrase_attributes(metadata_):
            attribute_dictionary_ = {}
            if self.focused_parameters["pair_sex"]:
                attribute_dictionary_["sex"] = []
                for individual_ in metadata_["sex"]:
                    if individual_ != "Unknown":
                        attribute_dictionary_["sex"].append(individual_)
                attribute_dictionary_["sex"] = set(attribute_dictionary_["sex"])
            if self.focused_parameters["pair_age"]:
                attribute_dictionary_["age"] = []
                for individual_ in metadata_["age"]:
                    if individual_ != "Unknown":
                        if individual_ == "Adult":
                            attribute_dictionary_["age"].append("Adult")
                        if individual_ == "Immature" or individual_ == "Subadult":
                            attribute_dictionary_["age"].append("Ambiguous")
                        if individual_ == "Juvenile" or individual_ == "Young":
                            attribute_dictionary_["age"].append("Juvenile")
                        if individual_ == "Nestling" or individual_ == "Fledgling" or individual_ == "Hatchling":
                            attribute_dictionary_["age"].append("Nestling")
                attribute_dictionary_["age"] = set(attribute_dictionary_["age"])
            if self.focused_parameters["pair_subspecies"]:
                subspecies_ = metadata_["subspecies"]
                group_ = metadata_["group"]
                uncertain_subspecies_ = metadata_.get("uncertain_subspecies", "")
                uncertain_group_ = metadata_.get("uncertain_group", "")
                subspecies_list_ = [x for x in [subspecies_, group_, uncertain_subspecies_, uncertain_group_] if x != ""]
                attribute_dictionary_["subspecies"] = "" if not subspecies_list_ else subspecies_list_[0].split(None, 2)[2]
                attribute_dictionary_["subspecies"] = set() if attribute_dictionary_["subspecies"] == "" else set(re.split(r"\W+", attribute_dictionary_["subspecies"]))
            if self.focused_parameters["pair_location"]:
                attribute_dictionary_["location"] = (metadata_["latitude"], metadata_["longitude"])
            if self.focused_parameters["pair_date"]:
                attribute_dictionary_["date"] = f"{metadata_['date'].split('-')[1]}-{metadata_['date'].split('-')[2]}"
            return attribute_dictionary_

        def remove_attributes(metadata_):
            general_metadata_attribute_pool_ = ["sex", "age", "general_vocalisation", "specific_vocalisation",
                                                "subspecies", "group", "uncertain_subspecies", "uncertain_group",
                                                "observer",
                                                "latitude", "longitude", "date",
                                                "duration", "rating", "has_background_species", "media_types", "media_links"]
            for attribute_ in general_metadata_attribute_pool_:
                if attribute_ in metadata_:
                    metadata_.pop(attribute_)
            return metadata_

        for i in range(0, len(metadata_list)):
            metadata = metadata_list[i]
            attribute_dictionary = rephrase_attributes(metadata)
            metadata = remove_attributes(metadata)
            metadata_list[i] = {**metadata, **attribute_dictionary}

        return metadata_list

    def determine_status(self, image_metadata, audio_metadata):
        individual_agreement = 0
        context_agreement = 0
        if self.focused_parameters["pair_sex"]:
            sex_agreement = self.agreement_by_sex(image_metadata, audio_metadata)
            individual_agreement += sex_agreement
        if self.focused_parameters["pair_age"]:
            age_agreement = self.agreement_by_age(image_metadata, audio_metadata)
            individual_agreement += age_agreement
        if self.focused_parameters["pair_subspecies"]:
            subspecies_agreement = self.agreement_by_subspecies(image_metadata, audio_metadata)
            individual_agreement += subspecies_agreement
        if self.focused_parameters["pair_location"]:
            location_agreement = self.agreement_by_location(image_metadata, audio_metadata)
            context_agreement += location_agreement
        if self.focused_parameters["pair_date"]:
            date_agreement = self.agreement_by_date(image_metadata, audio_metadata)
            context_agreement += date_agreement
        if self.focused_parameters["pair_time"]:
            time_agreement = self.agreement_by_time(image_metadata, audio_metadata)
            context_agreement += time_agreement
        agreement = individual_agreement + context_agreement
        agreement /= self.focused_parameters["weight"]
        individual_agreement /= self.focused_parameters["individual_weight"]
        context_agreement /= self.focused_parameters["context_weight"]
        return agreement, individual_agreement, context_agreement

    def agreement_by_sex(self, image_metadata, audio_metadata):
        sex_union = image_metadata["sex"].union(audio_metadata["sex"])
        sex_intersection = image_metadata["sex"].intersection(audio_metadata["sex"])
        if sex_intersection:
            agreement = (sum([self.focused_parameters[f"{sex}_weight"] for sex in sex_intersection]) / self.focused_parameters["sex_weight_bound"])
            agreement = agreement * 0.5 * len(sex_intersection) / len(sex_union) + 0.5
        elif sex_union:
            agreement = (sum([self.focused_parameters[f"{sex}_weight"] for sex in sex_union]) / self.focused_parameters["sex_weight_bound"])
            agreement = agreement * 0.5
        else:
            agreement = 0.5
        agreement = agreement * self.focused_parameters["sex_weight"]
        return agreement

    def agreement_by_age(self, image_metadata, audio_metadata):
        age_union = image_metadata["age"].union(audio_metadata["age"])
        age_intersection = image_metadata["age"].intersection(audio_metadata["age"])
        if age_intersection:
            agreement = (sum([self.focused_parameters[f"{age}_weight"] for age in age_intersection]) / self.focused_parameters["age_weight_bound"])
            agreement = agreement * 0.5 * len(age_intersection) / len(age_union) + 0.5
        elif age_union:
            agreement = (sum([self.focused_parameters[f"{age}_weight"] for age in age_union]) / self.focused_parameters["age_weight_bound"])
            agreement = agreement * 0.5
        else:
            agreement = 0.5
        agreement = agreement * self.focused_parameters["age_weight"]
        return agreement

    def agreement_by_subspecies(self, image_metadata, audio_metadata):
        subspecies_union = image_metadata["subspecies"].union(audio_metadata["subspecies"])
        subspecies_intersection = image_metadata["subspecies"].intersection(audio_metadata["subspecies"])
        if subspecies_intersection:
            agreement = min(self.focused_parameters["subspecies_class_weight"] * len(subspecies_intersection), self.focused_parameters["subspecies_weight_bound"])
            agreement = agreement * 0.5 * len(subspecies_intersection) / len(subspecies_union) + 0.5
        elif subspecies_union:
            agreement = min(self.focused_parameters["subspecies_class_weight"] * len(subspecies_union), self.focused_parameters["subspecies_weight_bound"])
            agreement = agreement * 0.5
        else:
            agreement = 0.5
        agreement = agreement * self.focused_parameters["subspecies_weight"]
        return agreement

    def agreement_by_location(self, image_metadata, audio_metadata):
        distance = geopy.distance.distance(image_metadata["location"], audio_metadata["location"]).km
        if distance <= self.focused_parameters["location_cap"]:
            agreement = (1.0 - (distance / self.focused_parameters["location_cap"])) * self.focused_parameters["location_weight"]
        else:
            agreement = 0.0
        return agreement

    def agreement_by_date(self, image_metadata, audio_metadata):
        distance = abs((datetime.strptime(f"2020-{image_metadata['date']}", "%Y-%m-%d").date() - datetime.strptime(f"2020-{audio_metadata['date']}", "%Y-%m-%d").date()).days)
        if distance > 182:
            distance = 365 - distance
        if distance <= self.focused_parameters["date_cap"]:
            agreement = (1.0 - (distance / self.focused_parameters["date_cap"])) * self.focused_parameters["date_weight"]
        else:
            agreement = 0.0
        return agreement

    def agreement_by_time(self, image_metadata, audio_metadata):
        distance = abs((datetime.strptime(image_metadata["time"], "%H:%M") - datetime.strptime(audio_metadata["time"], "%H:%M")).total_seconds() / 60.0)
        if distance > 720:
            distance = 1440 - distance
        if distance <= self.focused_parameters["time_cap"]:
            agreement = (1.0 - (distance / self.focused_parameters["time_cap"])) * self.focused_parameters["time_weight"]
        else:
            agreement = 0.0
        return agreement

    # ******************************************************************************************************************
