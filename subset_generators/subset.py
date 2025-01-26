import re
from pymongo import MongoClient
from timezonefinder import TimezoneFinder
from datetime import datetime
from datetime import timedelta
from dataset_generators.constants import *
from utility import *
import unidecode
import time
import shutil
import pytz


class Subset:

    # Utilities
    timezone_finder = TimezoneFinder(in_memory=True)

    # Arrays
    groups_to_replace = read_data_from_file_(f"{RESOURCE_PATH}arrays/groups_to_replace.json")

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self, subset_path, subset_tag, project_tag, parent_project_tag, minimum_year, maximum_year, base_taxonomy, subset_taxonomy_to_base_taxonomy_replacements, aligned_taxonomy, base_taxonomy_to_aligned_taxonomy_replacements):

        # Paths
        self.subset_path = subset_path
        self.downloaded_metadata_files_path = f"{self.subset_path}/Downloaded Metadata/"
        self.metadata_files_path = f"{self.subset_path}/Metadata/"
        self.aligned_metadata_files_path = f"{self.subset_path}/Aligned Metadata/"
        self.statistics_files_path = f"{self.subset_path}/Statistics/"
        self.aligned_statistics_files_path = f"{self.subset_path}/Aligned Statistics/"
        self.aligned_data_files_path = f"{self.subset_path}/Aligned Data/"
        self.aligned_signal_files_path = f"{self.subset_path}/Aligned Signal/"
        self.aligned_noise_files_path = f"{self.subset_path}/Aligned Noise/"
        self.aligned_spectrogram_files_path = f"{self.subset_path}/Aligned Spectrogram/"
        self.aligned_dataset_files_path = f"{self.subset_path}/Aligned Dataset/"

        self.external_subset_path = self.subset_path.replace("C:/Users/sanam/Documents", "D:")
        self.external_aligned_data_files_path = f"{self.external_subset_path}/Aligned Data/"
        self.external_aligned_signal_files_path = f"{self.external_subset_path}/Aligned Signal/"
        self.external_aligned_noise_files_path = f"{self.external_subset_path}/Aligned Noise/"
        self.external_aligned_spectrogram_files_path = f"{self.external_subset_path}/Aligned Spectrogram/"

        # Tags
        self.subset_tag = subset_tag
        self.project_tag = project_tag
        self.parent_project_tag = parent_project_tag

        # Database Names
        self.metadata_database_name = self.subset_tag
        self.aligned_metadata_database_name = f"aligned_{self.subset_tag}"

        # Years
        self.minimum_year = minimum_year
        self.maximum_year = maximum_year

        # Taxonomy
        self.base_taxonomy = base_taxonomy
        self.subset_taxonomy_to_base_taxonomy_replacements = subset_taxonomy_to_base_taxonomy_replacements
        self.aligned_taxonomy = aligned_taxonomy
        self.base_taxonomy_to_aligned_taxonomy_replacements = base_taxonomy_to_aligned_taxonomy_replacements

    # ************************************************ MAJOR METHODS ***************************************************

    def update_progress(self, data_type):

        if data_type == "data":
            aligned_path = self.aligned_data_files_path
            external_aligned_path = self.external_aligned_data_files_path
            progress_file_path = f"{self.subset_path}/downloaded.txt"
        elif data_type == "signal":
            aligned_path = self.aligned_signal_files_path
            external_aligned_path = self.external_aligned_signal_files_path
            progress_file_path = f"{self.subset_path}/signal_segmented.txt"
        elif data_type == "noise":
            aligned_path = self.aligned_noise_files_path
            external_aligned_path = self.external_aligned_noise_files_path
            progress_file_path = f"{self.subset_path}/noise_segmented.txt"
        elif data_type == "spectrogram":
            aligned_path = self.aligned_spectrogram_files_path
            external_aligned_path = self.external_aligned_spectrogram_files_path
            progress_file_path = f"{self.subset_path}/spectrogram_segmented.txt"
        else:
            aligned_path = None
            external_aligned_path = None
            progress_file_path = None

        if os.path.exists(external_aligned_path):

            data_id_list = []

            # for species in os.listdir(aligned_path):
            #     species_path = f"{aligned_path}{species}/"
            #     for data_name in os.listdir(species_path):
            #         if data_type == "spectrogram":
            #             data_id = f"{species}/{'_'.join(data_name.split('.')[0].split('_')[:-2])}"
            #         else:
            #             if "xenocanto" in data_name:
            #                 data_id = f"{species}/{data_name.split('.')[0]}"
            #                 data_id_list.append(data_id)

            for species in os.listdir(external_aligned_path):
                species_path = f"{external_aligned_path}{species}/"
                for data_name in os.listdir(species_path):
                    if data_type == "spectrogram":
                        data_id = f"{species}/{'_'.join(data_name.split('.')[0].split('_')[:-2])}"
                    else:
                        if "xenocanto" in data_name:
                            data_id = f"{species}/{data_name.split('.')[0]}"
                            data_id_list.append(data_id)

            save_data_to_file_(progress_file_path, data_id_list)

        else:

            print("Please connect hard drive!")

    def transfer(self, data_type):

        print("Transferring data...")

        if data_type == "data":
            aligned_path = self.aligned_data_files_path
            external_aligned_path = self.external_aligned_data_files_path
        elif data_type == "signal":
            aligned_path = self.aligned_signal_files_path
            external_aligned_path = self.external_aligned_signal_files_path
        elif data_type == "noise":
            aligned_path = self.aligned_noise_files_path
            external_aligned_path = self.external_aligned_noise_files_path
        elif data_type == "spectrogram":
            aligned_path = self.aligned_spectrogram_files_path
            external_aligned_path = self.external_aligned_spectrogram_files_path
        else:
            aligned_path = None
            external_aligned_path = None

        data_limit = 10000
        data_limit_reached = False

        i = 0

        if aligned_path is not None and os.path.exists(external_aligned_path):
            for species in os.listdir(aligned_path):
                species_path = f"{aligned_path}{species}/"
                external_species_path = f"{external_aligned_path}{species}/"
                create_folder_(external_species_path)
                for data_name in os.listdir(species_path):
                    data_path = f"{species_path}{data_name}"
                    external_data_path = f"{external_species_path}{data_name}"
                    shutil.move(data_path, external_data_path)
                    i += 1
                    if i % 50 == 0:
                        print(f"Data: {i}")
                    if i > data_limit:
                        data_limit_reached = True
                        break
                if data_limit_reached:
                    break

    def delete(self):

        print("Deleting data...")

        data_limit = 500000
        data_limit_reached = False

        i = 0

        converted_images = set(read_data_from_file_(f"{self.subset_path}/converted.txt"))

        if os.path.exists(self.external_aligned_data_files_path):
            for species in os.listdir(self.aligned_data_files_path):
                species_path = f"{self.aligned_data_files_path}{species}/"
                external_species_path = f"{self.external_aligned_data_files_path}{species}/"
                create_folder_(external_species_path)
                for data_name in os.listdir(species_path):
                    data_id = f"{species}/{data_name.split('.')[0]}"
                    data_path = f"{species_path}{data_name}"
                    external_data_path = f"{external_species_path}{data_name}"
                    if data_id not in converted_images:
                        shutil.move(data_path, external_data_path)
                        i += 1
                        if i % 50 == 0:
                            print(f"Data: {i}")
                        if i > data_limit:
                            data_limit_reached = True
                            break
                if data_limit_reached:
                    break
        else:
            print("Please connect hard drive!")

    def extract_project_taxonomy(self, is_aligned):
        taxonomy = {}
        project_files_path = Taxonomy.aligned_project_files_path if is_aligned else Taxonomy.base_project_files_path
        species_tag = self.determine_scientific_name_tag(is_species=True, is_aligned=is_aligned)
        subspecies_tag = self.determine_scientific_name_tag(is_species=False, is_aligned=is_aligned)
        source_metadata = self.get_source_metadata_handle(is_aligned)
        source_observation_list = self.read_source_observation_list(source_metadata)
        i = 0
        for observation in source_observation_list:
            metadata = self.read_metadata(source_metadata, observation)
            species = metadata[species_tag]
            subspecies = metadata[subspecies_tag]
            taxonomy = Taxonomy.update_base_taxonomy(taxonomy, species, subspecies)
            if i % 10000 == 0:
                print(f"Read File: {i}")
            i += 1
        save_data_to_file_(f"{project_files_path}{self.subset_tag}.json", taxonomy)

    def align_source_metadata(self):
        project_species_tag = f"{self.project_tag}_species"
        project_subspecies_tag = f"{self.project_tag}_subspecies"
        project_group_tag = f"{self.project_tag}_group"
        uncertain_project_subspecies_tag = f"uncertain_{project_subspecies_tag}"
        uncertain_project_group_tag = f"uncertain_{project_group_tag}"
        species_tag = "species"
        subspecies_tag = "subspecies"
        group_tag = "group"
        uncertain_subspecies_tag = "uncertain_subspecies"
        uncertain_group_tag = "uncertain_group"
        source_metadata = self.get_source_metadata_handle(is_aligned=False)
        source_observation_list = self.read_source_observation_list(source_metadata)
        aligned_dataset = self.get_dataset_handle(is_aligned=True)
        aligned_source_metadata = self.get_source_metadata_handle(is_aligned=True)
        i = 0
        for observation in source_observation_list:
            metadata = self.read_metadata(source_metadata, observation)
            project_species = metadata[project_species_tag]
            project_subspecies = metadata[project_subspecies_tag]
            project_group = metadata[project_group_tag]
            project_scientific_name = self.determine_scientific_name(project_species, project_subspecies, project_group)
            species, subspecies, group = self.convert_scientific_name(project_scientific_name, is_aligned=True)
            degree_of_match = self.determine_taxonomy_match(species, is_species=True, is_aligned=True)
            if degree_of_match == 1:
                metadata[species_tag] = species
                metadata[subspecies_tag] = subspecies
                metadata[group_tag] = group
                if uncertain_project_subspecies_tag in metadata:
                    uncertain_project_subspecies = metadata[uncertain_project_subspecies_tag]
                    uncertain_project_group = metadata[uncertain_project_group_tag]
                    uncertain_project_scientific_name = self.determine_scientific_name(project_species, uncertain_project_subspecies, uncertain_project_group)
                    uncertain_subspecies, uncertain_group = self.convert_scientific_name(uncertain_project_scientific_name, is_aligned=True)[1:]
                    metadata[uncertain_subspecies_tag] = uncertain_subspecies
                    metadata[uncertain_group_tag] = uncertain_group
                metadata.pop(project_species_tag, None)
                metadata.pop(project_subspecies_tag, None)
                metadata.pop(project_group_tag, None)
                metadata.pop(uncertain_project_subspecies_tag, None)
                metadata.pop(uncertain_project_group_tag, None)
                self.write_metadata(aligned_source_metadata, metadata)
                self.update_observation_list(aligned_dataset, species, observation)
                if i % 10000 == 0:
                    print(f"Aligned File: {i}")
                i += 1

    # ************************************************ MINOR METHODS ***************************************************

    def is_year_valid(self, date_):
        if date_.split("-")[0] < self.minimum_year:
            valid = False
        elif date_.split("-")[0] > self.maximum_year:
            valid = False
        else:
            valid = True
        return valid

    def determine_scientific_name_tag(self, is_species, is_aligned):
        taxon = "species" if is_species else "subspecies"
        scientific_name_tag = taxon if is_aligned else f"{self.project_tag}_{taxon}"
        return scientific_name_tag

    # Determine the degree to which the scientific name matches the taxonomy
    def determine_taxonomy_match(self, scientific_name, is_species, is_aligned):
        taxonomy = self.aligned_taxonomy if is_aligned else self.base_taxonomy
        if is_species:
            degree_of_match = 1 if scientific_name in taxonomy else 0
        else:
            species = f"{scientific_name.split()[0]} {scientific_name.split()[1]}"
            if species in taxonomy:
                degree_of_match = 1 if scientific_name in taxonomy[species] else 0.5
            else:
                degree_of_match = 0
        return degree_of_match

    # Determine the synonym (if any) of the scientific name as present in the final taxonomy
    def convert_scientific_name(self, scientific_name, is_aligned):
        taxonomy_replacements = self.base_taxonomy_to_aligned_taxonomy_replacements if is_aligned else self.subset_taxonomy_to_base_taxonomy_replacements
        for group_no in range(0, len(taxonomy_replacements)):
            species, subspecies, group = self.separate_scientific_name(scientific_name)
            if species != "":
                scientific_name = self.determine_scientific_name(species, subspecies, group)
                scientific_name = self.determine_scientific_name_replacement(scientific_name, group_no=group_no, is_aligned=is_aligned)
        species, subspecies, group = self.separate_scientific_name(scientific_name)
        subspecies, group = self.recreate_scientific_denomination(subspecies, group, is_aligned=is_aligned)
        return species, subspecies, group

    # Determine the synonym (if any) of the scientific name as present in the selected taxonomy group
    def determine_scientific_name_replacement(self, scientific_name, group_no, is_aligned):
        taxonomy_replacements = self.base_taxonomy_to_aligned_taxonomy_replacements if is_aligned else self.subset_taxonomy_to_base_taxonomy_replacements
        taxonomy_replacements_group = taxonomy_replacements[group_no]
        if scientific_name != "":
            if scientific_name in taxonomy_replacements_group:
                scientific_name = taxonomy_replacements_group[scientific_name]
            else:
                species = f"{scientific_name.split()[0]} {scientific_name.split()[1]}"
                if species in taxonomy_replacements_group:
                    scientific_name = f"{taxonomy_replacements_group[species]} {' '.join(scientific_name.split()[2:])}"
        return scientific_name

    # Designate a subspecies as a group if it is not in the taxonomy
    def recreate_scientific_denomination(self, subspecies, group, is_aligned):
        if subspecies != "":
            degree_of_match = self.determine_taxonomy_match(subspecies, is_species=False, is_aligned=is_aligned)
            if degree_of_match == 0.5:
                group = subspecies
                subspecies = ""
        return subspecies, group

    @classmethod
    def determine_scientific_name(cls, species, subspecies, group):
        if subspecies != "":
            scientific_name = subspecies
        elif group != "":
            scientific_name = group
        else:
            scientific_name = species
        return scientific_name

    # Separate scientific name into species, subspecies and group
    @classmethod
    def separate_scientific_name(cls, scientific_name):
        scientific_name = " ".join(unidecode.unidecode(scientific_name).lower().title().strip().split())
        scientific_name = re.sub(r"(\s+[/]\s+)|([/]\s+)|(\s+[/])", "/", scientific_name)
        temp = scientific_name
        scientific_name = re.sub(r"[()\]\[]", "", scientific_name)
        scientific_name = " ".join(scientific_name.split())
        if re.search(r"^[a-z|A-Z]{2,}\s+[a-z|A-Z]{2,}\s+[a-z|A-Z]{2,}(\s*/\s*[a-z|A-Z]{2,})+$", scientific_name):
            species = f"{scientific_name.split()[0]} {scientific_name.split()[1]}"
            subspecies = ""
            group = f"{species} ({scientific_name.split(None, 2)[2]})"
        elif re.search(r"^[a-z|A-Z]{2,}\s+[a-z|A-Z]{2,}\s+[a-z|A-Z]{2,}(\s+X\s+[a-z|A-Z]{2,})+$", scientific_name):
            species = f"{scientific_name.split()[0]} {scientific_name.split()[1]}"
            subspecies = ""
            group = f"{species} ({scientific_name.split(None, 2)[2]})"
        elif re.search(r"^[a-z|A-Z]{2,}\s+[a-z|A-Z]{2,}\s+[a-z|A-Z]{2,}\s+Group$", scientific_name):
            species = f"{scientific_name.split()[0]} {scientific_name.split()[1]}"
            subspecies = ""
            group = f"{species} ({scientific_name.split(None, 2)[2]})"
        elif re.search(r"^[a-z|A-Z]{2,}\s+Domesticus\s+(Ssp(\s*\.)*\s|Ss(\s*\.)*\s)*Domesticus$", scientific_name):
            species = f"{scientific_name.split()[0]} {scientific_name.split()[1]}"
            subspecies = f"{species} {scientific_name.split()[-1]}"
            group = ""
        elif re.search(r"^[a-z|A-Z]{2,}\s+[a-z|A-Z]{2,}\s+("+"|".join(list(Subset.groups_to_replace.keys())) + r")$", scientific_name):
            species = f"{scientific_name.split()[0]} {scientific_name.split()[1]}"
            subspecies = ""
            group = f"{species} ({Subset.groups_to_replace[scientific_name.split(None, 2)[2]]})"
        elif re.search(r"^[a-z|A-Z]{2,}\s+[a-z|A-Z]{2,}\s+Type\s+[0-9]+(\s*,\s*Type\s+[0-9]+)*$", scientific_name):
            species = f"{scientific_name.split()[0]} {scientific_name.split()[1]}"
            subspecies = ""
            group = f"{species} ({scientific_name.split(None, 2)[2]})"
        elif re.search(r"^[a-z|A-Z]{2,}\s+[a-z|A-Z]{2,}\s+N[0-9]+(\s*,\s*N[0-9]+)*$", scientific_name):
            species = f"{scientific_name.split()[0]} {scientific_name.split()[1]}"
            subspecies = ""
            group = f"{species} ({scientific_name.split(None, 2)[2]})"
        else:
            species = ""
            subspecies = ""
            group = ""
        scientific_name = temp
        if species == "":
            if re.search(r"^[a-z|A-Z]{2,}\s+[a-z|A-Z]{2,}$", scientific_name):
                species = scientific_name
                subspecies = ""
                group = ""
            elif re.search(r"^[a-z|A-Z]{2,}\s+[a-z|A-Z]{2,}\s+(Ssp(\s*\.)*\s|Ss(\s*\.)*\s)*([a-z|A-Z]{2,}|V-Nigrum)$", scientific_name):
                species = f"{scientific_name.split()[0]} {scientific_name.split()[1]}"
                subspecies = f"{species} {scientific_name.split()[-1]}"
                group = ""
            else:
                species = ""
                subspecies = ""
                group = ""
        return species, subspecies, group

    @classmethod
    def separate_time(cls, time_):
        time_ = "".join(time_.split())
        if re.search(r"^[0-9]{3,4}$", time_, re.IGNORECASE):
            hour = time_[0: len(time_) - 2].zfill(2)
            minute = time_[len(time_) - 2: len(time_)]
        elif re.search(r"^[0-9]{1,2}[:.;',_h-][0-9]{2}([:.;',_-][0-9]{1,2})*(min|hrs|hs|h)*$", time_, re.IGNORECASE):
            time_parts = re.split(r"[:.;',_hH-]", time_)
            hour = time_parts[0].lstrip("0").zfill(2)
            minute = re.sub(r"min|hrs|hs|h", "", time_parts[1], flags=re.IGNORECASE)
            minute = minute.lstrip("0").zfill(2)
        elif re.search(r"^[0-9]{1,3}[:.;',_h-][0-9]{1,3}([:.;',_-][0-9]{1,2})*(min|hrs|hs|h)*(a.m.|am.|am|a)*$", time_, re.IGNORECASE):
            time_ = re.sub(r"a.m.|am.|am|a", "", time_, flags=re.IGNORECASE)
            time_parts = re.split(r"[:.;',_hH-]", time_)
            hour = time_parts[0].lstrip("0").zfill(2)
            minute = time_parts[1].lstrip("0").zfill(2)
            if hour == "12": hour = "00"
        elif re.search(r"^[0-9]{1,3}[:.;',_h-][0-9]{1,3}([:.;',_-][0-9]{1,2})*(min|hrs|hs|h)*(p.m.|pm.|pm|p)*$", time_, re.IGNORECASE):
            time_ = re.sub(r"p.m.|pm.|pm|p", "", time_, flags=re.IGNORECASE)
            time_parts = re.split(r"[:.;',_hH-]", time_)
            hour = time_parts[0].lstrip("0").zfill(2)
            if "01" <= hour <= "11": hour = str(int(hour) + 12).zfill(2)
            minute = time_parts[1].lstrip("0").zfill(2)
        elif re.search(r"^[0-9]{2}:[0-9]{2}:[0-9]{2}Z$", time_, re.IGNORECASE):
            hour = time_.split(":")[0]
            minute = time_.split(":")[1]
        else:
            hour = ""
            minute = ""
        if hour != "" and minute != "":
            if int(hour) < 0 or int(hour) > 23 or int(minute) < 0 or int(minute) > 59:
                hour = ""
                minute = ""
        return hour, minute

    @classmethod
    def convert_date_and_time(cls, time_, date_, latitude, longitude, is_time_standard):
        timezone_name = cls.timezone_finder.timezone_at(lat=float(latitude), lng=float(longitude))
        utc_offset = pytz.timezone(timezone_name).localize(datetime.strptime(date_, "%Y-%m-%d")).strftime("%z")
        is_offset_positive = True if utc_offset[0] == "+" else False
        utc_hour_offset = int(utc_offset[1:3])
        utc_minute_offset = int(utc_offset[3:5])
        detailed_time = datetime.strptime(f"{date_} {time_}", "%Y-%m-%d %H:%M")
        if is_time_standard ^ is_offset_positive:
            new_detailed_time = detailed_time - timedelta(hours=utc_hour_offset, minutes=utc_minute_offset)
        else:
            new_detailed_time = detailed_time + timedelta(hours=utc_hour_offset, minutes=utc_minute_offset)
        new_date = new_detailed_time.strftime("%Y-%m-%d")
        new_time = new_detailed_time.strftime("%H:%M")
        return new_date, new_time

    @classmethod
    def determine_location_bin(cls, latitude, longitude, bin_size, bin_start_offset):

        bin_count = (180 / bin_size[0], 360 / bin_size[1])
        bin_positive_offset = (90, 180)

        def determine_coordinate_bin_(coordinate, coordinate_i):
            bin_i_ = math.floor((coordinate + bin_positive_offset[coordinate_i] - bin_start_offset[coordinate_i]) / bin_size[coordinate_i])
            if bin_i_ < 0: bin_i_ = bin_i_ + bin_count[coordinate_i]
            return bin_i_

        bin_i = (determine_coordinate_bin_(latitude, 0), determine_coordinate_bin_(longitude, 0))
        bin_i = int(bin_i[0] * bin_count[0] + bin_i[1])

        return bin_i

    @classmethod
    def determine_date_bin(cls, date_, bin_size=6, bin_start_offset=3):
        date_ = date_.split("-", 1)[-1]
        bin_count = 366 / bin_size
        date_ = datetime.strptime(f"2020-{date_}", "%Y-%m-%d").date()
        day_of_year = date_.timetuple().tm_yday - 1
        bin_i = math.floor((day_of_year - bin_start_offset) / bin_size)
        if bin_i < 0: bin_i = bin_i + bin_count
        return int(bin_i)

    @classmethod
    def determine_time_bin(cls, time_, bin_size=30, bin_start_offset=60):
        bin_count = 1440 / bin_size
        time_ = datetime.strptime(time_, "%H:%M")
        minute_of_day = time_.hour * 60 + time_.minute
        bin_i = math.floor((minute_of_day - bin_start_offset) / bin_size)
        if bin_i < 0: bin_i = bin_i + bin_count
        return int(bin_i)

    @classmethod
    def simplify_type(cls, sex_attribute_composition, age_attribute_composition, vocalisation_attribute_composition, attribute_count):
        sexes = []
        ages = []
        general_vocalisations = []
        specific_vocalisations = []
        individuals = []
        for i in range(0, attribute_count):
            sexes_i = ["Unknown"] if i not in sex_attribute_composition else sex_attribute_composition[i]
            ages_i = ["Unknown"] if i not in age_attribute_composition else age_attribute_composition[i]
            vocalisations_i = [("Unknown", "Unknown")] if i not in vocalisation_attribute_composition else vocalisation_attribute_composition[i]
            for j in range(0, len(sexes_i)):
                for k in range(0, len(ages_i)):
                    for m in range(0, len(vocalisations_i)):
                        sex = sexes_i[j]
                        age = ages_i[k]
                        general_vocalisation = vocalisations_i[m][0]
                        specific_vocalisation = vocalisations_i[m][1]
                        individual = [sex, age, general_vocalisation, specific_vocalisation]
                        if individual not in individuals:
                            sexes.append(sex)
                            ages.append(age)
                            general_vocalisations.append(general_vocalisation)
                            specific_vocalisations.append(specific_vocalisation)
                            individuals.append(individual)
        individuals = []
        temp_sexes = [x for x in sexes if x != "Unknown"]
        if len(temp_sexes) <= 1:
            if not temp_sexes:
                temp_sexes = ["Unknown"]
            temp_ages = [x for x in ages if x != "Unknown"]
            if len(temp_ages) <= 1:
                if not temp_ages:
                    temp_ages = ["Unknown"]
                temp_vocalisations = [x for x in list(zip(general_vocalisations, specific_vocalisations)) if x != ("Unknown", "Unknown")]
                if not temp_vocalisations:
                    temp_vocalisations = [("Unknown", "Unknown")]
                sexes = []
                ages = []
                general_vocalisations = []
                specific_vocalisations = []
                for j in range(0, len(temp_sexes)):
                    for k in range(0, len(temp_ages)):
                        for m in range(0, len(temp_vocalisations)):
                            sex = temp_sexes[j]
                            age = temp_ages[k]
                            general_vocalisation = temp_vocalisations[m][0]
                            specific_vocalisation = temp_vocalisations[m][1]
                            individual = [sex, age, general_vocalisation, specific_vocalisation]
                            if individual not in individuals:
                                sexes.append(sex)
                                ages.append(age)
                                general_vocalisations.append(general_vocalisation)
                                specific_vocalisations.append(specific_vocalisation)
                                individuals.append(individual)
        return sexes, ages, general_vocalisations, specific_vocalisations

    # ************************************************ READ METHODS ***************************************************

    @staticmethod
    def read_species_list(dataset):
        species_list = [species["_id"] for species in dataset.find()]
        return species_list

    @staticmethod
    def read_observation_list(dataset, species):
        observation_list = dataset.find({"_id": species})[0]["observation_ids"]
        return observation_list

    @staticmethod
    def read_source_observation_list(source_metadata):
        observation_list = [observation["_id"] for observation in source_metadata.find()]
        return observation_list

    def read_attribute_list(self, dataset, source_metadata, species, attribute_name):
        observation_list = self.read_observation_list(dataset, species)
        source_attribute_list = [observation[attribute_name] for observation in source_metadata.find({"_id": observation_list})]
        return source_attribute_list

    def read_multi_attribute_list(self, dataset, source_metadata, species, attribute_name_list):
        observation_list = self.read_observation_list(dataset, species)
        source_attribute_list = [tuple([observation[attribute_name] for attribute_name in attribute_name_list]) for observation in source_metadata.find({"_id": observation_list})]
        return source_attribute_list

    @staticmethod
    def read_source_attribute_list(source_metadata, attribute_name):
        source_attribute_list = [observation[attribute_name] for observation in source_metadata.find()]
        return source_attribute_list

    @staticmethod
    def read_source_multi_attribute_list(source_metadata, attribute_name_list):
        source_attribute_list = [tuple([observation[attribute_name] for attribute_name in attribute_name_list]) for observation in source_metadata.find()]
        return source_attribute_list

    @staticmethod
    def read_metadata(source_metadata, observation):
        metadata = source_metadata.find({"_id": observation})[0]
        return metadata

    def read_metadata_list(self, dataset, source_metadata, species):
        observation_list = self.read_observation_list(dataset, species)
        metadata_list = [_ for _ in source_metadata.find({"_id": {"$in": observation_list}})]
        return metadata_list

    # ************************************************ WRITE METHODS ***************************************************

    @staticmethod
    def write_metadata(source_metadata, metadata):
        source_metadata.insert_one(metadata)

    @staticmethod
    def write_checklist(source_checklist, checklist):
        source_checklist.insert_one(checklist)

    # ************************************************ UPDATE METHODS ***************************************************

    @staticmethod
    def update_observation_list(dataset, species, observation):
        dataset.find_one_and_update({"_id": species}, {"$push": {"observation_ids": observation}}, upsert=True)

    # ************************************************ GETTER METHODS **************************************************

    def get_database_handle(self, is_aligned):
        client = MongoClient()
        database_name = self.aligned_metadata_database_name if is_aligned else self.metadata_database_name
        database = client[database_name]
        return database

    def get_dataset_handle(self, is_aligned):
        database = self.get_database_handle(is_aligned)
        dataset = database.species
        return dataset

    def get_source_metadata_handle(self, is_aligned):
        database = self.get_database_handle(is_aligned)
        source_metadata = database.observation
        return source_metadata

    def get_source_checklist_handle(self, is_aligned):
        database = self.get_database_handle(is_aligned)
        source_checklist = database.checklist
        return source_checklist

    # ******************************************************************************************************************


from taxonomy_generators.taxonomy import Taxonomy
