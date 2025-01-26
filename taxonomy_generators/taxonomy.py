import os.path

from selenium import webdriver
from bs4 import BeautifulSoup
import time
from utility import *
import hashlib


class Taxonomy:

    # Paths
    taxonomy_path = "C:/Users/sanam/Documents/Masters/Resources/Taxonomy"
    downloaded_files_path = f"{taxonomy_path}/Downloaded/"
    base_files_path = f"{taxonomy_path}/Base/"
    downloaded_project_files_path = f"{taxonomy_path}/Downloaded Project/"
    base_project_files_path = f"{taxonomy_path}/Base Project/"
    aligned_project_files_path = f"{taxonomy_path}/Aligned Project/"
    removal_files_path = f"{taxonomy_path}/Removal/"
    replacement_files_path = f"{taxonomy_path}/Replacement/"
    aligned_files_path = f"{taxonomy_path}/Aligned/"

    # ************************************************ MAJOR METHODS ***************************************************

    def create_base(self):
        for downloaded_file in os.listdir(self.downloaded_files_path):
            taxonomy_name = downloaded_file.split(".")[0]
            if taxonomy_name == "xenocanto":
                taxonomy = self.create_base_from_xenocanto_formatted()
            elif taxonomy_name == "observationorg":
                taxonomy = self.create_base_from_observationorg_formatted()
            elif taxonomy_name == "inaturalist":
                taxonomy = None
            elif taxonomy_name == "clements2021":
                taxonomy = self.create_base_from_ebird_formatted(taxonomy_name)
            else:
                taxonomy = self.create_base_from_ioc_formatted(taxonomy_name)
            if taxonomy is not None: save_data_to_file_(f"{self.base_files_path}{taxonomy_name}.json", taxonomy)

    def create_base_from_ebird_formatted(self, taxonomy_name):
        taxonomy = {}
        with open(f"{self.downloaded_files_path}{taxonomy_name}.csv", 'r', encoding="ANSI") as taxonomy_file:
            reader = csv.reader((line.replace('\0', '') for line in taxonomy_file), delimiter=',')
            next(reader)
            next(reader)
            for row in reader:
                taxon_category = row[3]
                scientific_name = row[5]
                if taxon_category == "species" or taxon_category == "subspecies" or taxon_category == "group (monotypic)" or taxon_category == "group (polytypic)":
                    if scientific_name != "":
                        species, subspecies = Subset.separate_scientific_name(scientific_name)[0:2]
                        taxonomy = Taxonomy.update_base_taxonomy(taxonomy, species, subspecies)
        return taxonomy

    def create_base_from_ioc_formatted(self, taxonomy_name):
        taxonomy = {}
        with open(f"{self.downloaded_files_path}{taxonomy_name}.csv", 'r', encoding="UTF-8") as taxonomy_file:
            reader = csv.reader((line.replace('\0', '') for line in taxonomy_file), delimiter=';')
            next(reader)
            for row in reader:
                scientific_name = row[1] if "ioc" in taxonomy_name else row[5]
                if scientific_name != "":
                    species, subspecies = Subset.separate_scientific_name(scientific_name)[0:2]
                    taxonomy = Taxonomy.update_base_taxonomy(taxonomy, species, subspecies)
        return taxonomy

    def create_base_from_xenocanto_formatted(self):
        taxonomy = {}
        with open(f"{self.downloaded_files_path}xenocanto.csv", 'r') as taxonomy_file:
            reader = csv.reader((line.replace('\0', '') for line in taxonomy_file), delimiter=',')
            next(reader)
            for row in reader:
                scientific_name = row[0]
                # status = row[2]
                if scientific_name != "":
                    species, subspecies = Subset.separate_scientific_name(scientific_name)[0:2]
                    taxonomy = Taxonomy.update_base_taxonomy(taxonomy, species, subspecies)
        return taxonomy

    def create_base_from_observationorg_formatted(self):
        taxonomy = {}
        with open(f"{self.downloaded_files_path}observationorg.csv", 'r', encoding="UTF-8") as taxonomy_file:
            reader = csv.reader((line.replace('\0', '') for line in taxonomy_file), delimiter=';')
            next(reader)
            for row in reader:
                scientific_name = row[2]
                taxon_rank = row[8]
                if taxon_rank == "species" or taxon_rank == "subspecies":
                    if scientific_name != "":
                        species, subspecies = Subset.separate_scientific_name(scientific_name)[0:2]
                        taxonomy = Taxonomy.update_base_taxonomy(taxonomy, species, subspecies)
        return taxonomy

    def create_comparison(self):
        driver = webdriver.Chrome("C:/Users/sanam/Documents/Installers/chrome_driver/chromedriver.exe")
        base_url = "https://avibase.bsc-eoc.org/compare.jsp?[?]&continent=&reg_type=9"
        taxonomy_url_parts = {"clements2021": "source[?]=clements&version[?]=CLEMENTS2021",
                              "clements2019": "source[?]=clements&version[?]=CLEMENTS6_19",
                              "clements2018": "source[?]=clements&version[?]=CLEMENTS6_18",
                              "clements2017": "source[?]=clements&version[?]=CLEMENTS6_17",
                              "clements2016": "source[?]=clements&version[?]=CLEMENTS6_16",
                              "clements2015": "source[?]=clements&version[?]=CLEMENTS6_15",
                              "clements2014": "source[?]=clements&version[?]=CLEMENTS6_14",
                              "clements2013": "source[?]=clements&version[?]=CLEMENTS6_13",
                              "clements2012": "source[?]=clements&version[?]=CLEMENTS6_12",
                              "clements2011": "source[?]=clements&version[?]=CLEMENTS6_17",
                              "clements2010": "source[?]=clements&version[?]=CLEMENTS6_10",
                              "clements2009": "source[?]=clements&version[?]=CLEMENTS6_09",
                              "clements2008": "source[?]=clements&version[?]=CLEMENTS6_08",
                              "clements2007": "source[?]=clements&version[?]=CLEMENTS6_07",
                              "clements2006": "source[?]=clements&version[?]=CLEMENTS6",
                              "ioc112": "source[?]=ioc&version[?]=IOC11_02",
                              "ioc111": "source[?]=ioc&version[?]=IOC11_01",
                              "ioc102": "source[?]=ioc&version[?]=IOC10_02",
                              "ioc101": "source[?]=ioc&version[?]=IOC10_01",
                              "ioc92": "source[?]=ioc&version[?]=IOC09_02",
                              "ioc91": "source[?]=ioc&version[?]=IOC09_01",
                              "ioc82": "source[?]=ioc&version[?]=IOC08_02",
                              "ioc81": "source[?]=ioc&version[?]=IOC08_01",
                              "ioc73": "source[?]=ioc&version[?]=IOC07_03",
                              "ioc72": "source[?]=ioc&version[?]=IOC07_02",
                              "ioc71": "source[?]=ioc&version[?]=IOC07_01",
                              "ioc64": "source[?]=ioc&version[?]=IOC06_04",
                              "ioc63": "source[?]=ioc&version[?]=IOC06_03",
                              "ioc62": "source[?]=ioc&version[?]=IOC06_02",
                              "ioc61": "source[?]=ioc&version[?]=IOC06_01",
                              "ioc54": "source[?]=ioc&version[?]=IOC05_04",
                              "ioc53": "source[?]=ioc&version[?]=IOC05_03",
                              "ioc52": "source[?]=ioc&version[?]=IOC05_02",
                              "ioc51": "source[?]=ioc&version[?]=IOC05_01",
                              "ioc44": "source[?]=ioc&version[?]=IOC04_04",
                              "ioc43": "source[?]=ioc&version[?]=IOC04_03",
                              "ioc42": "source[?]=ioc&version[?]=IOC04_02",
                              "ioc41": "source[?]=ioc&version[?]=IOC04_01",
                              "ioc35": "source[?]=ioc&version[?]=IOC03_05",
                              "ioc34": "source[?]=ioc&version[?]=IOC03_04",
                              "ioc33": "source[?]=ioc&version[?]=IOC03_03",
                              "ioc32": "source[?]=ioc&version[?]=IOC03_02",
                              "ioc31": "source[?]=ioc&version[?]=IOC03_01",
                              "ioc211": "source[?]=ioc&version[?]=IOC02_11",
                              "ioc210": "source[?]=ioc&version[?]=IOC02_10",
                              "ioc29": "source[?]=ioc&version[?]=IOC02_09",
                              "ioc28": "source[?]=ioc&version[?]=IOC02_08",
                              "ioc27": "source[?]=ioc&version[?]=IOC02_07",
                              "ioc26": "source[?]=ioc&version[?]=IOC02_06",
                              "ioc25": "source[?]=ioc&version[?]=IOC02_05",
                              "ioc24": "source[?]=ioc&version[?]=IOC02_04",
                              "ioc23": "source[?]=ioc&version[?]=IOC02_03",
                              "ioc22": "source[?]=ioc&version[?]=IOC02_02",
                              "ioc21": "source[?]=ioc&version[?]=IOC02_01",
                              "ioc17": "source[?]=ioc&version[?]=IOC01_07",
                              "ioc16": "source[?]=ioc&version[?]=IOC01_06",
                              "ioc15": "source[?]=ioc&version[?]=IOC01_05",
                              "ioc14": "source[?]=ioc&version[?]=IOC01_04",
                              "ioc13": "source[?]=ioc&version[?]=IOC01_03",
                              "ioc12": "source[?]=ioc&version[?]=IOC01_02",
                              "ioc11": "source[?]=ioc&version[?]=IOC01_01"}
        for x in taxonomy_url_parts.keys():
            for y in taxonomy_url_parts.keys():
                if x != y:
                    taxonomy_1_to_2_replacement_file_path = f"{self.replacement_files_path}{x}_to_{y}.json"
                    taxonomy_2_to_1_replacement_file_path = f"{self.replacement_files_path}{y}_to_{x}.json"
                    taxonomy_1_with_2_removal_file_path = f"{self.removal_files_path}{x}_with_{y}.txt"
                    taxonomy_2_with_1_removal_file_path = f"{self.removal_files_path}{y}_with_{x}.txt"
                    if not os.path.exists(taxonomy_1_to_2_replacement_file_path) or not os.path.exists(taxonomy_2_to_1_replacement_file_path) or not os.path.exists(taxonomy_1_with_2_removal_file_path) or not os.path.exists(taxonomy_2_with_1_removal_file_path):
                        print(f"{x} : {y}")
                        taxonomy_1_to_2_replacements = {}
                        taxonomy_2_to_1_replacements = {}
                        taxonomy_1_with_2_removals = set()
                        taxonomy_2_with_1_removals = set()
                        url = base_url.replace("[?]", f"{taxonomy_url_parts[x].replace('[?]', '1')}&{taxonomy_url_parts[y].replace('[?]', '2')}")
                        driver.get(url)
                        time.sleep(15)
                        html_structure = BeautifulSoup(driver.page_source, 'html.parser')
                        table_structure = html_structure.tbody
                        rows = table_structure.find_all("tr")
                        for i in range(1, len(rows)):
                            row = rows[i]
                            cells = row.find_all("td")
                            taxonomy_1_scientific_name = "" if cells[0].string is None else cells[0].string.lower().title().strip()
                            taxonomy_2_scientific_name = "" if cells[4].string is None else cells[4].string.lower().title().strip()
                            taxonomy_1_scientific_name_class = cells[0]["class"]
                            taxonomy_2_scientific_name_class = cells[4]["class"]
                            if "changed" in taxonomy_1_scientific_name_class and "changed" in taxonomy_2_scientific_name_class:
                                taxonomy_1_to_2_replacements[taxonomy_1_scientific_name] = taxonomy_2_scientific_name
                                taxonomy_2_to_1_replacements[taxonomy_2_scientific_name] = taxonomy_1_scientific_name
                            if "removed" in taxonomy_1_scientific_name_class:
                                taxonomy_1_with_2_removals.add(taxonomy_1_scientific_name)
                            if "added" in taxonomy_2_scientific_name_class:
                                taxonomy_2_with_1_removals.add(taxonomy_2_scientific_name)
                        save_data_to_file_(f"{self.replacement_files_path}{x}_to_{y}.json", taxonomy_1_to_2_replacements)
                        save_data_to_file_(f"{self.replacement_files_path}{y}_to_{x}.json", taxonomy_2_to_1_replacements)
                        save_data_to_file_(f"{self.removal_files_path}{x}_with_{y}.txt", list(taxonomy_1_with_2_removals))
                        save_data_to_file_(f"{self.removal_files_path}{y}_with_{x}.txt", list(taxonomy_2_with_1_removals))

    # Returns a single main taxonomy that is aligned to several secondary taxonomies
    # This means that the returned taxonomy is the main taxonomy except the species that are in conflict between all the taxonomies are removed from it
    @classmethod
    def create_aligned(cls, main_taxonomy_name, secondary_taxonomy_name_list):
        main_taxonomy = read_data_from_file_(f"{cls.base_files_path}{main_taxonomy_name}.json")
        main_taxonomy_removals = set()
        for taxonomy_name in secondary_taxonomy_name_list:
            main_taxonomy_with_secondary_taxonomy_removal_file_path = f"{cls.removal_files_path}{main_taxonomy_name}_with_{taxonomy_name}.txt"
            main_taxonomy_with_secondary_taxonomy_removals = read_data_from_file_(main_taxonomy_with_secondary_taxonomy_removal_file_path)
            main_taxonomy_removals.update(main_taxonomy_with_secondary_taxonomy_removals)
        for taxonomy_name_1 in secondary_taxonomy_name_list:
            taxonomy_1_to_main_taxonomy_replacement_file_path = f"{cls.replacement_files_path}{taxonomy_name_1}_to_{main_taxonomy_name}.json"
            if os.path.exists(taxonomy_1_to_main_taxonomy_replacement_file_path):
                taxonomy_1_to_main_taxonomy_replacements = read_data_from_file_(taxonomy_1_to_main_taxonomy_replacement_file_path)
                for taxonomy_name_2 in secondary_taxonomy_name_list:
                    if taxonomy_name_1 != taxonomy_name_2:
                        taxonomy_1_with_2_removal_file_path = f"{cls.removal_files_path}{taxonomy_name_1}_with_{taxonomy_name_2}.txt"
                        if os.path.exists(taxonomy_1_with_2_removal_file_path):
                            taxonomy_1_with_2_removals = read_data_from_file_(taxonomy_1_with_2_removal_file_path)
                            for taxonomy_1_scientific_name in taxonomy_1_with_2_removals:
                                taxonomy_1_scientific_name = taxonomy_1_scientific_name if taxonomy_1_scientific_name not in taxonomy_1_to_main_taxonomy_replacements else taxonomy_1_to_main_taxonomy_replacements[taxonomy_1_scientific_name]
                                main_taxonomy_removals.add(taxonomy_1_scientific_name)
        aligned_taxonomy = {}
        for main_taxonomy_scientific_name in main_taxonomy.keys():
            if main_taxonomy_scientific_name not in main_taxonomy_removals:
                aligned_taxonomy[main_taxonomy_scientific_name] = main_taxonomy[main_taxonomy_scientific_name]
        aligned_taxonomy_code = hashlib.sha256("_".join(secondary_taxonomy_name_list).encode("utf-8")).hexdigest()
        aligned_taxonomy_file_path = f"{cls.aligned_files_path}{main_taxonomy_name}_wrt_{aligned_taxonomy_code}.json"
        save_data_to_file_(aligned_taxonomy_file_path, aligned_taxonomy)

    @classmethod
    def merge_taxonomies(cls, taxonomy_name_list):
        taxonomy_list = []
        for taxonomy_name in taxonomy_name_list:
            taxonomy_list.append(read_data_from_file_(f"{cls.base_files_path}{taxonomy_name}.json"))
        merged_taxonomy = merge_list_dictionaries_(taxonomy_list)
        return merged_taxonomy

    @classmethod
    def merge_taxonomy_replacements(cls, taxonomy_replacements_name_list):
        merged_taxonomy_replacements = []
        for taxonomy_replacements_group_name_dictionary in taxonomy_replacements_name_list:
            taxonomy_replacements_group = []
            for taxonomy_replacements_name_1 in taxonomy_replacements_group_name_dictionary:
                taxonomy_replacements_name_2 = taxonomy_replacements_group_name_dictionary[taxonomy_replacements_name_1]
                taxonomy_replacements = read_data_from_file_(f"{cls.replacement_files_path}{taxonomy_replacements_name_1}_to_{taxonomy_replacements_name_2}.json")
                taxonomy_replacements_group.append(taxonomy_replacements)
            merged_taxonomy_replacements.append(merge_list_dictionaries_(taxonomy_replacements_group))
        return merged_taxonomy_replacements

    @classmethod
    def compare_project_to_source(cls, project_taxonomy_name, base_taxonomy_name, secondary_taxonomy_name_list, project_files_path):
        project_taxonomy = cls.get_project(project_taxonomy_name, project_files_path)
        base_taxonomy_conflicts = cls.determine_taxonomy_conflicts(project_taxonomy, cls.get_base(base_taxonomy_name))[0]
        secondary_taxonomy_correspondences_list = cls.determine_secondary_taxonomy_correspondences(project_taxonomy, base_taxonomy_name, secondary_taxonomy_name_list)
        resolved_conflicts, unresolved_conflicts = cls.compare_conflicts_to_correspondences(base_taxonomy_conflicts, secondary_taxonomy_correspondences_list)
        print(f"Number of {base_taxonomy_name} Species Conflicts: {len(base_taxonomy_conflicts)}")
        print(base_taxonomy_conflicts)
        print("\n")
        for i in range(0, len(secondary_taxonomy_name_list)):
            secondary_taxonomy_name = secondary_taxonomy_name_list[i]
            secondary_taxonomy_conflicts = secondary_taxonomy_correspondences_list[i]
            if len(secondary_taxonomy_conflicts) != 0:
                print(f"Number of {secondary_taxonomy_name} Species Correspondences: {len(secondary_taxonomy_conflicts)}")
                print(secondary_taxonomy_conflicts)
        print("\n")
        print(f"Resolved Conflicts: {len(resolved_conflicts)}")
        print(resolved_conflicts)
        print(f"Unresolved Conflicts: {len(unresolved_conflicts)}")
        print(unresolved_conflicts)

    # ************************************************ MINOR METHODS ***************************************************

    @classmethod
    def update_base_taxonomy(cls, taxonomy, species, subspecies):
        if species not in taxonomy:
            taxonomy[species] = []
        if subspecies != "":
            if subspecies not in taxonomy[species]:
                taxonomy[species].append(subspecies)
                taxonomy[species].sort()
        return taxonomy

    @classmethod
    def determine_taxonomy_conflicts(cls, taxonomy_1, taxonomy_2):
        species_taxonomy_conflicts = set()
        subspecies_taxonomy_conflicts = set()
        for species in taxonomy_1:
            if species not in taxonomy_2:
                species_taxonomy_conflicts.add(species)
            for subspecies in taxonomy_1[species]:
                if subspecies not in taxonomy_2:
                    subspecies_taxonomy_conflicts.add(subspecies)
        return species_taxonomy_conflicts, subspecies_taxonomy_conflicts

    @classmethod
    def determine_secondary_taxonomy_correspondences(cls, project_taxonomy, base_taxonomy_name, secondary_taxonomy_name_list):
        secondary_taxonomy_correspondences_list = []
        base_taxonomy = cls.get_base(base_taxonomy_name)
        for i in range(0, len(secondary_taxonomy_name_list)):
            secondary_taxonomy_name = secondary_taxonomy_name_list[i]
            secondary_taxonomy_replacements = cls.get_replacement(secondary_taxonomy_name, base_taxonomy_name)
            secondary_taxonomy_removals = cls.get_removal(secondary_taxonomy_name, base_taxonomy_name)
            secondary_taxonomy = set(secondary_taxonomy_replacements.keys()).union(secondary_taxonomy_removals)
            secondary_taxonomy_correspondences = set()
            for species in project_taxonomy:
                if species in secondary_taxonomy and species not in base_taxonomy:
                    secondary_taxonomy_correspondences.add(species)
            for j in range(0, i):
                previous_secondary_taxonomy_conflicts = secondary_taxonomy_correspondences_list[j]
                secondary_taxonomy_conflicts_intersection = secondary_taxonomy_correspondences.intersection(previous_secondary_taxonomy_conflicts)
                secondary_taxonomy_correspondences.difference_update(secondary_taxonomy_conflicts_intersection)
            secondary_taxonomy_correspondences_list.append(secondary_taxonomy_correspondences)
        return secondary_taxonomy_correspondences_list

    @classmethod
    def compare_conflicts_to_correspondences(cls, base_taxonomy_conflicts, secondary_taxonomy_correspondences_list):
        resolved_conflicts = set()
        for secondary_taxonomy_correspondences in secondary_taxonomy_correspondences_list:
            for species in secondary_taxonomy_correspondences:
                if species in base_taxonomy_conflicts:
                    resolved_conflicts.add(species)
        unresolved_conflicts = base_taxonomy_conflicts - resolved_conflicts
        return resolved_conflicts, unresolved_conflicts

    # ************************************************ GETTER METHODS **************************************************

    @classmethod
    def get_base(cls, taxonomy_name):
        if os.path.exists(f"{Taxonomy.base_files_path}{taxonomy_name}.json"):
            taxonomy = read_data_from_file_(f"{Taxonomy.base_files_path}{taxonomy_name}.json")
        else:
            taxonomy = {}
        return taxonomy

    @classmethod
    def get_project(cls, taxonomy_name, project_files_path):
        if os.path.exists(f"{project_files_path}{taxonomy_name}.json"):
            taxonomy = read_data_from_file_(f"{project_files_path}{taxonomy_name}.json")
        else:
            taxonomy = {}
        return taxonomy

    @classmethod
    def get_replacement(cls, base_taxonomy_name, reference_taxonomy_name):
        base_taxonomy_to_reference_taxonomy_replacement_file_path = f"{cls.replacement_files_path}{base_taxonomy_name}_to_{reference_taxonomy_name}.json"
        if os.path.exists(base_taxonomy_to_reference_taxonomy_replacement_file_path):
            base_taxonomy_to_reference_taxonomy_replacements = read_data_from_file_(base_taxonomy_to_reference_taxonomy_replacement_file_path)
        else:
            base_taxonomy_to_reference_taxonomy_replacements = {}
        return base_taxonomy_to_reference_taxonomy_replacements

    @classmethod
    def get_removal(cls, base_taxonomy_name, reference_taxonomy_name):
        base_taxonomy_with_reference_taxonomy_replacement_file_path = f"{cls.removal_files_path}{base_taxonomy_name}_with_{reference_taxonomy_name}.txt"
        if os.path.exists(base_taxonomy_with_reference_taxonomy_replacement_file_path):
            base_taxonomy_with_reference_taxonomy_replacements = set(read_data_from_file_(base_taxonomy_with_reference_taxonomy_replacement_file_path))
        else:
            base_taxonomy_with_reference_taxonomy_replacements = []
        return base_taxonomy_with_reference_taxonomy_replacements

    @classmethod
    def get_aligned(cls, main_taxonomy_name, secondary_taxonomy_name_list):
        aligned_taxonomy_code = hashlib.sha256("_".join(secondary_taxonomy_name_list).encode("utf-8")).hexdigest()
        aligned_taxonomy_file_path = f"{cls.aligned_files_path}{main_taxonomy_name}_wrt_{aligned_taxonomy_code}.json"
        # if not os.path.exists(aligned_taxonomy_file_path):
        #     cls.create_aligned(main_taxonomy_name, secondary_taxonomy_name_list)
        # taxonomy = read_data_from_file_(aligned_taxonomy_file_path)
        taxonomy = {}
        return taxonomy

    # ******************************************************************************************************************


from subset_generators.subset import Subset
