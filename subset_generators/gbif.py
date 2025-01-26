import unidecode
import re
from subset_generators.subset import Subset
from utility import *
from taxonomy_generators.taxonomy import Taxonomy
from dataset_generators.constants import *
from pymongo import MongoClient
import pprint


class GBIF(Subset):

    # ************************************************ CONSTRUCTOR *****************************************************

    # Arrays

    observation_issue_indicators = read_data_from_file_(f"{RESOURCE_PATH}arrays/observation_issue_indicators.txt")

    def __init__(self, subset_path, subset_tag, project_tag,
                 base_taxonomy, subset_taxonomy_to_base_taxonomy_replacements, aligned_taxonomy, base_taxonomy_to_aligned_taxonomy_replacements):

        # Tags
        parent_project_tag = "gbif"

        # Years
        minimum_year = "2009" if project_tag == "observationorg" else ("2006" if subset_tag == "inaturalist_image" else "2014")
        maximum_year = "2019" if project_tag == "observationorg" else "2021"

        # Parent Constructor
        super(GBIF, self).__init__(subset_path, subset_tag, project_tag, parent_project_tag,
                                   minimum_year, maximum_year,
                                   base_taxonomy, subset_taxonomy_to_base_taxonomy_replacements, aligned_taxonomy, base_taxonomy_to_aligned_taxonomy_replacements)

        # Paths
        self.occurence_file_path = f"{self.downloaded_metadata_files_path}occurrence.txt"
        self.multimedia_file_path = f"{self.downloaded_metadata_files_path}multimedia.txt"

    # ************************************************ MAJOR METHODS ***************************************************

    def extract_downloaded_project_taxonomy(self):
        taxonomy = {}
        i = 0
        with open(self.occurence_file_path, "r", encoding="UTF-8") as occurrence_file:
            for line in occurrence_file:
                if i > 0:
                    keys = line.split("\t")
                    project_scientific_name = keys[241]
                    project_species, project_subspecies = self.separate_scientific_name(project_scientific_name)[0:2]
                    if project_species != "": taxonomy = Taxonomy.update_base_taxonomy(taxonomy, project_species, project_subspecies)
                    if i % 10000 == 0:
                        print(f"Read File: {i}")
                i += 1
        save_data_to_file_(f"{Taxonomy.downloaded_project_files_path}{self.subset_tag}.json", taxonomy)

    def extract_metadata_structure(self):
        with open(self.occurence_file_path, "r", encoding="UTF-8") as occurrence_file:
            rows = [next(occurrence_file).split("\t") for _ in range(2)]
            for i in range(0, len(rows[0])):
                print(f"{i} : {rows[0][i]} : {rows[1][i]}")

    def edit_source_metadata(self):
        dataset = self.get_dataset_handle(is_aligned=False)
        source_metadata = self.get_source_metadata_handle(is_aligned=False)
        i = 0
        j = 0
        k = 0
        with open(self.occurence_file_path, "r", encoding="UTF-8") as occurrence_file:
            for line in occurrence_file:
                if i > 0:
                    metadata = line.split("\t")
                    project_scientific_name = metadata[241]
                    project_species, project_subspecies, project_group = self.convert_scientific_name(project_scientific_name, is_aligned=False)
                    if self.is_valid(project_species, metadata):
                        print(f"{metadata[68]} : {metadata[102]} : {metadata[103]}")
                        metadata = self.format(project_species, project_subspecies, project_group, metadata, j)
                        print(metadata["time"])
                        if self.is_year_valid(metadata["date"]):
                            # constructor.write_metadata(source_metadata, metadata)
                            # constructor.update_observation_list(dataset, project_species, metadata["_id"])
                            k += 1
                        j += 1
                    if i % 10000 == 0:
                        print(f"Edited File: {i}")
                i += 1
        # save_data_to_file_(f"{constructor.statistics_files_path}count.json", {"count": {"total_count": i, "valid_format_count": j, "valid_year_count": k}})

    def enrich_source_metadata(self):
        client = MongoClient()
        metadata_database = client[self.metadata_database_name]
        observation_table = metadata_database.observation
        gbif_id_to_id = {}
        for metadata in observation_table.find(): gbif_id_to_id[metadata["gbif_id"]] = metadata["_id"]
        i = 0
        with open(self.multimedia_file_path, 'r', encoding="UTF-8") as occurrence_file:
            for line in occurrence_file:
                if i > 0:
                    keys = line.split("\t")
                    gbif_id = keys[0]
                    media_type_ = keys[1]
                    media_link = keys[3]
                    if media_type_ == "StillImage":
                        media_type_ = "Image"
                    if media_type_ == "Sound":
                        media_type_ = "Audio"
                    if gbif_id in gbif_id_to_id:
                        observation_table.find_one_and_update({"_id": gbif_id_to_id[gbif_id]}, {"$push": {"media_links": media_link, "media_types": media_type_}})
                if i % 10000 == 0:
                    print(f"Enriched File: {i}")
                i += 1

    # ************************************************ MINOR METHODS ***************************************************

    def is_valid(self, project_species, metadata):
        scientific_name = metadata[188]
        accepted_scientific_name = metadata[240]
        day = metadata[108]
        month = metadata[107]
        year = metadata[106]
        time_ = metadata[103]
        hour, minute = self.separate_time(time_) if self.project_tag == "observationorg" else self.separate_time(metadata[102].split("T")[1])
        latitude = metadata[137]
        longitude = metadata[138]
        location_uncertainty = metadata[139]
        age = metadata[76].lower().title()
        degree_of_match = self.determine_taxonomy_match(project_species, is_species=True, is_aligned=False)
        issues = metadata[225].strip().split(";")
        if scientific_name != accepted_scientific_name and self.project_tag == "observationorg":
            valid = False
        elif day == "" or month == "" or year == "":
            valid = False
        elif time_ == "" and self.project_tag == "inaturalist":
            valid = False
        elif hour == "" or minute == "":
            valid = False
        elif latitude == "":
            valid = False
        elif longitude == "":
            valid = False
        elif location_uncertainty == "":
            valid = False
        elif age == "Egg":
            valid = False
        elif degree_of_match == 0:
            valid = False
        elif any(x in self.observation_issue_indicators for x in issues):
            valid = False
        else:
            valid = True
        return valid

    def format(self, project_species, project_subspecies, project_group, metadata, i):
        id_ = f"{self.subset_tag}_{i}"
        gbif_id = metadata[0]
        project_id = metadata[68] if self.project_tag == "inaturalist" else metadata[68].split(".")[1]
        latitude = float(metadata[137])
        longitude = float(metadata[138])
        location_uncertainty = float(metadata[139])
        date_ = metadata[102].split("T")[0]
        hour, minute = self.separate_time(metadata[103]) if self.project_tag == "observationorg" else self.separate_time(metadata[102].split("T")[1])
        time_ = f"{hour}:{minute}"
        sex = metadata[75].lower().title() if (metadata[75] == "FEMALE" or metadata[75] == "MALE") else "Unknown"
        age = metadata[76] if metadata[76] != "" else "Unknown"
        observer = " ".join(unidecode.unidecode(re.sub("[,.-]", " ", metadata[70])).lower().title().split())
        formatted_metadata = {"_id": id_,
                              f"{self.parent_project_tag}_id": gbif_id,
                              f"{self.project_tag}_id": project_id,
                              f"{self.project_tag}_species": project_species,
                              f"{self.project_tag}_subspecies": project_subspecies,
                              f"{self.project_tag}_group": project_group,
                              "latitude": latitude,
                              "longitude": longitude,
                              "location_uncertainty": location_uncertainty,
                              "date": date_,
                              "time": time_,
                              "sex": [sex],
                              "age": [age],
                              "media_types": [],
                              "media_links": []}
        if self.project_tag == "inaturalist": formatted_metadata["observer"] = observer
        return formatted_metadata

    # ******************************************************************************************************************
