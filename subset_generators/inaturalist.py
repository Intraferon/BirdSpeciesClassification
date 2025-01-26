from subset_generators.gbif import GBIF
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from preprocessors.image_preprocessor import ImagePreprocessor
from bs4 import BeautifulSoup
import mimetypes
mimetypes.init()
import time
from taxonomy_generators.taxonomy import Taxonomy
from preprocessors.audio_preprocessor import AudioPreprocessor
from utility import *
import enum
import requests
import shutil
import asyncio
import aiohttp
import aiofile


# Citation: GBIF.org (20 February 2022) GBIF Occurrence Download https://doi.org/10.15468/dl.dsnax4 (Image)
# Citation: GBIF.org (20 February 2022) GBIF Occurrence Download https://doi.org/10.15468/dl.x7x52h (Audio)
# Dataset Update Date: 10 February 2022
# Metadata Update Date: 19 February 2022
# Taxonomy Download Date: 21 February 2022


class INaturalist(GBIF):

    # Taxonomy
    clements2019_to_clements2021_replacements = Taxonomy.get_replacement("clements2019", "clements2021")
    clements2021_to_clements2019_replacements = Taxonomy.get_replacement("clements2021", "clements2019")
    clements2019_with_clements2021_removals = Taxonomy.get_removal("clements2019", "clements2021")
    clements2021_with_clements2019_removals = Taxonomy.get_removal("clements2021", "clements2019")

    class TaxonChanges(enum.Enum):
        DROP = 1
        SWAP = 2
        MERGE = 3
        SPLIT = 4
        UNKNOWN = 5

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self, media_type, base_taxonomy, subset_taxonomy_to_base_taxonomy_replacements, aligned_taxonomy, base_taxonomy_to_aligned_taxonomy_replacements):

        self.media_type = media_type

        # Paths
        subset_path = f"C:/Users/sanam/Documents/Masters/Resources/Citizen Science Projects/iNaturalist/{media_type}"

        # Tags
        subset_tag = f"inaturalist_{media_type.lower()}"
        project_tag = "inaturalist"

        # Parent Constructor
        super(INaturalist, self).__init__(subset_path, subset_tag, project_tag,
                                          base_taxonomy, subset_taxonomy_to_base_taxonomy_replacements, aligned_taxonomy, base_taxonomy_to_aligned_taxonomy_replacements)

    # ************************************************ MAJOR METHODS ***************************************************

    def download_source_data(self, constructor, dataset, source_metadata, dataset_type_=None, solution_type_=""):

        valid_data_format_list = []
        for data_format in mimetypes.types_map:
            if mimetypes.types_map[data_format].split("/")[0] == self.media_type.lower():
                valid_data_format_list.append(data_format.replace(".", ""))

        def download_batch(x):
            media_url_ = x[0]
            data_id_ = x[1]
            try:
                request_ = requests.get(media_url_)
                data_format_ = media_url_.split(".")[-1].lower() if self.media_type == "Image" else media_url_.split("?")[0].split(".")[-1].lower()
                if data_format_ in valid_data_format_list:
                    if self.media_type == "Image":
                        data_file_path_ = f"{self.aligned_data_files_path}{data_id_}.{data_format_}"
                    else:
                        data_file_path_ = f"{self.external_aligned_data_files_path}{data_id_}.{data_format_}"
                    with open(data_file_path_, "wb") as data_file_:
                        data_file_.write(request_.content)
                    return data_id_
                else:
                    return None
            except Exception as e:
                print(e)
                return None

        print("Reading Downloaded Files...")
        downloaded_files = set(read_data_from_file_(f"{self.subset_path}/downloaded.txt"))
        print("Reading Species...")
        species_list = constructor.read_species_list(dataset)
        request_size = 10
        data_id_list = []
        media_url_list = []
        request_list_size = 0
        i = 0
        batch_i = 0
        batch_limit = 1000
        batch_limit_reached = False
        metadata_limit = 10000000
        t = 0
        with ThreadPoolExecutor(max_workers=50) as thread_pool:
            for species in species_list:
                if self.media_type == "Image":
                    species_path = f"{self.aligned_data_files_path}{species}"
                else:
                    species_path = f"{self.external_aligned_data_files_path}{species}"
                print("Reading Metadata...")
                metadata_list = constructor.read_compound_attribute_list(dataset, source_metadata, species, ["_id", "media_links", "media_types"], dataset_type_=dataset_type_, solution_type_=solution_type_)
                if len(metadata_list) <= metadata_limit:
                    print("Downloading Batches...")
                    create_folder_(species_path)
                    for metadata in metadata_list:
                        id_ = metadata["_id"]
                        media_urls = metadata["media_links"]
                        media_types = metadata["media_types"]
                        media_url_count = len(media_urls)
                        j = 0
                        for k in range(media_url_count):
                            if media_types[k] == self.media_type:
                                if request_list_size < request_size:
                                    data_id = f"{species}/{id_}_{j}"
                                    media_url = media_urls[k]
                                    if data_id not in downloaded_files:
                                        data_id_list.append(data_id)
                                        media_url_list.append(media_url)
                                        request_list_size += 1
                                if request_list_size == request_size:
                                    start = time.time()
                                    downloaded_data_id_list = list(thread_pool.map(download_batch, zip(media_url_list, data_id_list)))
                                    end = time.time()
                                    t += (end - start)
                                    downloaded_files.update([downloaded_data_id for downloaded_data_id in downloaded_data_id_list if downloaded_data_id is not None])
                                    media_url_list = []
                                    data_id_list = []
                                    request_list_size = 0
                                    batch_i += 1
                                    print(f"Batch: {batch_i}")
                                j += 1
                            i += 1
                        if batch_i > batch_limit:
                            batch_limit_reached = True
                            break
                if batch_limit_reached:
                    break
            start = time.time()
            downloaded_data_id_list = list(thread_pool.map(download_batch, zip(media_url_list, data_id_list)))
            end = time.time()
            t += (end - start)
            downloaded_files.update([downloaded_data_id for downloaded_data_id in downloaded_data_id_list if downloaded_data_id is not None])
        print(f"t: {t}")
        save_data_to_file_(f"{self.subset_path}/downloaded.txt", downloaded_files)

    @staticmethod
    def download_source_taxonomy():

        def extract_detailed_taxon_list(detailed_taxon_column):
            detailed_taxon_list = []
            detailed_taxon_column_rows = detailed_taxon_column.find_all("li")
            for k in range(0, len(detailed_taxon_column_rows)):
                taxon_details = detailed_taxon_column_rows[k].find("div", {"class": "image_and_content"})
                scientific_name = taxon_details.find("span", {"class": "sciname"}).string.lower().title().strip()
                status = taxon_details.find_all("span")[-1].string
                taxon_code = taxon_details.find("a")
                for s in taxon_code.select_population("span"): s.segment_audio()
                taxon_code = taxon_code.string.strip()
                if status is None: status = taxon_details.find_all("span")[-2].string
                if scientific_name != "Not Assigned": detailed_taxon_list.append([taxon_code, scientific_name, status])
            return detailed_taxon_list

        taxonomy = []
        driver = webdriver.Chrome("C:/Users/sanam/Documents/Installers/chrome_driver/chromedriver.exe")
        base_taxonomy_url = "https://www.inaturalist.org/taxon_changes?filters%5Bancestor_taxon_id%5D=&filters%5Bancestor_taxon_name%5D=&filters%5Bchange_group%5D=&filters%5Bcommitted%5D=Yes&filters%5Bdrop%5D=0&filters%5Biconic_taxon_id%5D=3&filters%5Bmerge%5D=0&filters%5Bsource_id%5D=&filters%5Bsplit%5D=0&filters%5Bstage%5D=0&filters%5Bswap%5D=0&filters%5Btaxon_id%5D=&filters%5Btaxon_name%5D=&filters%5Btaxon_scheme_id%5D=&filters%5Buser_id%5D=&page=(?)&utf8=%E2%9C%93"
        number_pages = 180
        for i in range(1, number_pages + 1):
            print(f"Page: {i}")
            taxonomy_url = base_taxonomy_url.replace("(?)", str(i))
            driver.get(taxonomy_url)
            time.sleep(10)
            html_structure = BeautifulSoup(driver.page_source, "html.parser")
            table_structure = html_structure.find("div", {"class": "col-xs-8"})
            rows = table_structure.find_all("div", {"class": "taxon_change"})
            for j in range(0, len(rows)):
                header = rows[j].find("h2")
                for a in header.select_population("a"): a.segment_audio()
                date_comitted = header.getText()[-12:-2]
                row = rows[j].find("tbody").find("tr")
                old_detailed_taxon_column = row.find_all("td")[0].find("ul", {"class": "change_taxon"})
                new_detailed_taxon_column = row.find_all("td")[2].find("ul", {"class": "change_taxon"})
                old_detailed_taxon_list = extract_detailed_taxon_list(old_detailed_taxon_column)
                new_detailed_taxon_list = extract_detailed_taxon_list(new_detailed_taxon_column)
                old_detailed_taxon_list_string = "#".join([f"{x[0]}|{x[1]}|{x[2]}" for x in old_detailed_taxon_list])
                new_detailed_taxon_list_string = "#".join([f"{x[0]}|{x[1]}|{x[2]}" for x in new_detailed_taxon_list])
                taxonomy.append([old_detailed_taxon_list_string, new_detailed_taxon_list_string, date_comitted])
        save_data_to_file_(f"{Taxonomy.downloaded_files_path}inaturalist.csv", taxonomy)

    def preprocess_data(self, parameters=None, constructor=None, dataset=None, solution_type_=""):
        if self.media_type == "Image":
            image_preprocessor = ImagePreprocessor(self, constructor=constructor, dataset=dataset, solution_type_=solution_type_)
            image_preprocessor.run()
        else:
            audio_preprocessor = AudioPreprocessor(self, constructor=constructor, dataset=dataset, solution_type_=solution_type_)
            audio_preprocessor.run(parameters)

    @classmethod
    def create_comparison_from_taxonomy(cls):

        def extract_taxon_details(taxon_details_list):
            taxon_codes = [x[0] for x in taxon_details_list]
            scientific_names_ = [x[1] for x in taxon_details_list]
            statuses_ = [x[2] for x in taxon_details_list]
            return taxon_codes, scientific_names_, statuses_

        inaturalist_with_other_removals = set()
        other_to_inaturalist_replacements = {}
        taxonomy = read_data_from_file_(f"{Taxonomy.downloaded_files_path}inaturalist.csv")
        for row in taxonomy:
            old_detailed_taxon_list_string = row[0]
            new_detailed_taxon_list_string = row[1]
            date_committed = f"{row[2].split('-')[2]}-{row[2].split('-')[1]}-{row[2].split('-')[0]}"
            if date_committed < "2022-02-19":
                old_detailed_taxon_list = [x.split("|") for x in old_detailed_taxon_list_string.split("#")]
                if old_detailed_taxon_list == [[""]]: old_detailed_taxon_list = []
                new_detailed_taxon_list = [x.split("|") for x in new_detailed_taxon_list_string.split("#")]
                if new_detailed_taxon_list == [[""]]: new_detailed_taxon_list = []
                old_taxon_codes, old_scientific_names, old_statuses = extract_taxon_details(old_detailed_taxon_list)
                new_taxon_codes, new_scientific_names, new_statuses = extract_taxon_details(new_detailed_taxon_list)
                taxon_change_type = cls.determine_taxon_change_type(old_scientific_names, new_scientific_names)
                clements2021_update = cls.is_clements2021_update(old_scientific_names, new_scientific_names, date_committed) if taxon_change_type != cls.TaxonChanges.DROP else None
                cross_species = cls.is_cross_species(old_scientific_names, new_scientific_names) if taxon_change_type != cls.TaxonChanges.DROP else False
                taxon_change_unambiguous = cls.is_taxon_change_unambiguous(old_statuses, new_statuses) if taxon_change_type != cls.TaxonChanges.DROP else True
                if taxon_change_unambiguous:
                    if taxon_change_type == cls.TaxonChanges.DROP:
                        inaturalist_with_other_removals.add(old_scientific_names[0])
                    elif taxon_change_type == cls.TaxonChanges.SWAP:
                        if clements2021_update:
                            other_to_inaturalist_replacements[new_scientific_names[0]] = old_scientific_names[0]
                            inaturalist_with_other_removals.add(new_scientific_names[0])
                        else:
                            other_to_inaturalist_replacements[old_scientific_names[0]] = new_scientific_names[0]
                            inaturalist_with_other_removals.add(old_scientific_names[0])
                    elif taxon_change_type == cls.TaxonChanges.MERGE:
                        if clements2021_update:
                            if cross_species:
                                inaturalist_with_other_removals.add(new_scientific_names[0])
                        else:
                            for i in range(0, len(old_scientific_names)):
                                other_to_inaturalist_replacements[old_scientific_names[i]] = new_scientific_names[0]
                                inaturalist_with_other_removals.add(old_scientific_names[i])
                    elif taxon_change_type == cls.TaxonChanges.SPLIT:
                        if clements2021_update:
                            for i in range(0, len(new_scientific_names)):
                                other_to_inaturalist_replacements[new_scientific_names[i]] = old_scientific_names[0]
                                inaturalist_with_other_removals.add(new_scientific_names[i])
                        else:
                            if cross_species:
                                inaturalist_with_other_removals.add(old_scientific_names[0])
                    else:
                        for i in range(0, len(old_scientific_names)):
                            inaturalist_with_other_removals.add(old_scientific_names[i])
                        for i in range(0, len(new_scientific_names)):
                            inaturalist_with_other_removals.add(new_scientific_names[i])
                else:
                    scientific_names = old_scientific_names + new_scientific_names
                    statuses = old_statuses + new_statuses
                    if cross_species:
                        if clements2021_update:
                            for i in range(0, len(scientific_names)):
                                inaturalist_with_other_removals.add(scientific_names[i])
                        else:
                            for i in range(0, len(scientific_names)):
                                if statuses[i] != "Active":
                                    inaturalist_with_other_removals.add(scientific_names[i])
        save_data_to_file_(f"{Taxonomy.removal_files_path}inaturalist_with_other.txt", inaturalist_with_other_removals)
        save_data_to_file_(f"{Taxonomy.replacement_files_path}other_to_inaturalist.json", other_to_inaturalist_replacements)

    # ************************************************ MINOR METHODS ***************************************************

    @classmethod
    def determine_taxon_change_type(cls, old_scientific_names, new_scientific_names):
        old_scientific_names_count = len(old_scientific_names)
        new_scientific_names_count = len(new_scientific_names)
        if old_scientific_names_count == 1 and new_scientific_names_count == 0:
            taxon_change_type = cls.TaxonChanges.DROP
        elif old_scientific_names_count == 1 and new_scientific_names_count == 1:
            taxon_change_type = cls.TaxonChanges.SWAP
        elif old_scientific_names_count > 1 and new_scientific_names_count == 1:
            taxon_change_type = cls.TaxonChanges.MERGE
        elif old_scientific_names_count == 1 and new_scientific_names_count > 1:
            taxon_change_type = cls.TaxonChanges.SPLIT
        else:
            taxon_change_type = cls.TaxonChanges.UNKNOWN
        return taxon_change_type

    @classmethod
    def is_clements2021_update(cls, old_scientific_names, new_scientific_names, date_committed):
        clements2021_update = False
        if date_committed >= "2021-07-24":
            old_species = [cls.separate_scientific_name(x)[0] for x in old_scientific_names]
            new_species = [cls.separate_scientific_name(x)[0] for x in new_scientific_names]
            if cls.is_cross_species(old_scientific_names, new_scientific_names):
                if all(x in cls.clements2019_with_clements2021_removals for x in old_species) and all(x in cls.clements2021_with_clements2019_removals for x in new_species):
                    clements2021_update = True
                if all(x in cls.clements2019_to_clements2021_replacements for x in old_species) and all(cls.clements2019_to_clements2021_replacements[x] == y for x in old_species for y in new_species):
                    clements2021_update = True
            else:
                clements2021_update = True
        return clements2021_update

    @classmethod
    def is_cross_species(cls, old_scientific_names, new_scientific_names):
        cross_species = True
        old_species = [cls.separate_scientific_name(x)[0] for x in old_scientific_names]
        new_species = [cls.separate_scientific_name(x)[0] for x in new_scientific_names]
        if all(x == old_species[0] for x in (old_species + new_species)):
            cross_species = False
        return cross_species

    @classmethod
    def is_taxon_change_unambiguous(cls, old_statuses, new_statuses):
        taxon_change_unambiguous = False
        if all(x == "Inactive" for x in old_statuses) and all(x == "Active" for x in new_statuses):
            taxon_change_unambiguous = True
        return taxon_change_unambiguous

    # ******************************************************************************************************************
