from utility import *
import mimetypes
mimetypes.init()
from preprocessors.audio_preprocessor import AudioPreprocessor
from taxonomy_generators.taxonomy import Taxonomy
from subset_generators.subset import Subset
from concurrent.futures import ThreadPoolExecutor
import requests
import unidecode
from nltk.stem import WordNetLemmatizer
import collections
import re
import datetime
from selenium import webdriver
from dataset_generators.constants import *
from bs4 import BeautifulSoup
import time
import csv


# Metadata Download Date: 20 February 2022
# Taxonomy Download Data: 20 February 2022


class XenoCanto(Subset):

    # Arrays
    english_words = set(read_data_from_file_(f"{RESOURCE_PATH}arrays/english_words.txt"))
    uncertainty_indicators = read_data_from_file_(f"{RESOURCE_PATH}arrays/uncertainty_indicators.txt")
    relevancy_indicators = read_data_from_file_(f"{RESOURCE_PATH}arrays/relevancy_indicators.txt")
    irrelevancy_indicators = read_data_from_file_(f"{RESOURCE_PATH}arrays/irrelevancy_indicators.txt")
    lemmatizer_corrections = read_data_from_file_(f"{RESOURCE_PATH}arrays/lemmatizer_corrections.json")
    punctuation_to_remove = read_data_from_file_(f"{RESOURCE_PATH}arrays/punctuation_to_remove.txt")
    punctuation_to_replace = read_data_from_file_(f"{RESOURCE_PATH}arrays/punctuation_to_replace.json")
    words_to_remove = set(read_data_from_file_(f"{RESOURCE_PATH}arrays/words_to_remove.txt"))
    words_to_replace = read_data_from_file_(f"{RESOURCE_PATH}arrays/words_to_replace.json")
    words_to_correct = read_data_from_file_(f"{RESOURCE_PATH}arrays/words_to_correct.json")
    words_to_rearrange = read_data_from_file_(f"{RESOURCE_PATH}arrays/words_to_rearrange.txt")
    words_to_separate = read_data_from_file_(f"{RESOURCE_PATH}arrays/words_to_separate.json")
    numbers_to_words = read_data_from_file_(f"{RESOURCE_PATH}arrays/numbers_to_words.json")
    vocalisation_indicators = read_data_from_file_(f"{RESOURCE_PATH}arrays/vocalisation_indicators.txt")
    subspecies_uncertainty_indicators = read_data_from_file_(f"{RESOURCE_PATH}arrays/subspecies_uncertainty_indicators.txt")

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self, base_taxonomy, subset_taxonomy_to_base_taxonomy_replacements, aligned_taxonomy, base_taxonomy_to_aligned_taxonomy_replacements):

        # Paths
        subset_path = "C:/Users/sanam/Documents/Masters/Resources/Citizen Science Projects/Xeno Canto"

        # Tags
        subset_tag = project_tag = parent_project_tag = "xenocanto"

        # Years
        minimum_year = "1994"
        maximum_year = "2021"

        # Taxonomy
        self.xenocanto_taxonomy = Taxonomy.get_base("xenocanto")

        # Parent Constructor
        super(XenoCanto, self).__init__(subset_path, subset_tag, project_tag, parent_project_tag,
                                        minimum_year, maximum_year,
                                        base_taxonomy, subset_taxonomy_to_base_taxonomy_replacements, aligned_taxonomy, base_taxonomy_to_aligned_taxonomy_replacements)

        # Working Variables
        self.attribute_vocabulary = set()

    # ************************************************ MAJOR METHODS ***************************************************

    def download_source_data(self, constructor, dataset, source_metadata, dataset_type_=None, solution_type_=""):

        def download_batch(x):
            media_url_ = x[0]
            data_file_path_ = x[1]
            try:
                request_ = requests.get(media_url_)
                data_url_ = request_.url
                with open(data_file_path_, "wb") as data_file_:
                    data_file_.write(request_.content)
            except Exception as e:
                print(f"exception: {e}")

        print("Reading Species...")
        species_list = constructor.read_species_list(dataset)
        request_size = 4
        data_file_path_list = []
        media_url_list = []
        request_list_size = 0
        i = 0
        batch_i = 0
        batch_limit = 2000
        batch_limit_reached = False
        metadata_limit = 1000000
        t = 0
        with ThreadPoolExecutor(max_workers=50) as thread_pool:
            for species in species_list:
                species_path = f"{self.external_aligned_data_files_path}{species}"
                create_folder_(species_path)
                print("Reading Metadata...")
                metadata_list = constructor.read_compound_attribute_list(dataset, source_metadata, species, ["_id", "media_links", "media_types"], dataset_type_=dataset_type_, solution_type_=solution_type_)
                print("Downloading Batches...")
                if len(metadata_list) <= metadata_limit:
                    for metadata in metadata_list:
                        id_ = metadata["_id"]
                        media_urls = metadata["media_links"]
                        media_types = metadata["media_types"]
                        j = 0
                        for k in range(len(media_urls)):
                            if media_types[k] == "Audio":
                                if request_list_size < request_size:
                                    data_file_path = f"{self.external_aligned_data_files_path}{species}/{id_}_{j}.mp3"
                                    media_url = media_urls[k]
                                    if not os.path.exists(data_file_path):
                                        data_file_path_list.append(data_file_path)
                                        media_url_list.append(media_url)
                                        request_list_size += 1
                                if request_list_size == request_size:
                                    start = time.time()
                                    list(thread_pool.map(download_batch, zip(media_url_list, data_file_path_list)))
                                    end = time.time()
                                    t += (end - start)
                                    media_url_list = []
                                    data_file_path_list = []
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
            list(thread_pool.map(download_batch, zip(media_url_list, data_file_path_list)))
            end = time.time()
            t += (end - start)
        print(f"t: {t}")

    def download_source_metadata(self):
        base_file_path = f"{self.downloaded_metadata_files_path}xenocanto_"
        base_url = "https://www.xeno-canto.org/api/2/recordings?query="
        ratings = ["A", "B", "C", "D", "E"]
        for rating in ratings:
            default_file_path = f"{base_file_path}{rating}_"
            default_url = f"{base_url}q:{rating}"
            r = requests.get(url=default_url)
            data = r.json()
            paged_file_path = f"{default_file_path}1.json"
            save_data_to_file_(paged_file_path, data)
            num_pages = int(data["numPages"])
            print(f"{rating}: 1 of {num_pages}")
            for page in range(2, num_pages + 1):
                paged_file_path = f"{default_file_path}{page}.json"
                paged_url = f"{default_url}&page={page}"
                r = requests.get(url=paged_url)
                data = r.json()
                save_data_to_file_(paged_file_path, data)
                print(f"{rating}: {page} of {num_pages}")

    @staticmethod
    def download_source_taxonomy():
        taxonomy = [["Scientific Name", "Common Name", "Status"]]
        driver = webdriver.Chrome("C:/Users/sanam/Documents/Installers/chrome_driver/chromedriver.exe")
        collected_species_url = "https://xeno-canto.org/collection/species/all"
        driver.get(collected_species_url)
        time.sleep(5)
        html_structure = BeautifulSoup(driver.page_source, "html.parser")
        table_structure = html_structure.find("div", {"id": "content-area"}).find("table", {"class": "results"}).tbody
        rows = table_structure.find_all("tr")
        for i in range(0, len(rows)):
            row = rows[i]
            cells = row.find_all("td")
            common_name = "" if cells[0].span.a.string is None else cells[0].span.a.string.strip()
            scientific_name = cells[1].string.lower().title().strip()
            status = "" if cells[2].string is None else cells[2].string.strip()
            taxonomy.append([scientific_name, common_name, status])
        base_wanted_species_url = "https://xeno-canto.org/collection/species/wanted"
        driver.get(base_wanted_species_url)
        time.sleep(5)
        html_structure = BeautifulSoup(driver.page_source, "html.parser")
        selector_structure = html_structure.find("div", {"id": "content-area"}).find("li", {"class": "current"})
        options = selector_structure.find_all("option")
        for i in range(1, len(options)):
            option = options[i]
            region_code = option["value"].split("=")[1]
            print(region_code)
            wanted_species_url = f"{base_wanted_species_url}?cnt={region_code}"
            driver.get(wanted_species_url)
            time.sleep(5)
            html_structure = BeautifulSoup(driver.page_source, "html.parser")
            table_structure = html_structure.find("table", {"id": "main-table"}).find("table", {"class": "results"}).tbody
            rows = table_structure.find_all("tr")
            for j in range(0, len(rows)):
                row = rows[j]
                cells = row.find_all("td")
                common_name = "" if cells[0].a.string is None else cells[0].a.string.strip()
                scientific_name = cells[1].string.lower().title().strip()
                status = ""
                taxonomy.append([scientific_name, common_name, status])
        taxonomy_file_path = f"{Taxonomy.downloaded_files_path}xenocanto.csv"
        taxonomy_file = open(taxonomy_file_path, "w", newline="")
        writer = csv.writer(taxonomy_file)
        writer.writerows(taxonomy)
        taxonomy_file.close()

    def preprocess_data(self, parameters, constructor=None, dataset=None, solution_type_=""):
        audio_preprocessor = AudioPreprocessor(self, constructor=constructor, dataset=dataset, solution_type_=solution_type_)
        audio_preprocessor.run(parameters)

    def extract_downloaded_project_taxonomy(self):
        taxonomy = {}
        base_file_path = f"{self.downloaded_metadata_files_path}xenocanto_"
        ratings = ["A", "B", "C", "D", "E"]
        num_pages = [452, 531, 264, 66, 17]
        i = 0
        for r in range(0, len(ratings)):
            rating = ratings[r]
            for page in range(1, num_pages[r] + 1):
                file_path = f"{base_file_path}{rating}_{page}.json"
                data = read_data_from_file_(file_path)
                for recording in data["recordings"]:
                    xenocanto_scientific_name = f"{recording['gen']} {recording['sp']} {recording['ssp']}"
                    xenocanto_species, xenocanto_subspecies = self.separate_scientific_name(xenocanto_scientific_name)[0:2]
                    if xenocanto_species != "": taxonomy = Taxonomy.update_base_taxonomy(taxonomy, xenocanto_species, xenocanto_subspecies)
                    if i % 10000 == 0:
                        print(f"Read File: {i}")
                    i += 1
        save_data_to_file_(f"{Taxonomy.downloaded_project_files_path}{self.project_tag}.json", taxonomy)

    def extract_metadata_structure(self):
        file_path = f"{self.downloaded_metadata_files_path}xenocanto_A_1.json"
        data = read_data_from_file_(file_path)
        print(data["recordings"][0])

    def edit_source_metadata(self):
        dataset = self.get_dataset_handle(is_aligned=False)
        source_metadata = self.get_source_metadata_handle(is_aligned=False)
        base_file_path = f"{self.downloaded_metadata_files_path}xenocanto_"
        ratings = ["A", "B", "C", "D", "E"]
        num_pages = [452, 531, 264, 66, 17]
        i = 0
        j = 0
        k = 0
        for r in range(0, len(ratings)):
            rating = ratings[r]
            for page in range(1, num_pages[r] + 1):
                file_path = f"{base_file_path}{rating}_{page}.json"
                data = read_data_from_file_(file_path)
                for metadata in data["recordings"]:
                    xenocanto_scientific_name = f"{metadata['gen']} {metadata['sp']} {metadata['ssp']}"
                    xenocanto_species, xenocanto_subspecies, xenocanto_group = self.convert_scientific_name(xenocanto_scientific_name, is_aligned=False)
                    if xenocanto_species == "":
                        xenocanto_scientific_name = f"{metadata['gen']} {metadata['sp']}"
                        xenocanto_species, xenocanto_subspecies, xenocanto_group = self.convert_scientific_name(xenocanto_scientific_name, is_aligned=False)
                    if self.is_valid(xenocanto_species, metadata):
                        metadata = self.format(xenocanto_species, xenocanto_subspecies, xenocanto_group, metadata, j)
                        if self.is_year_valid(metadata["date"]):
                            self.write_metadata(source_metadata, metadata)
                            self.update_observation_list(dataset, xenocanto_species, metadata["_id"])
                            k += 1
                        j += 1
                    if i % 10000 == 0:
                        print(f"Edited File: {i}")
                    i += 1
        save_data_to_file_(f"{self.statistics_files_path}count.json", {"count": {"total_count": i, "valid_format_count": j, "valid_year_count": k}})

    # ************************************************ MINOR METHODS ***************************************************

    def is_valid(self, xenocanto_species, metadata):
        date_ = metadata["date"]
        hour, minute = self.separate_time(metadata["time"])
        latitude = metadata["lat"]
        longitude = metadata["lng"]
        degree_of_match = self.determine_taxonomy_match(xenocanto_species, is_species=True, is_aligned=False)
        if not self.is_date_valid(date_):
            valid = False
        elif hour == "" or minute == "":
            valid = False
        elif latitude == "" or latitude is None:
            valid = False
        elif longitude == "" or longitude is None:
            valid = False
        elif degree_of_match == 0:
            valid = False
        else:
            valid = True
        return valid

    def format(self, xenocanto_species, xenocanto_subspecies, xenocanto_group, metadata, i):
        id_ = f"{self.subset_tag}_{i}"
        xenocanto_id = metadata["id"]
        uncertain_xenocanto_subspecies = uncertain_xenocanto_group = ""
        if xenocanto_subspecies == "" and xenocanto_group == "":
            uncertain_xenocanto_scientific_name = f"{metadata['gen']} {metadata['sp']} {metadata['ssp']}"
            uncertain_xenocanto_subspecies, uncertain_xenocanto_group = self.convert_uncertain_scientific_name(uncertain_xenocanto_scientific_name)
        date_ = f"{metadata['date'].split('-')[0]}-{metadata['date'].split('-')[1].zfill(2)}-{metadata['date'].split('-')[2].zfill(2)}".strip()
        hour, minute = self.separate_time(metadata["time"])
        time_ = f"{hour}:{minute}"
        latitude = float(metadata["lat"])
        longitude = float(metadata["lng"])
        attributes = self.format_type(WordNetLemmatizer(), metadata["type"])
        sexes, ages, general_vocalisations, specific_vocalisations = self.separate_type(attributes)
        observer = " ".join(unidecode.unidecode(re.sub("[,.-]", " ", metadata["rec"])).lower().title().strip().split())
        has_background_species = 0 if (len(metadata["also"]) == 1 and metadata["also"][0] == "") else 1
        rating_letter = metadata["q"]
        if rating_letter == "A":
            rating = 5.00
        elif rating_letter == "B":
            rating = 4.00
        elif rating_letter == "C":
            rating = 3.00
        elif rating_letter == "D":
            rating = 2.00
        elif rating_letter == "E":
            rating = 1.00
        else:
            rating = 0.00
        duration = metadata["length"]
        media_types = ["Audio"]
        media_links = [metadata["file"].strip()]
        formatted_metadata = {"_id": id_,
                              f"{self.project_tag}_id": xenocanto_id,
                              f"{self.project_tag}_species": xenocanto_species,
                              f"{self.project_tag}_subspecies": xenocanto_subspecies,
                              f"{self.project_tag}_group": xenocanto_group,
                              f"uncertain_{self.project_tag}_subspecies": uncertain_xenocanto_subspecies,
                              f"uncertain_{self.project_tag}_group": uncertain_xenocanto_group,
                              "latitude": latitude,
                              "longitude": longitude,
                              "date": date_,
                              "time": time_,
                              "sex": sexes,
                              "age": ages,
                              "general_vocalisation": general_vocalisations,
                              "specific_vocalisation": specific_vocalisations,
                              "observer": observer,
                              "has_background_species": has_background_species,
                              "rating": rating,
                              "duration": duration,
                              "media_types": media_types,
                              "media_links": media_links}
        return formatted_metadata

    @staticmethod
    def is_date_valid(date_):
        if date_ == "":
            valid = False
        elif not re.search(r"^(19|20)[0-9][0-9]$", date_.split("-")[0]):
            valid = False
        elif int(date_.split("-")[0]) > 2022:
            valid = False
        else:
            try:
                datetime.datetime(int(date_.split("-")[0]), int(date_.split("-")[1]), int(date_.split("-")[2]))
                valid = True
            except ValueError:
                valid = False
        return valid

    def convert_uncertain_scientific_name(self, scientific_name):
        scientific_name = " ".join(unidecode.unidecode(scientific_name).lower().title().strip().split())
        scientific_name = re.sub(r"\.*[\[(]*(" + "|".join(list(self.subspecies_uncertainty_indicators)) + r")[])]*\.*", "", scientific_name)
        uncertain_subspecies, uncertain_group = self.convert_scientific_name(scientific_name, is_aligned=False)[1:]
        return uncertain_subspecies, uncertain_group

    def separate_type(self, attributes):

        def update(attribute_composition_, i_, attribute_):
            if i_ not in attribute_composition_:
                attribute_composition_[i_] = [attribute_]
            else:
                attribute_composition_[i_].append(attribute_)
            return attribute_composition_

        sex_attribute_composition = {}
        age_attribute_composition = {}
        vocalisation_attribute_composition = {}
        attribute_count = len(attributes)
        for i in range(0, attribute_count):
            attribute = attributes[i]
            if "Male" in attribute:
                attribute = attribute.replace("Male", "").strip()
                sex_attribute_composition = update(sex_attribute_composition, i, "Male")
            if "Female" in attribute:
                attribute = attribute.replace("Female", "").strip()
                sex_attribute_composition = update(sex_attribute_composition, i, "Female")
            if "Duet" in attribute:
                sex_attribute_composition = update(sex_attribute_composition, i, "Male")
                sex_attribute_composition = update(sex_attribute_composition, i, "Female")
                age_attribute_composition = update(age_attribute_composition, i, "Adult")
            if "Adult" in attribute:
                attribute = attribute.replace("Adult", "").strip()
                age_attribute_composition = update(age_attribute_composition, i, "Adult")
            if "Immature" in attribute:
                attribute = attribute.replace("Immature", "").strip()
                age_attribute_composition = update(age_attribute_composition, i, "Immature")
            if "Plastic Song" in attribute:
                age_attribute_composition = update(age_attribute_composition, i, "Immature")
            if "Juvenile" in attribute:
                attribute = attribute.replace("Juvenile", "").strip()
                age_attribute_composition = update(age_attribute_composition, i, "Juvenile")
            if "Sub Song" in attribute:
                age_attribute_composition = update(age_attribute_composition, i, "Juvenile")
            if "Fledgling" in attribute:
                attribute = attribute.replace("Fledgling", "").strip()
                age_attribute_composition = update(age_attribute_composition, i, "Fledgling")
            if "Hatchling" in attribute:
                attribute = attribute.replace("Hatchling", "").strip()
                age_attribute_composition = update(age_attribute_composition, i, "Hatchling")
            if "Nestling" in attribute:
                attribute = attribute.replace("Nestling", "").strip()
                age_attribute_composition = update(age_attribute_composition, i, "Nestling")
            if "Song" in attribute or "Duet" in attribute or "Call" in attribute or "Wing" in attribute:
                if "Song" in attribute:
                    temp = attribute
                    attribute = attribute.replace("Duet", "").strip()
                    attribute = attribute.replace("Call", "").strip()
                    attribute = attribute.replace("Wing", "").strip()
                    if attribute != "Song":
                        vocalisation_attribute_composition = update(vocalisation_attribute_composition, i, ("Song", attribute))
                    else:
                        vocalisation_attribute_composition = update(vocalisation_attribute_composition, i, ("Song", "Unknown"))
                    attribute = temp
                if "Call" in attribute:
                    temp = attribute
                    attribute = attribute.replace("Song", "").strip()
                    attribute = attribute.replace("Duet", "").strip()
                    attribute = attribute.replace("Wing", "").strip()
                    if attribute != "Call":
                        vocalisation_attribute_composition = update(vocalisation_attribute_composition, i, ("Call", attribute))
                    else:
                        vocalisation_attribute_composition = update(vocalisation_attribute_composition, i, ("Call", "Unknown"))
                    attribute = temp
                if "Duet" in attribute:
                    temp = attribute
                    attribute = attribute.replace("Song", "").strip()
                    attribute = attribute.replace("Call", "").strip()
                    attribute = attribute.replace("Wing", "").strip()
                    if attribute != "Duet":
                        vocalisation_attribute_composition = update(vocalisation_attribute_composition, i, ("Duet", attribute))
                    else:
                        vocalisation_attribute_composition = update(vocalisation_attribute_composition, i, ("Duet", "Unknown"))
                    attribute = temp
                if "Wing" in attribute:
                    vocalisation_attribute_composition = update(vocalisation_attribute_composition, i, ("Other", "Unknown"))
            else:
                for vocalisation_indicator in self.vocalisation_indicators:
                    if vocalisation_indicator in attribute:
                        if attribute != "Vocalisation" and attribute != "Voice" and attribute != "Noise" and attribute != "Sound":
                            vocalisation_attribute_composition = update(vocalisation_attribute_composition, i, ("Vocalisation", attribute))
                        else:
                            vocalisation_attribute_composition = update(vocalisation_attribute_composition, i, ("Vocalisation", "Unknown"))
        sexes, ages, general_vocalisations, specific_vocalisations = self.simplify_type(sex_attribute_composition, age_attribute_composition, vocalisation_attribute_composition, attribute_count)
        return sexes, ages, general_vocalisations, specific_vocalisations

    def format_type(self, lemmatizer, type_string):
        attributes = []
        if type_string != "":
            type_list = type_string.split(",")
            for type_ in type_list:
                type_ = self.strip_punctuation(type_)
                words = type_.split()
                type_ = ""
                for word in words:
                    word = self.replace_numbers(word)
                    word = self.strip_punctuation(word)
                    if word.lower() not in self.english_words:
                        if word in self.words_to_correct:
                            word = self.words_to_correct[word]
                    word = self.lemmatize(lemmatizer, word)
                    type_ = f"{type_} {word}"
                type_ = " ".join(type_.split()).strip()
                if self.is_type_certain(type_):
                    words = type_.split()
                    attribute = ""
                    for word in words:
                        sub_words = self.separate_word(word)
                        for sub_word in sub_words:
                            sub_word = self.simplify_word(sub_word)
                            sub_word = self.standardadise_word(sub_word)
                            attribute = f"{attribute} {sub_word}"
                    attribute = " ".join(attribute.split()).strip()
                    sub_attributes = self.separate_attribute(attribute)
                    for sub_attribute in sub_attributes:
                        sub_attribute = sub_attribute.strip()
                        sub_attribute = self.rearrange(sub_attribute)
                        sub_attribute = " ".join(sub_attribute.split()).strip()
                        if sub_attribute != "":
                            if self.is_attribute_relevant(sub_attribute):
                                attributes.append(sub_attribute)
        attributes = list(dict.fromkeys(attributes))
        for i in range(0, len(attributes)):
            attributes[i] = self.standardadise_attribute(attributes[i])
        return attributes

    def is_type_certain(self, type_):
        valid = True
        type_ = f" {type_.lower().title().strip()} "
        if any(f" {x} " in type_ for x in self.uncertainty_indicators):
            valid = False
        return valid

    def replace_numbers(self, word):
        for number in self.numbers_to_words.keys():
            word = word.replace(number, f" {self.numbers_to_words[number]} ")
        word = " ".join(word.split()).lower().title().strip()
        return word

    def is_attribute_relevant(self, attribute):
        valid = False
        attribute = f" {attribute.lower().title().strip()} "
        if any(f" {x} " in attribute for x in self.relevancy_indicators):
            valid = True
        if any(f" {x} " in attribute for x in self.irrelevancy_indicators):
            valid = False
        return valid

    def strip_punctuation(self, type_):
        type_ = type_.lower().title().strip()
        for punctuation in self.punctuation_to_remove:
            type_ = type_.replace(punctuation, " ")
        for punctuation in self.punctuation_to_replace.keys():
            type_ = type_.replace(punctuation, self.punctuation_to_replace[punctuation])
        type_ = unidecode.unidecode(type_)
        type_ = " ".join(type_.split())
        return type_

    def lemmatize(self, lemmatizer, word):
        word = word.lower().strip()
        word = lemmatizer.lemmatize(word, pos="v")
        word = lemmatizer.lemmatize(word, pos="n")
        word = word.title()
        if word in self.lemmatizer_corrections:
            word = self.lemmatizer_corrections[word]
        return word

    def simplify_word(self, word):
        word = word.lower().title().strip()
        if word in self.words_to_remove:
            word = ""
        return word

    def standardadise_word(self, word):
        word = word.lower().title()
        if word in self.words_to_replace:
            word = self.words_to_replace[word]
        return word

    def separate_word(self, word):
        word = f" {word.strip()} ".lower().title()
        for key in self.words_to_separate.keys():
            word = word.replace(key, self.words_to_separate[key])
            word = f" {word.strip()} ".lower().title()
        sub_words = word.split()
        for i in range(0, len(sub_words)):
            sub_words[i] = sub_words[i].lower().title().strip()
        return sub_words

    @staticmethod
    def separate_attribute(attribute):
        attribute = f" {attribute.strip()} ".lower().title()
        sub_attributes = attribute.split(" And ")
        for i in range(0, len(sub_attributes)):
            sub_attributes[i] = f" {sub_attributes[i].strip()} ".lower().title()
            sub_attributes[i] = sub_attributes[i].replace(" And ", " ")
            sub_attributes[i] = sub_attributes[i].lower().title().strip()
        return sub_attributes

    def rearrange(self, attribute):
        attribute = attribute.lower().title().strip()
        for word in self.words_to_rearrange:
            if word in attribute:
                attribute = attribute.replace(word, "")
                attribute = attribute.strip()
                attribute = f"{attribute} {word}"
        attribute = attribute.lower().title().strip()
        return attribute

    def standardadise_attribute(self, attribute):
        is_attribute_seen = False
        if attribute in self.attribute_vocabulary:
            is_attribute_seen = True
        else:
            attribute_words = attribute.split()
            standard_attributes = list(self.attribute_vocabulary)
            i = 0
            while i < len(standard_attributes) and not is_attribute_seen:
                standard_attribute = standard_attributes[i]
                standard_attribute_words = standard_attribute.split()
                if collections.Counter(attribute_words) == collections.Counter(standard_attribute_words):
                    attribute = standard_attribute
                    is_attribute_seen = True
                i += 1
        if not is_attribute_seen:
            self.attribute_vocabulary.add(attribute)
        return attribute

    # ******************************************************************************************************************
