from selenium import webdriver
from bs4 import BeautifulSoup
import time
from subset_generators.subset import Subset
from utility import *


class SensitiveSpecies:

    # Paths
    sensitive_species_path = "C:/Users/sanam/Documents/Masters/Resources/Sensitive Species"
    downloaded_file_path = f"{sensitive_species_path}/downloaded.txt"

    # ************************************************ MAJOR METHODS ***************************************************

    def download(self):
        sensitive_species_set = set()
        driver = webdriver.Chrome("C:/Users/sanam/Documents/Installers/chrome_driver/chromedriver.exe")
        driver.get("https://support.ebird.org/en/support/solutions/articles/48000803210-sensitive-species-in-ebird#Sensitive-Species-List")
        time.sleep(15)
        html_structure = BeautifulSoup(driver.page_source, "html.parser")
        table_structure_list = html_structure.find_all("tbody")[:2]
        t = 0
        for table_structure in table_structure_list:
            rows = table_structure.find_all("tr")
            for i in range(0, len(rows)):
                row = rows[i]
                cells = row.find_all("td")
                scientific_name = cells[2].string if t == 0 else cells[1].string
                sensitive_species = Subset.separate_scientific_name(scientific_name)[0]
                if sensitive_species != "":
                    sensitive_species_set.add(sensitive_species)
            t += 1
        print(len(sensitive_species_set))
        save_data_to_file_(self.downloaded_file_path, list(sensitive_species_set))

    # ************************************************ GETTER METHODS **************************************************

    def get(self):
        sensitive_species_set = set(read_data_from_file_(self.downloaded_file_path))
        return sensitive_species_set

    # ******************************************************************************************************************
