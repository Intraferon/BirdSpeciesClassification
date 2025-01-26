from subset_generators.gbif import GBIF
from utility import *
import requests
import mimetypes
mimetypes.init()
from preprocessors.image_preprocessor import ImagePreprocessor
from concurrent.futures import ThreadPoolExecutor


# Dataset Update Date: 9 September 2021
# Metadata Update Date: 19 October 2021


class ObservationOrg(GBIF):

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self, base_taxonomy, subset_taxonomy_to_base_taxonomy_replacements, aligned_taxonomy, base_taxonomy_to_aligned_taxonomy_replacements):

        # Paths
        subset_path = f"C:/Users/sanam/Documents/Masters/Resources/Citizen Science Projects/Observation Org"

        # Tags
        subset_tag = project_tag = "observationorg"

        # Parent Constructor
        super(ObservationOrg, self).__init__(subset_path, subset_tag, project_tag,
                                             base_taxonomy, subset_taxonomy_to_base_taxonomy_replacements, aligned_taxonomy, base_taxonomy_to_aligned_taxonomy_replacements)

    # ******************************************************************************************************************

    def download_source_data(self, constructor, dataset, source_metadata, dataset_type_=None, solution_type_=""):

        valid_data_format_list = []
        for data_format in mimetypes.types_map:
            if mimetypes.types_map[data_format].split("/")[0] == "image":
                valid_data_format_list.append(data_format.replace(".", ""))

        def download_batch(x):
            media_url_ = x[0]
            data_id_ = x[1]
            try:
                request_ = requests.get(media_url_)
                data_url_ = request_.url
                data_format_ = data_url_.split(".")[-1].lower()
                if data_format_ in valid_data_format_list:
                    data_file_path_ = f"{self.aligned_data_files_path}{data_id_}.{data_format_}"
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
        batch_limit = 20
        batch_limit_reached = False
        metadata_limit = 1000000
        with ThreadPoolExecutor(max_workers=50) as pool:
            for species in species_list:
                species_path = f"{self.aligned_data_files_path}{species}"
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
                            if media_types[k] == "Image":
                                if request_list_size < request_size:
                                    data_id = f"{species}/{id_}_{j}"
                                    media_url = media_urls[k]
                                    if data_id not in downloaded_files:
                                        data_id_list.append(data_id)
                                        media_url_list.append(media_url)
                                        request_list_size += 1
                                if request_list_size == request_size:
                                    downloaded_data_id_list = list(pool.map(download_batch, zip(media_url_list, data_id_list)))
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
            downloaded_data_id_list = list(pool.map(download_batch, zip(media_url_list, data_id_list)))
            downloaded_files.update([downloaded_data_id for downloaded_data_id in downloaded_data_id_list if downloaded_data_id is not None])
        save_data_to_file_(f"{self.subset_path}/downloaded.txt", downloaded_files)

    def preprocess_data(self):
        image_preprocessor = ImagePreprocessor(self.subset_path)
        image_preprocessor.run()
