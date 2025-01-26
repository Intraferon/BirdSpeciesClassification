import pprint
import time
from dataset_generators.constants import *
from PIL import Image
import multiprocessing
from utility import *


class ImagePreprocessor:

    def __init__(self, subset, constructor, dataset, solution_type_):
        self.image_subset_path = subset.subset_path
        self.image_data_path = subset.aligned_data_files_path
        self.constructor = constructor
        self.dataset = dataset
        self.solution_type_ = solution_type_

    class ImageConverter:

        def __init__(self, converted_image_file):
            self.converted_image_file = converted_image_file

        def __call__(self, image_id, image_path_tuple):
            old_image_path = image_path_tuple[0]
            new_image_path = image_path_tuple[1]
            image = None
            try:
                image = Image.open(old_image_path)
                width, height = image.size
                if not (width == 800 or height == 800):
                    image.thumbnail((IMAGE_MAXIMUM_DIMENSION, IMAGE_MAXIMUM_DIMENSION), Image.ANTIALIAS)
                    image = image.convert("RGB")
                    os.remove(old_image_path)
                    image.save(new_image_path, format=IMAGE_FORMAT, quality=90)
                image.close()
                self.converted_image_file.write(f"{image_id}\n")
            except Exception as e:
                if image is not None:
                    image.close()
                os.remove(old_image_path)
                print(e)

    def run(self):
        self.convert_image_format()

    def convert_image_format(self):
        image_path_tuple_list = []
        converted_image_id_list = []
        image_limit = 500000
        image_limit_reached = False
        image_i = 0
        print("Reading Images...")
        species_list = self.constructor.read_species_list(self.dataset)
        converted_images = set(read_data_from_file_(f"{self.image_subset_path}/converted.txt"))
        i = 0
        for species in species_list:
            species_path = f"{self.image_data_path}{species}/"
            observation_set = set(self.constructor.read_observation_list(self.dataset, species, "partition", self.solution_type_))
            for image_name in os.listdir(species_path):
                image_base_name = "_".join(image_name.split('.')[0].split("_")[:-1])
                image_id = f"{species}/{image_name.split('.')[0]}"
                # if image_base_name in observation_set and image_id not in converted_images:
                if image_id not in converted_images:
                    old_image_path = f"{species_path}{image_name}"
                    new_image_path = f"{species_path}{image_name.split('.')[0]}.{IMAGE_FORMAT}"
                    image_path_tuple_list.append((old_image_path, new_image_path))
                    converted_image_id_list.append(image_id)
                    image_i += 1
                    if image_i > image_limit:
                        image_limit_reached = True
                        break
            print(f"Species: {i}")
            i += 1
            if image_limit_reached:
                break
        print("Converting Images...")
        converted_image_file = open(f"{self.image_subset_path}/converted.txt", "a")
        image_converter = self.ImageConverter(converted_image_file)
        start = time.time()
        for i in range(len(converted_image_id_list)):
            image_id = converted_image_id_list[i]
            image_path_tuple = image_path_tuple_list[i]
            image_converter(image_id, image_path_tuple)
            if i % 50 == 0:
                print(f"Image: {i}")
        end = time.time()
        print(end - start)
        converted_image_file.close()

