from datasets import *
from subsets import *


def create_inaturalist_and_xenocanto_dataset():
    inaturalist = get_inaturalist("Image")
    xenocanto = get_xenocanto()
    master_database_structure = {"image": {"train-validation-test": inaturalist.aligned_metadata_database_name},
                                 "audio": {"train-validation-test": xenocanto.aligned_metadata_database_name}}
    get_inaturalist_and_xenocanto_dataset_constructor().configure_database(master_database_structure)


def create_inaturalist_and_xenocanto_derivative_datasets():
    inaturalist = get_inaturalist("Image")
    xenocanto = get_xenocanto()
    master_database_structure = {"image": {"train-validation-test": inaturalist.aligned_metadata_database_name},
                                 "audio": {"train-validation-test": xenocanto.aligned_metadata_database_name}}
    source_dataset_structure = {"image": {"train-validation-test": get_inaturalist_and_xenocanto_dataset_constructor().get_dataset_handle(version_configuration=3, modality="image", partition="train-validation-test")},
                                "audio": {"train-validation-test": get_inaturalist_and_xenocanto_dataset_constructor().get_dataset_handle(version_configuration=3, modality="audio", partition="train-validation-test")}}
    get_inaturalist_and_xenocanto_300_dataset_constructor().configure_database(master_database_structure, version_info="Source: iNaturalist and Xeno Canto", source_dataset_structure=source_dataset_structure)
    get_inaturalist_and_xenocanto_1000_dataset_constructor().configure_database(master_database_structure, version_info="Source: iNaturalist and Xeno Canto", source_dataset_structure=source_dataset_structure)


if __name__ == "__main__":
    create_inaturalist_and_xenocanto_derivative_datasets()
