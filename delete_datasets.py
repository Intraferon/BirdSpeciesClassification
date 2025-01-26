from datasets import *


def delete_derivative_datasets():

    get_inaturalist_and_xenocanto_1000_dataset_constructor().delete()
    get_inaturalist_and_xenocanto_1000_dataset_constructor().delete(dataset_type_="partition")
    get_inaturalist_and_xenocanto_1000_dataset_constructor().delete(dataset_type_="pair")

    get_inaturalist_and_xenocanto_300_dataset_constructor().delete()
    get_inaturalist_and_xenocanto_300_dataset_constructor().delete(dataset_type_="partition")


if __name__ == "__main__":
    delete_derivative_datasets()
