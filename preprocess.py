from subsets import *
from datasets import *
from partition_configurations import *


def preprocess_inaturalist_data(constructor, dataset_id):
    inaturalist = get_inaturalist("Image")
    dataset = constructor.get_dataset_handle(dataset_id[0], dataset_id[1], dataset_id[2], dataset_type_="partition")
    solution_type_ = determine_partition_solution_type(dataset_id_[0])
    inaturalist.preprocess_data(constructor=constructor, dataset=dataset, solution_type_=solution_type_)


def preprocess_xenocanto_data(parameters, constructor, dataset_id):
    xenocanto = get_xenocanto()
    dataset = constructor.get_dataset_handle(dataset_id[0], dataset_id[1], dataset_id[2], dataset_type_="partition")
    solution_type_ = determine_partition_solution_type(dataset_id_[0])
    xenocanto.preprocess_data(parameters, constructor=constructor, dataset=dataset, solution_type_=solution_type_)


if __name__ == "__main__":

    # constructor_ = get_inaturalist_and_xenocanto_1000_dataset_constructor()
    # dataset_id_ = ("all_0_3_0", "image", "train")
    # preprocess_inaturalist_data(constructor_, dataset_id_)

    parameters_ = {"segment_signal": False,
                   "segment_noise": False,
                   "load_signal": False,
                   "load_noise": True}

    constructor_ = get_inaturalist_and_xenocanto_1000_dataset_constructor()

    dataset_id_ = ("random_0", "audio", "train")
    preprocess_xenocanto_data(parameters_, constructor_, dataset_id_)
    #
    # dataset_id_ = ("random_0", "audio", "validation")
    # preprocess_xenocanto_data(parameters_, constructor_, dataset_id_)
    #
    # dataset_id_ = ("random_0", "audio", "test")
    # preprocess_xenocanto_data(parameters_, constructor_, dataset_id_)

    dataset_id_ = ("random_1", "audio", "train")
    preprocess_xenocanto_data(parameters_, constructor_, dataset_id_)
    #
    # dataset_id_ = ("random_1", "audio", "validation")
    # preprocess_xenocanto_data(parameters_, constructor_, dataset_id_)
    #
    # dataset_id_ = ("random_1", "audio", "test")
    # preprocess_xenocanto_data(parameters_, constructor_, dataset_id_)

    dataset_id_ = ("random_2", "audio", "train")
    preprocess_xenocanto_data(parameters_, constructor_, dataset_id_)

    # dataset_id_ = ("random_2", "audio", "validation")
    # preprocess_xenocanto_data(parameters_, constructor_, dataset_id_)
    #
    # dataset_id_ = ("random_2", "audio", "test")
    # preprocess_xenocanto_data(parameters_, constructor_, dataset_id_)

    dataset_id_ = ("all_0_3_0", "audio", "train")
    preprocess_xenocanto_data(parameters_, constructor_, dataset_id_)

    # dataset_id_ = ("all_0_3_0", "audio", "validation")
    # preprocess_xenocanto_data(parameters_, constructor_, dataset_id_)
    #
    # dataset_id_ = ("all_0_3_0", "audio", "test")
    # preprocess_xenocanto_data(parameters_, constructor_, dataset_id_)



