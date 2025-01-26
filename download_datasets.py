from subsets import *
from datasets import *


def download_data(subset, constructor, dataset_id):
    print("Updating progess...")
    subset_.update_progress("data")
    # dataset = constructor.get_dataset_handle(version_configuration=dataset_id[0], modality=dataset_id[1], partition=dataset_id[2], dataset_type_=dataset_id[3])
    # source_metadata = constructor.get_source_metadata_handle(version_configuration=dataset_id[0], modality=dataset_id[1], partition=dataset_id[2], dataset_type_=dataset_id[3])
    # subset.download_source_data(constructor, dataset, source_metadata, dataset_type_=dataset_id[3], solution_type_=dataset_id[4])

# iNaturalist random_0: Done
# iNaturalist random_1: Done
# iNaturalist random_2: Done
# iNaturalist context_123_3_0_0_0: Done
# iNaturalist all_0_1_0_0_0: Done
# iNaturalist all_0_3_0_0_0: Done

subset_ = get_xenocanto()
constructor_ = get_inaturalist_and_xenocanto_1000_dataset_constructor()

if __name__ == "__main__":

    dataset_id_ = ("all_0_3_0", "image", "validation", "partition", "best")
    download_data(subset_, constructor_, dataset_id_)

    # dataset_id_ = ("context_123_3_0_0_0", "image", "validation", "partition", "best")
    # download_data(subset_, constructor_, dataset_id_)
    #
    # dataset_id_ = ("context_123_3_0_0_0", "image", "test", "partition", "best")
    # download_data(subset_, constructor_, dataset_id_)




