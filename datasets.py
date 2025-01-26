from dataset_generators.constructor import Constructor


def get_inaturalist_and_xenocanto_dataset_constructor():
    inaturalist_and_xenocanto_dataset_constructor = Constructor("iNaturalist and Xeno Canto", "inaturalist_and_xenocanto_dataset")
    return inaturalist_and_xenocanto_dataset_constructor


def get_inaturalist_and_xenocanto_300_dataset_constructor():
    inaturalist_and_xenocanto_300_dataset_constructor = Constructor("iNaturalist and Xeno Canto (300)", "inaturalist_and_xenocanto_300_dataset")
    return inaturalist_and_xenocanto_300_dataset_constructor


def get_inaturalist_and_xenocanto_1000_dataset_constructor():
    inaturalist_and_xenocanto_1000_dataset_constructor = Constructor("iNaturalist and Xeno Canto (1000)", "inaturalist_and_xenocanto_1000_dataset")
    return inaturalist_and_xenocanto_1000_dataset_constructor


