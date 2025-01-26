from taxonomy_generators.taxonomy import Taxonomy
from subset_generators.inaturalist import INaturalist
from subset_generators.xeno_canto import XenoCanto
from subset_generators.observation_org import ObservationOrg
from dataset_generators.constants import *
from utility import *


def get(subset_name):
    if subset_name == "xenocanto":
        subset = get_xenocanto()
    elif subset_name == "observationorg":
        subset = get_observationorg()
    elif subset_name == "inaturalist_audio":
        subset = get_inaturalist("Audio")
    elif subset_name == "inaturalist_image":
        subset = get_inaturalist("Image")
    else:
        subset = None
    return subset


def get_xenocanto():
    xenocanto_base_taxonomy = merge_list_dictionaries_([Taxonomy.get_aligned("ioc91", ["other"]), {"Dicaeum Dayakorum": []}])
    xenocanto_base_taxonomy_replacements = [{}]
    aligned_taxonomy = Taxonomy.get_aligned("clements2021", ["clements2019", "ioc91", "ioc112"])
    xenocanto_aligned_taxonomy_replacements = [Taxonomy.get_replacement("ioc91", "clements2021")]
    xenocanto = XenoCanto(xenocanto_base_taxonomy, xenocanto_base_taxonomy_replacements, aligned_taxonomy, xenocanto_aligned_taxonomy_replacements)
    return xenocanto


def prepare_xenocanto():
    xenocanto = get_xenocanto()
    xenocanto.extract_downloaded_project_taxonomy()
    Taxonomy.compare_project_to_source("xenocanto", "ioc91", ["ioc112", "other"], Taxonomy.downloaded_project_files_path)
    xenocanto.edit_source_metadata()
    xenocanto.extract_project_taxonomy(is_aligned=False)
    Taxonomy.compare_project_to_source("xenocanto", "ioc91", ["ioc112", "other"], Taxonomy.base_project_files_path)
    xenocanto.align_source_metadata()
    xenocanto.extract_project_taxonomy(is_aligned=True)
    Taxonomy.compare_project_to_source("xenocanto", "clements2021", ["ioc112"] + read_data_from_file_(f"{RESOURCE_PATH}arrays/ioc_taxonomies.txt"), Taxonomy.aligned_project_files_path)


def get_observationorg():
    observationorg_base_taxonomy = Taxonomy.get_aligned("ioc112", ["other"] + read_data_from_file_(f"{RESOURCE_PATH}arrays/observationorg_ioc_taxonomies.txt"))
    observationorg_base_taxonomy_replacements = Taxonomy.merge_taxonomy_replacements([{"other": "ioc112"}] + [{x: "ioc112"} for x in read_data_from_file_(f"{RESOURCE_PATH}arrays/observationorg_ioc_taxonomies.txt")])
    aligned_taxonomy = Taxonomy.get_aligned("clements2021", ["clements2019", "ioc91", "ioc112"])
    observationorg_aligned_taxonomy_replacements = [Taxonomy.get_replacement("ioc112", "clements2021")]
    observationorg = ObservationOrg(observationorg_base_taxonomy, observationorg_base_taxonomy_replacements, aligned_taxonomy, observationorg_aligned_taxonomy_replacements)
    return observationorg


def prepare_observationorg():
    observationorg = get_observationorg()
    observationorg.extract_downloaded_project_taxonomy()
    Taxonomy.compare_project_to_source("observationorg", "ioc112", ["other"] + read_data_from_file_(f"{RESOURCE_PATH}arrays/ioc_taxonomies.txt")[1:], Taxonomy.downloaded_project_files_path)
    observationorg.edit_source_metadata()
    observationorg.enrich_source_metadata()
    observationorg.extract_project_taxonomy(is_aligned=False)
    Taxonomy.compare_project_to_source("observationorg", "ioc112", ["other"] + read_data_from_file_(f"{RESOURCE_PATH}arrays/ioc_taxonomies.txt")[1:], Taxonomy.base_project_files_path)
    observationorg.align_source_metadata()
    observationorg.extract_project_taxonomy(is_aligned=True)
    Taxonomy.compare_project_to_source("observationorg", "clements2021", ["ioc112"] + read_data_from_file_(f"{RESOURCE_PATH}arrays/ioc_taxonomies.txt")[1:], Taxonomy.aligned_project_files_path)


def get_inaturalist(media):
    inaturalist_base_taxonomy = merge_list_dictionaries_([Taxonomy.get_aligned("clements2019", ["other"]), {"Heliothraupis Oneilli": []}])
    inaturalist_base_taxonomy_replacements = [Taxonomy.get_replacement("other", "clements2019")]
    aligned_taxonomy = Taxonomy.get_aligned("clements2021", ["clements2019", "ioc91", "ioc112"])
    inaturalist_aligned_taxonomy_replacements = [Taxonomy.get_replacement("clements2019", "clements2021")]
    inaturalist = INaturalist(media, inaturalist_base_taxonomy, inaturalist_base_taxonomy_replacements, aligned_taxonomy, inaturalist_aligned_taxonomy_replacements)
    return inaturalist


def prepare_inaturalist(media):
    inaturalist = get_inaturalist(media)
    inaturalist.extract_downloaded_project_taxonomy()
    Taxonomy.compare_project_to_source(f"inaturalist_{media}", "clements2019", ["other", "clements2021", "clements2018", "clements2017", "clements2016", "clements2015"], Taxonomy.downloaded_project_files_path)
    inaturalist.edit_source_metadata()
    inaturalist.enrich_source_metadata()
    inaturalist.extract_project_taxonomy(is_aligned=False)
    Taxonomy.compare_project_to_source(f"inaturalist_{media}", "clements2019", ["other", "clements2021", "clements2018", "clements2017", "clements2016", "clements2015"], Taxonomy.base_project_files_path)
    inaturalist.align_source_metadata()
    inaturalist.extract_project_taxonomy(is_aligned=True)
    Taxonomy.compare_project_to_source(f"inaturalist_{media}", "clements2021", ["clements2019", "clements2018", "clements2017", "clements2016", "clements2015"], Taxonomy.aligned_project_files_path)


if __name__ == "__main__":
    print("Hello")




