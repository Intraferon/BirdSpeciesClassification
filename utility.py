import os
import json
import csv
import itertools
import math
from collections.abc import Iterable


def create_folder_(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def read_data_from_file_(file_path):
    file_type = file_path.split(".")[1]
    with open(file_path, "r") as file:
        if file_type == "json":
            data = json.load(file)
        elif file_type == "txt":
            data = list(file.read().splitlines())
        elif file_type == "csv":
            data = list(csv.reader(file))
        elif file_type == "log":
            data = list(csv.reader(file))
        else:
            data = None
    return data


def save_data_to_file_(file_path, data):
    file_type = file_path.split(".")[1]
    with open(file_path, "w", newline="") as file:
        if file_type == "json":
            file.write(json.dumps(data, indent=4, sort_keys=True))
        elif file_type == "txt":
            file.write("\n".join(data))
        elif file_type == "csv":
            writer = csv.writer(file)
            writer.writerows(data)
        else:
            file.write(data)


def update_frequency_dictionary_(key, value, frequency_dictionary):
    if key in frequency_dictionary:
        frequency_dictionary[key] += value
    else:
        frequency_dictionary[key] = value
    return frequency_dictionary


def merge_frequency_dictionaries_(frequency_dictionary_list):
    merged_frequency_dictionary = {}
    for dictionary in frequency_dictionary_list:
        for key in dictionary:
            merged_frequency_dictionary[key] = 0
    for key in merged_frequency_dictionary:
        for dictionary in frequency_dictionary_list:
            if key in dictionary:
                merged_frequency_dictionary[key] += dictionary[key]
    return merged_frequency_dictionary


def merge_list_dictionaries_(list_dictionary_list):
    merged_list_dictionary = {}
    for list_dictionary in list_dictionary_list:
        for key in list_dictionary:
            if key in merged_list_dictionary:
                merged_list_dictionary[key].extend(list_dictionary[key])
            else:
                merged_list_dictionary[key] = list_dictionary[key]
    return merged_list_dictionary


def flatten(multidimensional_list, element_type):
    for element in multidimensional_list:
        if isinstance(element, Iterable) and not isinstance(element, element_type):
            yield from flatten(element, element_type)
        else:
            yield element


def determine_powerset(pool):
    powerset = list(itertools.chain.from_iterable(itertools.combinations(pool, r) for r in range(len(pool) + 1)))[1:]
    return powerset


def group_consecutive_data_(data):
    data = iter(data)
    val = next(data)
    chunk = []
    try:
        while True:
            chunk.append(val)
            val = next(data)
            if val != chunk[-1] + 1:
                yield chunk
                chunk = []
    except StopIteration:
        if chunk:
            yield chunk


def central_linspace(a, n):

    b = []

    def determine_b(x, n):
        if n % 2 != 0:
            i = int(len(x) / 2)
            b.append(x[i])
        if n == 1:
            return
        else:
            mid = int(math.floor(len(x) / 2))
            n = int(n / 2)
            determine_b(x[:mid], n)
            determine_b(x[mid:], n)

    if n < len(a):
        determine_b(a, n)
        b.sort()
    else:
        b = a

    return b



