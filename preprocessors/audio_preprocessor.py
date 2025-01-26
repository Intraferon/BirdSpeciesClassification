import pprint

from dataset_generators.constants import *
import audiofile
from features.spectrogram import Spectrogram
import ffmpeg
from PIL import Image
import albumentations
import cv2
import shutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from utility import *
import itertools
import numpy as np
import struct
import time
import librosa
import warnings


class AudioPreprocessor:

    def __init__(self, subset, constructor, dataset, solution_type_):
        self.audio_subset_path = subset.subset_path
        self.audio_data_path = subset.aligned_data_files_path
        self.external_audio_data_path = subset.external_aligned_data_files_path
        self.signal_data_path = subset.aligned_signal_files_path
        self.external_signal_data_path = subset.external_aligned_signal_files_path
        self.noise_data_path = subset.aligned_noise_files_path
        self.external_noise_data_path = subset.external_aligned_noise_files_path
        self.spectrogram_data_path = subset.aligned_spectrogram_files_path
        self.constructor = constructor
        self.dataset = dataset
        self.solution_type_ = solution_type_

    def run(self, parameters):
        warnings.simplefilter("error")
        if parameters["segment_signal"]:
            self.segment_signal()
        if parameters["segment_noise"]:
            self.segment_noise()
        if parameters["load_signal"]:
            self.load_signal()
        if parameters["load_noise"]:
            self.load_noise()

    def segment_signal(self):
        audio_limit = 10000
        audio_limit_reached = False
        i = 0
        audio_path_tuple_list = []
        create_folder_(self.external_noise_data_path)
        print("Reading Audio...")
        species_list = self.constructor.read_species_list(self.dataset)
        for species in species_list:
            audio_species_path = f"{self.external_audio_data_path}{species}/"
            signal_species_path = f"{self.external_signal_data_path}{species}/"
            create_folder_(audio_species_path)
            create_folder_(signal_species_path)
            observation_set = set(self.constructor.read_observation_list(self.dataset, species, "partition", self.solution_type_))
            for audio_name in os.listdir(audio_species_path):
                audio_base_name = f"{'_'.join(audio_name.split('_')[:-1])}"
                if audio_base_name in observation_set:
                    audio_path = f"{audio_species_path}{audio_name}"
                    signal_path = f"{signal_species_path}{audio_name.split('.')[0]}.{TRAIN_AUDIO_FORMAT}"
                    if os.path.exists(audio_path) and not os.path.exists(signal_path):
                        temp_audio_path = f"{audio_species_path}{audio_name.split('.')[0]}_temp.{TRAIN_AUDIO_FORMAT}"
                        audio_path_tuple_list.append((audio_path, temp_audio_path, signal_path))
                        i += 1
                        if i > audio_limit:
                            audio_limit_reached = True
                            break
            print(f"Species: {i}")
            i += 1
            if audio_limit_reached:
                break
        print("Segmenting Audios...")
        spectrogram = Spectrogram()
        signal_segmenter = self.SignalSegmenter(spectrogram)
        start = time.time()
        for i, audio_path_tuple in enumerate(audio_path_tuple_list):
            signal_segmenter(audio_path_tuple)
            if i % 1 == 0:
                print(f"Audio: {i}")
        end = time.time()
        print(end - start)

    class SignalSegmenter:

        def __init__(self, spectogram):
            self.spectrogram = spectogram

        def __call__(self, audio_path_tuple):
            audio_path = audio_path_tuple[0]
            temp_audio_path = audio_path_tuple[1]
            signal_path = audio_path_tuple[2]
            success = True
            try:
                input_ = ffmpeg.input(audio_path)
                audio = input_.audio
                audio = ffmpeg.output(audio, temp_audio_path, **{"ar": str(AUDIO_SAMPLE_RATE), "format": TRAIN_AUDIO_FORMAT, "ac": 1})
                ffmpeg.run(audio, capture_stdout=False, capture_stderr=True, overwrite_output=True)
                audio = audiofile.read(temp_audio_path)[0]
                signal_indicator = self.spectrogram.extract_indicator(audio, extract_noise=False)[0]
                signal_audio = self.spectrogram.segment_audio(audio, signal_indicator)
                audiofile.write(signal_path, signal_audio, AUDIO_SAMPLE_RATE)
            except RuntimeWarning as e:
                print("warning: divide by zero encountered")
                success = False
            except Exception as e:
                print(f"exception: {e}")
                success = False
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                if not success:
                    os.remove(audio_path)

    def segment_noise(self):
        audio_limit = 10000
        audio_limit_reached = False
        i = 0
        audio_i = 0
        audio_path_tuple_list = []
        create_folder_(self.external_noise_data_path)
        print("Reading Audio...")
        species_list = self.constructor.read_species_list(self.dataset)
        for species in species_list:
            audio_species_path = f"{self.external_audio_data_path}/{species}/"
            noise_species_path = f"{self.external_noise_data_path}/{species}/"
            create_folder_(noise_species_path)
            observation_set = set(self.constructor.read_observation_list(self.dataset, species, "partition", self.solution_type_))
            for audio_name in os.listdir(audio_species_path):
                audio_base_name = f"{audio_name.split('_')[0]}_{audio_name.split('_')[1]}"
                if audio_base_name in observation_set:
                    audio_id = f"{species}/{audio_name.split('.')[0]}"
                    audio_path = f"{audio_species_path}{audio_name}"
                    noise_path = f"{noise_species_path}{audio_name.split('.')[0]}.{TRAIN_AUDIO_FORMAT}"
                    if os.path.exists(audio_path) and not os.path.exists(noise_path):
                        temp_audio_path = f"{audio_species_path}{audio_name.split('.')[0]}_temp.{TRAIN_AUDIO_FORMAT}"
                        audio_path_tuple_list.append((audio_path, temp_audio_path, noise_path))
                        audio_i += 1
                        if audio_i > audio_limit:
                            audio_limit_reached = True
                            break
            print(f"Species: {i}")
            i += 1
            if audio_limit_reached:
                break
        print("Segmenting Audios...")
        spectrogram = Spectrogram()
        noise_segmenter = self.NoiseSegmenter(spectrogram)
        start = time.time()
        for i, audio_path_tuple in enumerate(audio_path_tuple_list):
            noise_segmenter(audio_path_tuple)
            if i % 1 == 0:
                print(f"Audio: {i}")
        end = time.time()
        print(end - start)

    class NoiseSegmenter:

        def __init__(self, spectogram):
            self.spectrogram = spectogram

        def __call__(self, audio_path_tuple):
            audio_path = audio_path_tuple[0]
            temp_audio_path = audio_path_tuple[1]
            noise_path = audio_path_tuple[2]
            success = True
            try:
                input_ = ffmpeg.input(audio_path)
                audio = input_.audio
                audio = ffmpeg.output(audio, temp_audio_path, **{"ar": str(AUDIO_SAMPLE_RATE), "format": TRAIN_AUDIO_FORMAT, "ac": 1})
                ffmpeg.run(audio, capture_stdout=False, capture_stderr=True, overwrite_output=True)
                audio = audiofile.read(temp_audio_path)[0]
                noise_indicator = self.spectrogram.extract_indicator(audio, extract_noise=True)[1]
                if noise_indicator is not None:
                    noise_audio = self.spectrogram.segment_audio(audio, noise_indicator, extract_noise=True)
                    audiofile.write(noise_path, noise_audio, AUDIO_SAMPLE_RATE)
                else:
                    success = False
            except RuntimeWarning as e:
                print("warning: divide by zero encountered")
                success = False
            except Exception as e:
                print(f"exception: {e}")
                success = False
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

    def load_signal(self):
        audio_limit = 30000
        audio_limit_reached = False
        audio_i = 0
        i = 0
        create_folder_(self.signal_data_path)
        print("Reading Audio...")
        species_list = self.constructor.read_species_list(self.dataset)
        for species in species_list:
            signal_species_path = f"{self.signal_data_path}/{species}/"
            external_signal_species_path = f"{self.external_signal_data_path}/{species}/"
            create_folder_(signal_species_path)
            observation_set = set(self.constructor.read_observation_list(self.dataset, species, "partition", self.solution_type_))
            for signal_name in os.listdir(external_signal_species_path):
                signal_base_name = f"{signal_name.split('_')[0]}_{signal_name.split('_')[1]}"
                if signal_base_name in observation_set:
                    signal_path = f"{signal_species_path}{signal_name}"
                    external_signal_path = f"{external_signal_species_path}{signal_name}"
                    if not os.path.exists(signal_path):
                        shutil.copy(external_signal_path, signal_path)
                        audio_i += 1
                        if audio_i % 50 == 0:
                            print(f"Audio: {audio_i}")
                        if audio_i > audio_limit:
                            audio_limit_reached = True
                            break
            i += 1
            if audio_limit_reached:
                break

    def load_noise(self):
        audio_limit = 15000
        audio_limit_reached = False
        audio_i = 0
        i = 0
        j = 0
        create_folder_(self.noise_data_path)
        print("Reading Audio...")
        species_list = self.constructor.read_species_list(self.dataset)
        for species in species_list:
            noise_species_path = f"{self.noise_data_path}/{species}/"
            external_noise_species_path = f"{self.external_noise_data_path}/{species}/"
            create_folder_(noise_species_path)
            observation_set = set(self.constructor.read_observation_list(self.dataset, species, "partition", self.solution_type_))
            for noise_name in os.listdir(external_noise_species_path):
                noise_base_name = f"{noise_name.split('_')[0]}_{noise_name.split('_')[1]}"
                if noise_base_name in observation_set:
                    noise_path = f"{noise_species_path}{noise_name}"
                    external_noise_path = f"{external_noise_species_path}{noise_name}"
                    if not os.path.exists(noise_path):
                        j += 1
                        if j % 5 == 0:
                            shutil.copy(external_noise_path, noise_path)
                            audio_i += 1
                            if audio_i % 1 == 0:
                                print(f"Audio: {audio_i}")
                    if audio_i > audio_limit:
                        audio_limit_reached = True
                        break
            i += 1
            if audio_limit_reached:
                break


