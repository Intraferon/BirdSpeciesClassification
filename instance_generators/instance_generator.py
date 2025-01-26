from tensorflow import keras
from sklearn import preprocessing
from sklearn import model_selection
from dataset_generators.constants import *
from features.spectrogram import Spectrogram
import math
import albumentations
import cv2
import random
import numpy as np
from PIL import Image
from utility import *
import soundfile
import itertools
import pprint


class InstanceGenerator(keras.utils.Sequence):

    def __init__(self, subset, modality, partition, species_count, epoch_count, epoch_cycle_length, **kwargs):

        self.subset = subset
        self.modality = modality
        self.partition = partition
        self.species_count = species_count
        self.epoch_count = epoch_count
        self.epoch_cycle_length = epoch_cycle_length

        self.batch_offset = 0
        self.batch_count = 0

        self.observation_species_list = kwargs["observation_species_list"]
        self.observation_context_list = kwargs.get("observation_context_list", None)
        self.observation_image_name_list = kwargs.get("observation_image_name_list", None)
        self.observation_image_no_list = kwargs.get("observation_image_no_list", None)
        self.observation_image_context_list = kwargs.get("observation_image_context_list", None)
        self.observation_signal_name_list = kwargs.get("observation_signal_name_list", None)
        self.observation_signal_no_list = kwargs.get("observation_signal_no_list", None)
        self.observation_signal_length_list = kwargs.get("observation_signal_length_list", None)
        self.observation_signal_context_list = kwargs.get("observation_signal_context_list", None)
        self.instance_noise_name_list = kwargs.get("instance_noise_name_list", None)
        self.instance_noise_length_list = kwargs.get("instance_noise_length_list", None)
        label_encoder = preprocessing.LabelEncoder()
        label_encoder = label_encoder.fit(self.observation_species_list)
        self.observation_label_list = label_encoder.transform(self.observation_species_list)
        self.observation_count = len(self.observation_species_list)
        self.observation_id_list = None if self.partition != "test" else list(range(self.observation_count))
        self.species_list = label_encoder.classes_

        self.instance_count = None
        self.instance_species_list = None
        self.instance_label_list = None
        self.instance_context_list = None

        self.instance_id_list = None
        self.instance_image_name_list = None
        self.instance_signal_name_list = None
        self.instance_signal_bound_list = None
        self.instance_signal_length_list = None

        self.instance_i_list = None
        self.kth_instance_i_list = None
        self.kth_instance_count = None
        self.kth_batch_count = None

        self.observation_signal_count_dictionary = None
        self.observation_signal_i_dictionary = None
        self.observation_signal_bound_dictionary = None

        self.instance_species_length = 0
        self.instance_image_name_length = 0
        self.instance_signal_name_length = 0

        if self.observation_image_name_list is not None or self.observation_signal_name_list is not None:

            if self.modality == "image":
                self.image_dataset_path = subset.aligned_data_files_path
                self.batch_size = IMAGE_BATCH_SIZE

            elif self.modality == "audio":
                self.signal_dataset_path = subset.aligned_signal_files_path
                self.noise_dataset_path = subset.aligned_noise_files_path
                self.batch_size = AUDIO_BATCH_SIZE

            elif self.modality == "image-audio":
                self.image_dataset_path = subset[0].aligned_data_files_path
                self.signal_dataset_path = subset[1].aligned_signal_files_path
                self.noise_dataset_path = subset[1].aligned_noise_files_path
                self.batch_size = JOINT_BATCH_SIZE

        else:

            self.batch_size = CONTEXT_BATCH_SIZE

        if self.modality == "image" or self.modality == "image-audio":
            self.augment_image = self.create_image_augment_transform()
            self.resize_image = self.create_image_resize_transform

        if self.modality == "audio" or self.modality == "image-audio":
            self.spectrogram = Spectrogram()
            self.resize_spectrogram = self.create_spectrogram_resize_transform()

        if self.partition == "train":
            self.on_epoch_end()
        else:
            self.construct_dataset()
            self.kth_instance_count = self.instance_count
            self.kth_batch_count = int(math.ceil(self.kth_instance_count / self.batch_size))
            self.kth_instance_i_list = [i for i in range(self.kth_instance_count)]

    # --------------------------------------------------- Generator ---------------------------------------------------------------

    def __len__(self):
        return self.kth_batch_count

    def __getitem__(self, batch_index):

        start = batch_index * self.batch_size
        end = min(start + self.batch_size, self.kth_instance_count)
        batch_size = end - start

        y_label = self.get_y_label(start, end)

        if self.instance_image_name_list is None and self.instance_signal_name_list is None:

            x_context = self.get_x_context(start, end)

            if self.instance_id_list is None:
                return x_context, y_label

            else:
                y_id = self.get_y_id(start, end)
                return x_context, y_id, y_label

        elif self.instance_image_name_list is None:

            x_spectrogram = self.get_x_spectrogram(batch_size, start, end)

            if self.instance_context_list is None and self.instance_id_list is None:
                return x_spectrogram, y_label

            elif self.instance_context_list is None:
                y_id = self.get_y_id(start, end)
                return x_spectrogram, y_id, y_label

            elif self.instance_id_list is None:
                x_context = self.get_x_context(start, end)
                return [x_spectrogram, x_context], y_label

            else:
                y_id = self.get_y_id(start, end)
                x_context = self.get_x_context(start, end)
                return [x_spectrogram, x_context], y_id, y_label

        elif self.instance_signal_name_list is None:

            x_image = self.get_x_image(batch_size, start, end)

            if self.instance_context_list is None and self.instance_id_list is None:
                return x_image, y_label

            elif self.instance_context_list is None:
                y_id = self.get_y_id(start, end)
                return x_image, y_id, y_label

            elif self.instance_id_list is None:
                x_context = self.get_x_context(start, end)
                return [x_image, x_context], y_label

            else:
                y_id = self.get_y_id(start, end)
                x_context = self.get_x_context(start, end)
                return [x_image, x_context], y_id, y_label

        else:

            x_spectrogram = self.get_x_spectrogram(batch_size, start, end)
            x_image = self.get_x_image(batch_size, start, end)

            if self.instance_context_list is None and self.instance_id_list is None:
                return [x_image, x_spectrogram], y_label

            elif self.instance_context_list is None:
                y_id = self.get_y_id(start, end)
                return [x_image, x_spectrogram], y_id, y_label

            elif self.instance_id_list is None:
                x_context = self.get_x_context(start, end)
                return [x_image, x_spectrogram, x_context], y_label

            else:
                y_id = self.get_y_id(start, end)
                x_context = self.get_x_context(start, end)
                return [x_image, x_spectrogram, x_context], y_id, y_label

    def on_epoch_end(self):
        if self.partition == "train":
            self.prepare()

    def prepare(self):
        epoch_factor = self.epoch_count % self.epoch_cycle_length
        if epoch_factor == 0:
            self.construct_dataset()
            i = np.random.permutation(self.instance_count)
            self.instance_species_list = self.shuffle(self.instance_species_list, i)
            self.instance_label_list = self.shuffle(self.instance_label_list, i)
            self.instance_context_list = self.shuffle(self.instance_context_list, i)
            self.instance_id_list = self.shuffle(self.instance_id_list, i)
            self.instance_image_name_list = self.shuffle(self.instance_image_name_list, i)
            self.instance_signal_name_list = self.shuffle(self.instance_signal_name_list, i)
            self.instance_signal_bound_list = self.shuffle(self.instance_signal_bound_list, i)
            self.instance_signal_length_list = self.shuffle(self.instance_signal_length_list, i)
            self.stratify()
        self.prepare_epoch(epoch_factor)

    def shuffle(self, instance_list, i):
        if instance_list is not None:
            instance_list = np.take(instance_list, i, axis=0)
        return instance_list

    def stratify(self):
        if self.epoch_cycle_length != 1:
            self.instance_i_list = []
            stratified_k_fold = model_selection.StratifiedKFold(n_splits=self.epoch_cycle_length)
            for (_, kth_instance_i_list) in stratified_k_fold.split(self.instance_species_list, self.instance_label_list):
                self.instance_i_list.append(kth_instance_i_list)
        else:
            self.instance_i_list = [[i for i in range(self.instance_count)]]

    def prepare_epoch(self, epoch_factor):
        self.kth_instance_i_list = self.instance_i_list[epoch_factor]
        self.kth_instance_count = len(self.kth_instance_i_list)
        self.kth_batch_count = int(math.ceil(self.kth_instance_count / self.batch_size))
        self.epoch_count += 1

    # --------------------------------------------------- Getters ---------------------------------------------------------------

    def get_x_image(self, batch_size, start, end):
        x_image = np.zeros((batch_size, *INCEPTION_DIMENSIONS), dtype=np.float32)
        for j, i in enumerate(self.kth_instance_i_list[start:end]):
            image_path = f"{self.image_dataset_path}{self.instance_species_list[i]}/{self.instance_image_name_list[i]}.{IMAGE_FORMAT}"
            if self.partition == "train":
                x_image[j, :, :, :] = self.augment_image(image=np.array(Image.open(image_path)))["image"].astype(np.float32)
            else:
                image = np.array(Image.open(image_path))
                x_image[j, :, :, :] = self.resize_image(image.shape[0], image.shape[1])(image=image)["image"].astype(np.float32)
        return x_image

    def get_x_spectrogram(self, batch_size, start, end):
        x_spectrogram = np.zeros((batch_size, *INCEPTION_DIMENSIONS), dtype=np.float32)
        for j, i in enumerate(self.kth_instance_i_list[start:end]):
            signal_path = f"{self.signal_dataset_path}{self.instance_species_list[i]}/{self.instance_signal_name_list[i]}.{TRAIN_AUDIO_FORMAT}"
            x_spectrogram[j, :, :, :] = self.resize_spectrogram(image=np.stack((self.read_spectrogram(signal_path, self.instance_signal_bound_list[i], self.instance_signal_length_list[i]),) * SPECTROGRAM_DEPTH, axis=-1))["image"]
        return x_spectrogram

    def get_x_context(self, start, end):
        x_context = self.instance_context_list[self.kth_instance_i_list[start:end], :]
        return x_context

    def get_y_id(self, start, end):
        y_id = self.instance_id_list[self.kth_instance_i_list[start:end]]
        return y_id

    def get_y_label(self, start, end):
        y_label = keras.utils.to_categorical(self.instance_label_list[self.kth_instance_i_list[start:end]], num_classes=self.species_count)
        if self.partition == "train":
            y_label = (1 - LABEL_SMOOTHING_E) * y_label + (LABEL_SMOOTHING_E / self.species_count)
        return y_label

    # --------------------------------------------------- Transforms ---------------------------------------------------------------

    def create_image_augment_transform(self):
        transform = albumentations.Compose([
            albumentations.HorizontalFlip(p=HORIZONTAL_FLIP_PROBABILITY),
            albumentations.RandomResizedCrop(height=INCEPTION_DIMENSIONS[0],
                                             width=INCEPTION_DIMENSIONS[1],
                                             scale=(IMAGE_MINIMUM_SCALE, IMAGE_MAXIMUM_SCALE),
                                             ratio=(IMAGE_MINIMUM_RATIO, IMAGE_MAXIMUM_RATIO),
                                             interpolation=cv2.INTER_LINEAR,
                                             always_apply=True)
        ])
        return transform

    def create_image_resize_transform(self, image_height, image_width):
        transform = albumentations.Compose([
            albumentations.CenterCrop(height=int(image_height * IMAGE_CROP_RATIO),
                                      width=int(image_width * IMAGE_CROP_RATIO),
                                      p=1),
            albumentations.Resize(height=INCEPTION_DIMENSIONS[0],
                                  width=INCEPTION_DIMENSIONS[1],
                                  interpolation=cv2.INTER_LINEAR,
                                  p=1)
        ])
        return transform

    def create_spectrogram_resize_transform(self):
        transform = albumentations.Compose([
            albumentations.Resize(height=INCEPTION_DIMENSIONS[0],
                                  width=INCEPTION_DIMENSIONS[1],
                                  interpolation=cv2.INTER_LINEAR,
                                  p=1)
        ])
        return transform

    # --------------------------------------------------- Readers ---------------------------------------------------------------

    def read_spectrogram(self, signal_path, signal_bound, signal_length):

        if signal_length <= AUDIO_LENGTH:
            noise = self.read_noise(signal_length)
            if noise is None:
                signal = self.read_audio(signal_path, 0, signal_length)
            else:
                signal = self.read_audio(signal_path, 0, signal_length) + noise
            spectrogram = self.spectrogram.create(signal)
            spectrogram = np.pad(spectrogram, ((0, 0), (0, SPECTROGRAM_LENGTH - spectrogram.shape[1])), mode="wrap")
            spectrogram = np.roll(spectrogram, -int(signal_bound / SPECTROGRAM_HOP_LENGTH), axis=1)

        else:

            noise = self.read_noise(AUDIO_LENGTH)
            if (signal_length - signal_bound) >= AUDIO_LENGTH:
                if noise is None:
                    signal = self.read_audio(signal_path, signal_bound, AUDIO_LENGTH)
                else:
                    signal = self.read_audio(signal_path, signal_bound, AUDIO_LENGTH) + noise
                spectrogram = self.spectrogram.create(signal)

            else:

                signal_length_list = [signal_length - signal_bound, AUDIO_LENGTH - signal_length + signal_bound]

                if signal_length_list[0] <= MINIMUM_SIGNAL_LENGTH:
                    signal_bound = signal_length - AUDIO_LENGTH
                    if noise is None:
                        signal = self.read_audio(signal_path, signal_bound, AUDIO_LENGTH)
                    else:
                        signal = self.read_audio(signal_path, signal_bound, AUDIO_LENGTH) + noise
                    spectrogram = self.spectrogram.create(signal)

                elif signal_length_list[1] <= MINIMUM_SIGNAL_LENGTH:
                    signal_bound = 0
                    if noise is None:
                        signal = self.read_audio(signal_path, signal_bound, AUDIO_LENGTH)
                    else:
                        signal = self.read_audio(signal_path, signal_bound, AUDIO_LENGTH) + noise
                    spectrogram = self.spectrogram.create(signal)

                else:
                    if noise is None:
                        spectrogram = np.concatenate((self.spectrogram.create(self.read_audio(signal_path, signal_bound, signal_length_list[0])),
                                                      self.spectrogram.create(self.read_audio(signal_path, 0, signal_length_list[1]))),
                                                     axis=1)
                    else:
                        spectrogram = np.concatenate((self.spectrogram.create(self.read_audio(signal_path, signal_bound, signal_length_list[0])
                                                                              + noise[signal_length_list[1]:]),
                                                      self.spectrogram.create(self.read_audio(signal_path, 0, signal_length_list[1])
                                                                              + noise[:signal_length_list[1]])),
                                                     axis=1)

        return spectrogram

    def read_audio(self, audio_path, audio_bound, audio_length):
        audio_file = soundfile.SoundFile(audio_path)
        audio_file.seek(audio_bound)
        audio = audio_file.read(audio_length)
        return audio

    def read_noise(self, signal_length):
        noise = np.zeros((signal_length,), dtype=np.float32)
        if self.partition == "train":
            if self.instance_noise_name_list is not None:
                for i in range(NOISE_ADDITION_COUNT):
                    if random.uniform(0, 1) <= NOISE_ADDITION_PROBABILITY:
                        j = random.randint(0, len(self.instance_noise_name_list) - 1)
                        noise_bound = random.randint(0, self.instance_noise_length_list[j] - signal_length)
                        noise_path = f"{self.noise_dataset_path}{self.instance_noise_name_list[j]}.{TRAIN_AUDIO_FORMAT}"
                        noise += random.uniform(MINIMUM_NOISE_DAMPING_FACTOR, MAXIMUM_NOISE_DAMPING_FACTOR) * self.read_audio(noise_path, noise_bound, signal_length)
        return noise

    # --------------------------------------------------- Constructors ------------------------------------------------------------

    def construct_dataset(self):

        instance_species_list = self.instantiate_observation_list(self.observation_species_list)
        instance_label_list = self.instantiate_observation_list(self.observation_label_list)
        instance_context_list = []
        instance_id_list = self.instantiate_observation_list(self.observation_id_list)
        instance_image_name_list = self.instantiate_observation_list(self.observation_image_name_list)
        instance_signal_name_list = self.instantiate_observation_list(self.observation_signal_name_list)
        instance_signal_bound_list = self.instantiate_observation_list(self.observation_signal_name_list)
        instance_signal_length_list = self.instantiate_observation_list(self.observation_signal_name_list)
        self.observation_signal_count_dictionary = self.instantiate_observation_count_dictionary(self.observation_signal_name_list)
        self.observation_signal_i_dictionary = self.instantiate_observation_i_dictionary(self.observation_signal_name_list)

        self.instance_count = 0

        for i in range(self.observation_count):

            modality_list = self.get_modality_list(i)

            for modality in modality_list:

                image_no = modality[0]
                signal_no = modality[1][0]
                signal_bound = modality[1][1]
                signal_length = 0 if self.observation_signal_length_list is None else self.observation_signal_length_list[i][signal_no]

                image_name = "" if self.observation_image_name_list is None else f"{self.observation_image_name_list[i]}_{image_no}"
                signal_name = "" if self.observation_signal_name_list is None else f"{self.observation_signal_name_list[i]}_{signal_no}"

                instance_species_list = self.add_from_i(instance_species_list, self.observation_species_list, i)
                instance_label_list = self.add_from_i(instance_label_list, self.observation_label_list, i)
                instance_context_list = self.add_from_i(instance_context_list, self.observation_signal_context_list, i)
                instance_id_list = self.add_from_i(instance_id_list, self.observation_id_list, i)
                instance_image_name_list = self.add(instance_image_name_list, image_name)
                instance_signal_name_list = self.add(instance_signal_name_list, signal_name)
                instance_signal_bound_list = self.add(instance_signal_bound_list, signal_bound)
                instance_signal_length_list = self.add(instance_signal_length_list, signal_length)

                self.instance_species_length = self.update_length(self.instance_species_length, len(self.observation_species_list[i]))
                self.instance_image_name_length = self.update_length(self.instance_image_name_length, len(image_name))
                self.instance_signal_name_length = self.update_length(self.instance_signal_name_length, len(signal_name))

                self.instance_count += 1

        self.instance_species_list = self.convert(instance_species_list, itemsize=self.instance_species_length)
        self.instance_label_list = self.convert(instance_label_list, dtype=np.int32)
        self.instance_context_list = self.convert(instance_context_list, dtype=np.float32)
        self.instance_id_list = self.convert(instance_id_list, dtype=np.int32)
        self.instance_image_name_list = self.convert(instance_image_name_list, itemsize=self.instance_image_name_length)
        self.instance_signal_name_list = self.convert(instance_signal_name_list, itemsize=self.instance_signal_name_length)
        self.instance_signal_bound_list = self.convert(instance_signal_bound_list, dtype=np.int32)
        self.instance_signal_length_list = self.convert(instance_signal_length_list, dtype=np.int32)

    def get_modality_list(self, i):

        if self.observation_image_name_list is not None or self.observation_signal_name_list is not None:

            if self.modality == "image":

                image_no_list = self.get_image_no_list(self.observation_image_no_list[i])
                signal_bound_list = [(0, 0)] * len(image_no_list)
                modality_list = list(zip(image_no_list, signal_bound_list))

            elif self.modality == "audio":

                signal_bound_list = self.get_signal_bound_list(self.observation_signal_length_list[i], self.observation_signal_no_list[i])
                image_no_list = [0] * len(signal_bound_list)
                modality_list = list(zip(image_no_list, signal_bound_list))

            elif self.modality == "image-audio":

                if self.partition == "train":

                    image_no_list = self.get_image_no_list(self.observation_image_no_list[i], image_no_count=1)
                    signal_bound_list = self.get_signal_bound_list(self.observation_signal_length_list[i], self.observation_signal_no_list[i], signal_bound_count=1)
                    modality_list = list(zip(image_no_list, signal_bound_list))

                else:

                    image_no_list = self.get_image_no_list(self.observation_image_no_list[i], image_no_count=1)
                    signal_bound_list = self.get_signal_bound_list(self.observation_signal_length_list[i], self.observation_signal_no_list[i], signal_bound_count=1)
                    modality_list = self.combine(image_no_list, signal_bound_list)

        else:

            image_no_list = [0]
            signal_bound_list = [(0, 0)]
            modality_list = list(zip(image_no_list, signal_bound_list))

        return modality_list

    def get_image_no_list(self, image_no_list, image_no_count=None):

        if self.partition == "train" or self.modality == "image-audio" and self.partition == "validation":

            if image_no_count is None:
                image_no_count = 1

            image_no_list = random.sample(image_no_list, min(image_no_count, len(image_no_list)))
            image_no_cycle = itertools.cycle(image_no_list)
            image_no_list = [next(image_no_cycle) for _ in range(image_no_count)]

        return image_no_list

    def get_signal_bound_list(self, signal_length_list, signal_no_list, signal_bound_count=None):

        signal_bound_list = []

        if self.partition == "train" or self.modality == "image-audio" and self.partition == "validation":

            signal_length = signal_length_list[0]
            signal_no = signal_no_list[0]

            if signal_bound_count is None:
                signal_bound_count = max(min(int(signal_length / AUDIO_LENGTH), MAXIMUM_SPECTROGRAM_COUNT), 1)

            one_signal_bound_list = np.linspace(0, signal_length, signal_bound_count, endpoint=False, dtype=np.int32)
            one_signal_bound_list = (one_signal_bound_list + random.randint(0, signal_length)) % signal_length

            signal_bound_list.extend([(signal_no, signal_bound) for signal_bound in one_signal_bound_list])

        else:

            for signal_length, signal_no in zip(signal_length_list, signal_no_list):

                if self.partition == "validation":

                    if self.modality != "image-audio":
                        one_signal_bound_list = np.arange(0, signal_length, MAXIMUM_AUDIO_HOP_LENGTH, dtype=np.int32)

                else:

                    if self.modality != "image-audio":
                        one_signal_bound_list = np.arange(0, signal_length, MINIMUM_AUDIO_HOP_LENGTH, dtype=np.int32)
                    else:
                        one_signal_bound_list = np.arange(0, signal_length, MAXIMUM_AUDIO_HOP_LENGTH, dtype=np.int32)

                one_signal_bound_list = np.minimum(one_signal_bound_list, signal_length - SPECTROGRAM_WINDOW_LENGTH)
                signal_bound_list.extend([(signal_no, signal_bound) for signal_bound in one_signal_bound_list])

        return signal_bound_list

    def combine(self, image_no_list, signal_bound_list):

        image_no_count = len(image_no_list)
        signal_bound_count = len(signal_bound_list)

        if image_no_count > signal_bound_count:
            signal_bound_cycle = itertools.cycle(signal_bound_list)
            modality_list = [(image_no_list[i], next(signal_bound_cycle)) for i in range(image_no_count)]

        elif image_no_count < signal_bound_count:
            image_no_cycle = itertools.cycle(image_no_list)
            modality_list = [(next(image_no_cycle), signal_bound_list[i]) for i in range(signal_bound_count)]

        else:

            modality_list = list(zip(image_no_list, signal_bound_list))

        return modality_list

    # --------------------------------------------------- Trackers ------------------------------------------------------------

    def instantiate_observation_count_dictionary(self, observation_list):
        observation_dictionary = None
        if self.observation_image_name_list is not None and self.observation_signal_name_list is not None:
            observation_dictionary = {}
            for observation in observation_list:
                observation_dictionary = update_frequency_dictionary_(observation, 1, observation_dictionary)
        return observation_dictionary

    def instantiate_observation_i_dictionary(self, observation_list):
        instance_dictionary = None
        if self.observation_image_name_list is not None and self.observation_signal_name_list is not None:
            instance_dictionary = {observation: 0 for observation in observation_list}
        return instance_dictionary

# --------------------------------------------------- Helpers ------------------------------------------------------------

    def instantiate_observation_list(self, observation_list):
        if observation_list is not None:
            instance_list = []
        else:
            instance_list = None
        return instance_list


    def convert(self, instance_list, dtype=None, itemsize=None):
        instance_array = None
        if instance_list is not None:
            if dtype is not None:
                instance_array = np.array(instance_list, dtype=dtype)
            if itemsize is not None:
                instance_array = np.char.array(instance_list, itemsize=itemsize)
        return instance_array


    def add(self, instance_list, instance):
        if instance_list is not None:
            instance_list.append(instance)
        return instance_list


    def add_from_i(self, instance_list, observation_list, i):
        if instance_list is not None:
            instance_list.append(observation_list[i])
        return instance_list


    def update_length(self, maximum_length, length):
        maximum_length = max(maximum_length, length)
        return maximum_length
