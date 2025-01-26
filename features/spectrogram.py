import librosa
import numpy as np
from scipy import ndimage
import math
from dataset_generators.constants import *
import random


class Spectrogram:

    # ************************************************ CONSTRUCTOR *****************************************************

    def __init__(self):
        pass

    # ************************************************ MAJOR METHODS ***************************************************

    def is_valid(self, audio):
        valid = True
        spectrogram = self.create_from_audio(audio)
        if spectrogram is not None:
            spectrogram = 10.0 * np.log10(spectrogram + SPECTROGRAM_E)
            min_db = spectrogram.min()
            max_db = spectrogram.max()
            if np.isclose(min_db, max_db):
                valid = False
        else:
            valid = False
        return valid

    def create(self, audio):
        spectrogram = self.create_from_audio(audio)
        spectrogram = 10.0 * np.log10(spectrogram + SPECTROGRAM_E)
        min_db = spectrogram.min()
        max_db = spectrogram.max()
        if not np.isclose(min_db, max_db):
            spectrogram = (spectrogram - min_db) / (max_db - min_db)
        else:
            spectrogram = spectrogram * 0
        spectrogram_image = (spectrogram * 255).astype(np.uint8)
        spectrogram_image = np.flip(spectrogram_image, axis=0)
        spectrogram_image = 255 - spectrogram_image
        return spectrogram_image

    def create_from_audio(self, audio):
        if audio.shape[0] >= SPECTROGRAM_WINDOW_LENGTH:
            spectrogram = np.abs(
                librosa.stft(
                    y=audio,
                    n_fft=SPECTROGRAM_WINDOW_LENGTH,
                    win_length=SPECTROGRAM_WINDOW_LENGTH,
                    hop_length=SPECTROGRAM_HOP_LENGTH,
                    window=SPECTROGRAM_WINDOW_FUNCTION,
                    center=False,
                    pad_mode="constant"
                )
            ) ** 2
            if SPECTROGRAM_TRIM_UPPER:
                spectrogram = spectrogram[:SPECTROGRAM_MAXIMUM_FREQUENCY_BAND, :]
            if SPECTROGRAM_TRIM_LOWER:
                spectrogram = spectrogram[SPECTROGRAM_MINIMUM_FREQUENCY_BAND:, :]
        else:
            spectrogram = None
        return spectrogram

    def extract_indicator(self, audio, extract_noise=False):
        spectrogram = self.create_from_audio(audio)
        spectrogram_image = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        row_median = np.median(spectrogram_image, axis=1)[np.newaxis].transpose()
        column_median = np.median(spectrogram_image, axis=0)
        signal_indicator, median_threshold = self.median_clip(spectrogram_image, row_median, column_median, median_threshold=SIGNAL_MEDIAN_THRESHOLD)
        noise_indicator = None
        if extract_noise and signal_indicator is not None:
            median_threshold = round(median_threshold - NOISE_MEDIAN_THRESHOLD_DIFFERENCE, 1)
            if median_threshold > (MINIMUM_NOISE_MEDIAN_THRESHOLD - SIGNAL_MEDIAN_THRESHOLD_STEP):
                noise_indicator, _ = self.median_clip(spectrogram_image, row_median, column_median, median_threshold=median_threshold, extract_noise=True)
        return signal_indicator, noise_indicator

    def segment_audio(self, audio, spectrogram_indicator, extract_noise=False):
        audio_indicator = np.zeros(audio.shape[0], dtype=bool)
        for i in range(spectrogram_indicator.shape[0]):
            if spectrogram_indicator[i] == 1:
                start = i * SPECTROGRAM_HOP_LENGTH
                end = min(start + SPECTROGRAM_WINDOW_LENGTH, audio.shape[0])
                audio_indicator[start:end] = True
        segmented_audio = audio[audio_indicator]
        if extract_noise:
            if segmented_audio.shape[0] > MAXIMUM_NOISE_LENGTH:
                start = int((segmented_audio.shape[0] - MAXIMUM_NOISE_LENGTH) / 2)
                end = start + MAXIMUM_NOISE_LENGTH
                segmented_audio = segmented_audio[start:end]
        return segmented_audio

    def median_clip(self, spectrogram_image, row_median, column_median, median_threshold, extract_noise=False):
        spectrogram_indicator = None
        spectrogram_length = spectrogram_image.shape[1]
        if spectrogram_length > MINIMUM_SPECTROGRAM_LENGTH:
            spectrogram_indicator = self.median_clip_once(spectrogram_image, row_median, column_median, median_threshold, extract_noise=extract_noise)
            spectrogram_length = np.count_nonzero(spectrogram_indicator == 1)
            if not extract_noise:
                while (spectrogram_length < MINIMUM_SPECTROGRAM_LENGTH) and (median_threshold > MINIMUM_SIGNAL_MEDIAN_THRESHOLD):
                    median_threshold = round(median_threshold - SIGNAL_MEDIAN_THRESHOLD_STEP, 1)
                    spectrogram_indicator = self.median_clip_once(spectrogram_image, row_median, column_median, median_threshold, extract_noise=extract_noise)
                    spectrogram_length = np.count_nonzero(spectrogram_indicator == 1)
        if spectrogram_length <= MINIMUM_SPECTROGRAM_LENGTH:
            if extract_noise:
                spectrogram_indicator = None
            else:
                spectrogram_indicator = np.ones(spectrogram_image.shape[1]).astype(np.int32)
        return spectrogram_indicator, median_threshold

    @staticmethod
    def median_clip_once(spectrogram_image, row_median, column_median, median_threshold, extract_noise):
        row_median = row_median * median_threshold
        column_median = column_median * median_threshold
        spectrogram_of_concern = np.where((spectrogram_image > column_median) & (spectrogram_image > row_median), 1, 0)
        spectrogram_of_concern = ndimage.binary_erosion(spectrogram_of_concern, structure=np.ones((4, 4)))
        spectrogram_of_concern = ndimage.binary_dilation(spectrogram_of_concern, structure=np.ones((4, 4)))
        spectrogram_indicator = np.any(spectrogram_of_concern == 1, axis=0)
        spectrogram_indicator = ndimage.binary_dilation(spectrogram_indicator, structure=np.ones(4), iterations=2)
        if extract_noise: spectrogram_indicator = 1 - spectrogram_indicator
        return spectrogram_indicator.astype(np.int32)

# ******************************************************************************************************************
