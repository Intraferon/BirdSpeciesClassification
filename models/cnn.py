from tensorflow import keras
from dataset_generators.constants import *
import os


class CNN:

    start_stage = 1
    end_stage = 2

    def __init__(self, name, species_count):
        self.name = name
        self.species_count = species_count
        self.base_model_path = f"{RESOURCE_PATH}model_saves/{self.name}_{species_count}.hdf5"
        self.create_base()

    def create_base(self):
        if not os.path.exists(self.base_model_path):
            base_model = keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=INCEPTION_DIMENSIONS, name=f"{self.name}_inception_v3")
            inputs = keras.Input(shape=INCEPTION_DIMENSIONS, name=f"{self.name}_input")
            x = base_model(inputs)
            x = keras.layers.GlobalAveragePooling2D(name=f"{self.name}_global_average_pooling")(x)
            x = keras.layers.Dense(self.species_count, name=f"{self.name}_dense")(x)
            outputs = keras.layers.Softmax(name=f"{self.name}_softmax")(x)
            model = keras.Model(inputs, outputs, name=self.name)
            model.save(self.base_model_path, include_optimizer=False)

    def create(self):
        model = keras.models.load_model(self.base_model_path)
        return model

    @classmethod
    def alter_base_state(cls, stage, model, name):

        if stage == 1:

            model.get_layer(f"{name}_inception_v3").trainable = False

        if stage == 2:

            model.get_layer(f"{name}_inception_v3").trainable = True

