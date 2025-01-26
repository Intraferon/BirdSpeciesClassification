from tensorflow import keras
import os
from dataset_generators.constants import *


class ContextANN:

    name = "context_ann"
    start_stage = 0
    end_stage = 0

    def __init__(self, species_count, **kwargs):
        self.species_count = species_count
        self.context_ann_input_dimensions = kwargs["context_ann_input_dimensions"]
        self.base_model_path = f"{RESOURCE_PATH}model_saves/context_ann_{species_count}_{self.context_ann_input_dimensions}.hdf5"
        self.create_base()

    def create_base(self):
        if not os.path.exists(self.base_model_path):
            inputs = keras.Input(shape=self.context_ann_input_dimensions, name=f"{self.name}_input")
            x = keras.layers.Dense(256, activation="relu", name=f"{self.name}_dense_0")(inputs)
            x = keras.layers.Dense(516, activation="relu", name=f"{self.name}_dense_1")(x)
            x = keras.layers.Dense(1024, activation="relu", name=f"{self.name}_dense_2")(x)
            x = keras.layers.Dense(2048, activation="relu", name=f"{self.name}_dense_3")(x)
            x = keras.layers.Dense(self.species_count, name=f"{self.name}_dense_4")(x)
            outputs = keras.layers.Softmax(name=f"{self.name}_softmax")(x)
            model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            model.save(self.base_model_path, include_optimizer=False)

    def create(self):
        model = keras.models.load_model(self.base_model_path)
        return model

    def alter_state(self, stage, model):
        self.alter_base_state(stage, model)

    @classmethod
    def alter_base_state(cls, stage, model):
        if stage == 0:
            for i in range(5): model.get_layer(f"{cls.name}_dense_{i}").trainable = True

        if stage == 1:
            for i in range(4): model.get_layer(f"{cls.name}_dense_{i}").trainable = False

        if stage == 2:
            for i in range(4): model.get_layer(f"{cls.name}_dense_{i}").trainable = True
