from tensorflow import keras


class ContextCNN:
    name = "context_cnn"
    start_stage = 1
    end_stage = 2

    def __init__(self, species_count, **kwargs):
        self.species_count = species_count
        self.cnn_path = kwargs["cnn_path"]
        self.context_ann_path = kwargs["context_ann_path"]
        self.base_cnn_path = kwargs["base_cnn_path"]
        self.base_context_ann_path = kwargs["base_context_ann_path"]

    def create(self):

        cnn = keras.models.load_model(f"{self.cnn_path}.hdf5")
        context_ann = keras.models.load_model(f"{self.context_ann_path}.hdf5")

        # Add (keep dense weights)
        context_cnn = keras.layers.Add(name=f"{self.name}_add")([cnn.layers[-2].output, context_ann.layers[-2].output])
        outputs = keras.layers.Softmax(name=f"{self.name}_softmax")(context_cnn)

        model = keras.Model(inputs=[cnn.input, context_ann.input], outputs=[outputs], name=self.name)

        return model
