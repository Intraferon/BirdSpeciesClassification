from tensorflow import keras
from models.cnn import CNN
from models.image_cnn import ImageCNN
from models.audio_cnn import AudioCNN


class JointCNN:

    name = "joint_cnn"
    start_stage = 1
    end_stage = 1

    def __init__(self, species_count, **kwargs):
        self.species_count = species_count
        self.image_cnn_path = kwargs["image_cnn_path"]
        self.audio_cnn_path = kwargs["audio_cnn_path"]

    def create(self):

        image_cnn = keras.models.load_model(f"{self.image_cnn_path}.hdf5")
        audio_cnn = keras.models.load_model(f"{self.audio_cnn_path}.hdf5")

        # Add (keep dense weights)
        joint_cnn = keras.layers.Add(name=f"{self.name}_add")([image_cnn.layers[-2].output, audio_cnn.layers[-2].output])
        outputs = keras.layers.Softmax(name=f"{self.name}_softmax")(joint_cnn)

        model = keras.Model(inputs=[image_cnn.input, audio_cnn.input], outputs=[outputs], name=self.name)

        return model

    @staticmethod
    def alter_state(stage, model):
        ImageCNN.alter_base_state(stage, model)
        AudioCNN.alter_base_state(stage, model)



