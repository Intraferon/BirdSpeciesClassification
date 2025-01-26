from tensorflow import keras
from models.audio_cnn import AudioCNN
from models.context_cnn import ContextCNN
from models.context_ann import ContextANN


class AudioContextCNN(ContextCNN):

    def __init__(self, species_count, **kwargs):
        super(AudioContextCNN, self).__init__(species_count, **kwargs)

    def alter_state(self, stage, model):
        AudioCNN.alter_base_state(stage, model)
        ContextANN.alter_base_state(stage, model, species_count=self.species_count)
