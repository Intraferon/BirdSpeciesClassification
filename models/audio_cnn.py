from models.cnn import CNN

class AudioCNN(CNN):

    name_ = "audio_cnn"

    def __init__(self, species_count):
        super(AudioCNN, self).__init__(self.name_, species_count)

    def alter_state(self, stage, model):
        self.alter_base_state(stage, model)

    @classmethod
    def alter_base_state(cls, stage, model):
        CNN.alter_base_state(stage, model, cls.name_)

