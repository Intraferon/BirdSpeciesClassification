from models.image_cnn import ImageCNN
from models.context_cnn import ContextCNN
from models.context_ann import ContextANN


class ImageContextCNN(ContextCNN):

    def __init__(self, species_count, **kwargs):
        super(ImageContextCNN, self).__init__(species_count, **kwargs)

    def alter_state(self, stage, model):
        ImageCNN.alter_base_state(stage, model)
        ContextANN.alter_base_state(stage, model, species_count=self.species_count)
