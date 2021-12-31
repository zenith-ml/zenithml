from torch import nn

from condorml.torch.layers import NumericalLayer


class CosineSimilarityLayer(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.numeric_layer1 = NumericalLayer(dimension=dimension)
        self.numeric_layer2 = NumericalLayer(dimension=dimension)
        self.cosine_layer = nn.CosineSimilarity()

        self.trainable = False

    def forward(self, inputs):
        x, y = inputs
        return self.cosine_layer(self.numeric_layer1(x), self.numeric_layer2(y)).view(-1, 1)

    def output_dim(self):
        return 1
