from torch import nn
from torch.nn import functional as F

from zenithml.torch import BaseTorchModel, concat_layers


class LinearClassifier(BaseTorchModel):
    def __init__(self, feature_group_key="features", *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_dims = self.input_dimensions[feature_group_key]
        self.input_layer = nn.ModuleDict(self.preprocess_layers[feature_group_key])
        self.classifier = nn.Linear(in_features=input_dims, out_features=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, inputs):
        feature_layer = concat_layers(inputs, self.input_layer)
        return self.sigmod(self.classifier(feature_layer))

    def loss_fn(self, logits, truth):
        # TODO: user must be able to change the loss
        return F.binary_cross_entropy(logits, truth.unsqueeze(1))
