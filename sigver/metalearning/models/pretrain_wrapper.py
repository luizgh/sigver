from torch import nn


class PretrainWrapper(nn.Module):
    def __init__(self, model, weights):
        super(PretrainWrapper, self).__init__()

        self.model = model
        self.w_parameters = nn.ParameterDict({k: nn.Parameter(w) for k, w in weights.items()})
        self.feature_space_size = model.feature_space_size

    def forward(self, x):
        return self.model.get_features(x, self.w_parameters, self.training)
