"""Networks architectures for feature extraction."""

from __future__ import annotations

from copy import deepcopy

import omegaconf
import torch
from torch import jit, nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class MLP(jit.ScriptModule):
    """MLP network."""

    def __init__(self, input_size, n_layers=3, hidden_sizes=None, activation="relu"):
        """Initialize the network."""
        super().__init__()
        if isinstance(input_size, list | tuple):
            input_size = input_size[0]

        layers = [nn.Linear(input_size, hidden_sizes[0]), ACTIVATIONS[activation]()]
        for i in range(1, n_layers):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(ACTIVATIONS[activation]())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.layers(x)

    def reset(self, reset_ratio=0.5, shrink=0.4, perturb=0.1):
        """Reset the network."""
        if reset_ratio == 1:
            self.full_hard_reset()
        else:
            self.soft_reset(reset_ratio, shrink, perturb)

    def full_hard_reset(self):
        """Full hard reset of the network."""
        for n in range(0, len(self.layers), 2):
            self.layers[n].reset_parameters()

    def soft_reset(self, reset_ratio, shrink, perturb):
        """Soft reset of the network."""
        old_params = deepcopy(list(self.layers.parameters()))
        self.full_hard_reset()
        for old_param, new_param in zip(old_params, self.parameters(), strict=False):
            if torch.rand(1) < reset_ratio:
                new_param.data = deepcopy(
                    shrink * old_param.data + perturb * new_param.data
                )
            else:
                new_param.data = old_param.data

    def __getstate__(self):
        return self.layers

    def __setstate__(self, state):
        self.layers = state[0]


class CNN(jit.ScriptModule):
    """CNN network."""

    def __init__(
        self,
        obs_shape,
        n_convolutions,
        out_channels,
        sizes,
        strides,
        paddings,
        activation="relu",
        flatten=True,
        conv_dim=None,
    ):
        """Initialize the network."""
        super().__init__()
        cnn = []
        if conv_dim is not None:
            if conv_dim == 1:
                conv_layer = nn.Conv1d
            elif conv_dim == 2:  # noqa: PLR2004
                conv_layer = nn.Conv2d
        else:
            conv_layer = nn.Conv1d if len(obs_shape) == 1 else nn.Conv2d
        last_shape = obs_shape[-1]

        for i in range(n_convolutions):
            args = [last_shape, out_channels[i], sizes[i]]
            if strides is not None:
                args.append(strides[i])
            if paddings is not None:
                args.append(paddings[i])
            cnn.append(conv_layer(*args))
            cnn.append(ACTIVATIONS[activation]())
            last_shape = out_channels[i]
        if flatten:
            cnn.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn)

    def forward(self, x, transform: bool = True):
        """Forward pass."""
        if transform:
            x = x.permute(2, 0, 1) if len(x.shape) == 3 else x.permute(0, 3, 1, 2)  # noqa: PLR2004
        return self.cnn(x)

    def __getstate__(self):
        return self.cnn

    def __setstate__(self, state):
        self.cnn = state[0]


class ResNetBlock(jit.ScriptModule):
    """Single ResNet block."""

    def __init__(self, planes, activation="relu", stride=1):
        """Initialize the network."""
        super().__init__()
        conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        activ = ACTIVATIONS[activation]()
        conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.block = nn.Sequential(*[conv1, activ, conv2, activ])

    def forward(self, x):
        """Forward pass."""
        out = self.block(x)
        return out + x

    def __getstate__(self):
        return self.block

    def __setstate__(self, state):
        self.block = state[0]


class ResNetLayer(jit.ScriptModule):
    """Single ResNet layer."""

    def __init__(
        self,
        in_channels,
        planes,
        stride=1,
        activation="relu",
    ):
        """Initialize the network."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, planes, kernel_size=3, stride=stride)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = ResNetBlock(planes, activation, stride)
        self.block2 = ResNetBlock(planes, activation, stride)

    def forward(self, x):
        """Forward pass."""
        x = self.conv(x)
        x = self.pool(x)
        x = self.block1(x)
        return self.block2(x)

    def __getstate__(self):
        return (self.conv, self.pool, self.block1, self.block2)

    def __setstate__(self, state):
        self.conv, self.pool, self.block1, self.block2 = state


class ResNet(jit.ScriptModule):
    """ResNet with 3 layers network."""

    def __init__(
        self,
        obs_shape,
        planes,
        stride=1,
        activation="relu",
    ):
        """Initialize the network."""
        super().__init__()
        in_channels = obs_shape[-1]
        self.layer1 = ResNetLayer(in_channels, planes[0], stride, activation)
        self.layer2 = ResNetLayer(planes[0], planes[1], stride, activation)
        self.layer3 = ResNetLayer(planes[1], planes[2], stride, activation)
        self.flatten = nn.Flatten()
        self.activ = ACTIVATIONS[activation]()

    def forward(self, x, transform: bool = True):
        """Forward pass."""
        if transform:
            x = x.permute(2, 0, 1) if len(x.shape) == 3 else x.permute(0, 3, 1, 2)  # noqa: PLR2004
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        return self.activ(x)

    def __getstate__(self):
        return (self.layer1, self.layer2, self.layer3, self.flatten, self.activ)

    def __setstate__(self, state):
        self.layer1, self.layer2, self.layer3, self.flatten, self.activ = state


class LSTM(jit.ScriptModule):
    """LSTM network."""


class ComboNet(jit.ScriptModule):
    """Combination of several network architectures network."""

    def __init__(self, module1, module2):
        """Initialize the network."""
        super().__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, x):
        """Forward pass."""
        x = x.permute(2, 0, 1) if len(x.shape) == 3 else x.permute(0, 3, 1, 2)  # noqa: PLR2004
        x = self.module1(x, False)
        return self.module2(x)


class TorchHubModel(jit.ScriptModule):
    def __init__(
        self,
        obs_shape,
        model_name: str,
        repo: str = "pytorch/vision:v0.9.0",
        pretrained: bool = False,
    ):
        super().__init__()
        # Project obs to model input dims
        # TODO: make dims fit model
        self.projection_layer = nn.Conv2d(obs_shape[0], 3, kernel_size=1)
        self.projection_layer.to(device)
        self.projection_layer = torch.jit.script(self.projection_layer)

        # Load model from torch hub
        self.model = torch.hub.load(repo, model_name, pretrained=True)
        self.model.to(device)
        self.model = torch.jit.script(self.model)

    def forward(self, x):
        x = self.projection_layer(x)
        return self.model(x)


def make_feature_extractor(
    architecture="mlp",
    n_layers=3,
    hidden_sizes=None,
    activation="relu",
    obs_shape=None,
    n_convolutions=3,
    out_channels=None,
    sizes=None,
    strides=None,
    paddings=None,
    flatten_cnn=True,
    conv_dim=None,
    planes=None,
    model_name: str = "resnet18",
    repo: str = "pytorch/vision:v0.9.0",
    pretrained: bool = False,
):
    """Make a feature extractor network."""
    if planes is None:
        planes = [16, 32, 32]
    if sizes is None:
        sizes = [8, 4, 3]
    if out_channels is None:
        out_channels = [32, 64, 64]
    if hidden_sizes is None:
        hidden_sizes = [32, 32, 32]
    if architecture == "mlp":
        fe = MLP(obs_shape, n_layers, hidden_sizes, activation)
        output_size = hidden_sizes[-1]
    elif architecture == "cnn":
        fe = CNN(
            obs_shape,
            n_convolutions,
            out_channels,
            sizes,
            strides,
            paddings,
            activation,
            flatten_cnn,
            conv_dim,
        )
        output_size = list(fe(torch.rand((1, *obs_shape))).shape[1:])
    elif architecture == "resnet":
        fe = ResNet(obs_shape, planes, activation=activation)
        output_size = list(fe(torch.rand((1, *obs_shape))).shape[1:])
    elif architecture == "torchhub":
        fe = TorchHubModel(
            obs_shape, model_name=model_name, repo=repo, pretrained=pretrained
        )
        output_size = list(fe(torch.rand((1, *obs_shape))).shape[1:])
    elif isinstance(architecture, list | omegaconf.listconfig.ListConfig):
        modules = []
        original_obs_shape = obs_shape
        for arch in architecture:
            m, obs_shape = make_feature_extractor(
                arch,
                n_layers,
                hidden_sizes,
                activation,
                obs_shape,
                n_convolutions,
                out_channels,
                sizes,
                strides,
                paddings,
                flatten_cnn,
                conv_dim,
            )
            modules.append(m)

        fe = ComboNet(*modules)
        output_size = list(fe(torch.rand((1, *original_obs_shape))).shape[1:])
    else:
        raise ValueError(f"Unknown architecture {architecture}")

    return fe, output_size
