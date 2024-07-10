from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from mighty.mighty_models.networks import (
    ACTIVATIONS,
    CNN,
    MLP,
    ComboNet,
    ResNet,
    make_feature_extractor,
)

test_mlps = [
    (3, 2, [5, 5], "relu"),
    (3, 1, [2], "tanh"),
    (3, 5, [5, 5, 7, 8, 9], "sigmoid"),
]


def test_make_mlp():
    mlp, outsize = make_feature_extractor(
        "mlp", n_layers=3, hidden_sizes=[5, 5, 5], activation="relu", obs_shape=(3)
    )
    assert isinstance(mlp, MLP), "Method did not return MLP."
    assert isinstance(mlp, torch.jit.ScriptModule), "MLP is not a ScriptModule."
    assert outsize == 5, "Output size is not correct."
    assert mlp.layers[0].in_features == 3, "Wrong obs shape."
    assert mlp.layers[0].out_features == 5, "Wrong hidden size in layer 0."
    assert mlp.layers[2].out_features == 5, "Wrong hidden size in layer 1."
    assert mlp.layers[4].out_features == 5, "Wrong hidden size in layer 2."
    assert type(mlp.layers[1]) == ACTIVATIONS["relu"], "Activation 0 is not correct."
    assert type(mlp.layers[3]) == ACTIVATIONS["relu"], "Activation 1 is not correct."
    assert type(mlp.layers[5]) == ACTIVATIONS["relu"], "Activation 2 is not correct."
    with pytest.raises(IndexError):
        mlp.layers[6]
    dummy_input = torch.rand(3)
    mlp(dummy_input)


def test_make_cnn():
    cnn, outsize = make_feature_extractor(
        "cnn",
        out_channels=[32, 64, 64],
        sizes=[8, 4, 3],
        strides=[4, 2, 1],
        paddings=[0, 0, 0],
        activation="relu",
        obs_shape=(64, 64, 3),
    )
    assert isinstance(cnn, CNN), "Method did not return CNN."
    assert isinstance(cnn, torch.jit.ScriptModule), "CNN is not a ScriptModule."
    assert outsize == [1024], "Output size is not correct."
    assert cnn.cnn[0].in_channels == 3, "Wrong obs shape."
    assert cnn.cnn[0].out_channels == 32, "Wrong out_channels in layer 0."
    assert cnn.cnn[2].out_channels == 64, "Wrong out_channels in layer 1."
    assert cnn.cnn[4].out_channels == 64, "Wrong out_channels in layer 2."
    assert cnn.cnn[0].kernel_size == (8, 8), "Wrong kernel size in layer 0."
    assert cnn.cnn[2].kernel_size == (4, 4), "Wrong kernel size in layer 1."
    assert cnn.cnn[4].kernel_size == (3, 3), "Wrong kernel size in layer 2."
    assert cnn.cnn[0].stride == (4, 4), "Wrong stride in layer 0."
    assert cnn.cnn[2].stride == (2, 2), "Wrong stride in layer 1."
    assert cnn.cnn[4].stride == (1, 1), "Wrong stride in layer 2."
    assert cnn.cnn[0].padding == (0, 0), "Wrong padding in layer 0."
    assert cnn.cnn[2].padding == (0, 0), "Wrong padding in layer 1."
    assert cnn.cnn[4].padding == (0, 0), "Wrong padding in layer 2."
    assert type(cnn.cnn[1]) == ACTIVATIONS["relu"], "Activation 0 is not correct."
    assert type(cnn.cnn[3]) == ACTIVATIONS["relu"], "Activation 1 is not correct."
    assert type(cnn.cnn[5]) == ACTIVATIONS["relu"], "Activation 2 is not correct."
    assert isinstance(cnn.cnn[6], torch.nn.Flatten), "Flatten layer is not present."
    with pytest.raises(IndexError):
        cnn.cnn[7]
    dummy_input = torch.rand(1, 64, 64, 3)
    cnn(dummy_input)


def test_make_resnet():
    resnet, outsize = make_feature_extractor(
        "resnet", planes=[16, 32, 32], activation="relu", obs_shape=(64, 64, 3)
    )
    assert isinstance(resnet, ResNet), "Method did not return ResNet."
    assert isinstance(resnet, torch.jit.ScriptModule), "ResNet is not a ScriptModule."


def test_make_combo():
    combo, outsize = make_feature_extractor(
        ["resnet", "mlp"],
        planes=[16, 32, 32],
        activation="relu",
        obs_shape=(64, 64, 3),
    )
    assert isinstance(combo, ComboNet), "Method did not return ComboNet."
    assert isinstance(combo, torch.jit.ScriptModule), "ComboNet is not a ScriptModule."


def test_make_torchhub():
    hub, outsize = make_feature_extractor(
        "torchhub", model_name="resnet18", obs_shape=(64, 64, 3)
    )
    assert isinstance(hub, torch.jit.ScriptModule), "TorchHub is not a ScriptModule."
    test_input = torch.rand(1, 64, 64, 3)
    output = hub(test_input)
    assert output.shape == (1, outsize[0]), "Output shape is not correct."


class TestMLP:
    @pytest.mark.parametrize(
        ("input_size", "n_layers", "hidden_sizes", "activation"), test_mlps
    )
    def test_init(self, input_size, n_layers, hidden_sizes, activation):
        mlp = MLP(input_size, n_layers, hidden_sizes, activation)

        assert isinstance(mlp, torch.jit.ScriptModule), "MLP is not a ScriptModule."
        for n in range(n_layers):
            assert (
                type(mlp.layers[2 * n]) == torch.nn.Linear
            ), f"Layer {n} is not a Linear."
            assert (
                type(mlp.layers[2 * n + 1]) == ACTIVATIONS[activation]
            ), f"Activation {n} is not correct."
            assert (
                mlp.layers[2 * n].out_features == hidden_sizes[n]
            ), f"Wrong in_features in layer {n}."

        with pytest.raises(IndexError):
            mlp.layers[2 * n + 2]

    def test_soft_reset(self):
        dummy_input = torch.rand(1, 3)
        mlp = MLP(3, 2, [5, 5], "relu")
        original_pred = mlp(dummy_input)
        prev_model_weights = [
            deepcopy(mlp.layers[0].weight),
            deepcopy(mlp.layers[2].weight),
        ]
        mlp.soft_reset(0, 0.5, 0.5)
        reset_pred = mlp(dummy_input)
        assert torch.allclose(
            original_pred, reset_pred
        ), "Model prediction has changed."
        assert torch.allclose(
            mlp.layers[0].weight, prev_model_weights[0]
        ), "Weights have reset in layer 0 even though probability was 0."
        assert torch.allclose(
            mlp.layers[2].weight, prev_model_weights[1]
        ), "Weights have reset in layer 1 even though probability was 0."

        mlp.soft_reset(1, 0.5, 0.5)
        reset_pred = mlp(dummy_input)
        assert ~torch.allclose(
            original_pred, reset_pred
        ), "Model prediction has not changed."
        assert not any(
            torch.isclose(mlp.layers[0].weight, prev_model_weights[0])
            .flatten()
            .tolist()
        ), "Not all weights have been reset in layer 0 even though probability was 1."
        assert not any(
            torch.isclose(mlp.layers[2].weight, prev_model_weights[1])
            .flatten()
            .tolist()
        ), "Not all weights have been reset in layer 1 even though probability was 1."

        mlp.layers[0].weight = deepcopy(prev_model_weights[0])
        mlp.layers[2].weight = deepcopy(prev_model_weights[1])
        prev_params = deepcopy(list(mlp.parameters()))
        original_pred = mlp(dummy_input)
        mlp.soft_reset(1, 1, 0)
        reset_pred = mlp(dummy_input)
        assert torch.allclose(
            original_pred, reset_pred
        ), "Model prediction has changed though perturb was 0."
        for new_param, old_param in zip(mlp.parameters(), prev_params, strict=False):
            assert torch.allclose(
                new_param, old_param
            ), "Weights have been reset even though perturb was 0."

        mlp.soft_reset(1, 0.5, 0.0)
        reset_pred = mlp(dummy_input)
        assert (
            original_pred * 0.5 - reset_pred
        ).sum() < 0.1, "Model prediction didn't shrink with parameter value."
        for new_param, old_param in zip(mlp.parameters(), prev_params, strict=False):
            assert torch.allclose(
                new_param, old_param * 0.5
            ), "Weights have not been shrunk."

        mlp.layers[0].weight = deepcopy(prev_model_weights[0])
        mlp.layers[2].weight = deepcopy(prev_model_weights[1])
        prev_params = deepcopy(list(mlp.parameters()))
        original_pred = mlp(dummy_input)
        mlp.soft_reset(1, 1, 1)
        reset_pred = mlp(dummy_input)
        assert ~torch.allclose(
            original_pred, reset_pred
        ), "Model prediction has not changed."
        for new_param, old_param in zip(mlp.parameters(), prev_params, strict=False):
            assert ~torch.allclose(
                new_param, old_param
            ), "Weights have not been perturbed."

        mlp = MLP(3, 5, [100, 100, 100, 100, 100], "relu")
        n_reset = 0
        total_params = 0
        old_params = deepcopy(list(mlp.parameters()))
        for _ in range(20):
            mlp.soft_reset(0.5, 0, 1)
            for reset_param, old_param in zip(
                mlp.parameters(), old_params, strict=False
            ):
                total_params += 1
                if not torch.allclose(reset_param, old_param):
                    n_reset += 1
                    reset_param.data = old_param.data

        assert n_reset / total_params >= 0.25, "Weights reset too rarely."
        assert n_reset / total_params <= 0.75, "Weights reset too often."

    def test_hard_reset(self):
        dummy_input = torch.rand(1, 3)
        mlp = MLP(3, 2, [5, 5], "relu")
        original_pred = mlp(dummy_input)
        prev_model_weights = [
            deepcopy(mlp.layers[0].weight),
            deepcopy(mlp.layers[2].weight),
        ]
        mlp.full_hard_reset()
        reset_pred = mlp(dummy_input)
        assert ~torch.allclose(
            original_pred, reset_pred
        ), "Model prediction has not changed."
        assert ~torch.allclose(
            mlp.layers[0].weight, prev_model_weights[0]
        ), "Weights have not been reset in layer 0."
        assert ~torch.allclose(
            mlp.layers[2].weight, prev_model_weights[1]
        ), "Weights have not been reset in layer 1."

    def test_reset(self):
        dummy_input = torch.rand(1, 3)
        mlp = MLP(3, 2, [5, 5], "relu")
        original_pred = mlp(dummy_input)
        prev_model_weights = [
            deepcopy(mlp.layers[0].weight),
            deepcopy(mlp.layers[2].weight),
        ]
        mlp.reset(1)
        reset_pred = mlp(dummy_input)
        assert ~torch.allclose(
            original_pred, reset_pred
        ), "Model prediction has not changed."
        assert ~torch.allclose(
            mlp.layers[0].weight, prev_model_weights[0]
        ), "Weights have not been reset in layer 0."
        assert ~torch.allclose(
            mlp.layers[2].weight, prev_model_weights[1]
        ), "Weights have not been reset in layer 1."

        mlp = MLP(3, 5, [100, 100, 100, 100, 100], "relu")
        original_pred = mlp(dummy_input)
        prev_model_weights = [
            deepcopy(mlp.layers[0].weight),
            deepcopy(mlp.layers[2].weight),
        ]
        n_reset = 0
        total_params = 0
        old_params = deepcopy(list(mlp.parameters()))
        for _ in range(20):
            mlp.reset(0.5, 0, 1)
            reset_pred = mlp(dummy_input)
            for reset_param, old_param in zip(
                mlp.parameters(), old_params, strict=False
            ):
                total_params += 1
                if not torch.allclose(reset_param, old_param):
                    n_reset += 1
                    reset_param.data = old_param.data

        assert ~torch.allclose(
            original_pred, reset_pred
        ), "Model prediction has not changed."
        assert n_reset / total_params >= 0.25, "Weights reset too rarely in soft reset."
        assert n_reset / total_params <= 0.75, "Weights reset too often in soft reset."

    def test_forward(self):
        mlp = MLP(3, 2, [5, 5], "relu")
        dummy_input = torch.rand(3, 3)
        output = mlp(dummy_input)
        assert output.shape == (3, 5), "Output shape is not correct."
        assert output.dtype == torch.float32, "Output dtype is not correct."
        assert torch.allclose(
            output, mlp.layers(dummy_input)
        ), "Forward is not correct."


class TestCNN:
    def test_init(self):
        # TODO: test 1 and 2d convolutions
        pass

    def test_forward(self):
        # TODO: test transform true and false
        pass


class TestComboNet:
    def test_init(self):
        mlp = MLP(
            n_layers=2, hidden_sizes=[5, 5], activation="relu", input_size=(1024,)
        )
        cnn = CNN((64, 64, 3), 3, [32, 64, 64], [8, 4, 3], [4, 2, 1], [0, 0, 0], "relu")
        combo = ComboNet(cnn, mlp)
        assert isinstance(
            combo, torch.jit.ScriptModule
        ), "ComboNet is not a ScriptModule."
        assert combo.module1 == cnn, "CNN is not the first module."
        assert combo.module2 == mlp, "MLP is not the second module."

    def test_forward(self):
        mlp = MLP(
            n_layers=2, hidden_sizes=[5, 5], activation="relu", input_size=(1024,)
        )
        cnn = CNN((64, 64, 3), 3, [32, 64, 64], [8, 4, 3], [4, 2, 1], [0, 0, 0], "relu")
        combo = ComboNet(cnn, mlp)
        dummy_input = torch.rand(1, 64, 64, 3)

        output = combo(dummy_input)
        assert output.shape == (1, 5), "Output shape is not correct."
        assert output.dtype == torch.float32, "Output dtype is not correct."
        cnn_forward = cnn(dummy_input)
        mlp_forward = mlp(cnn_forward)
        assert torch.allclose(output, mlp_forward), "Forward is not correct."


class TestResNet:
    pass
