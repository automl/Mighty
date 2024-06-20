from __future__ import annotations

from copy import deepcopy

import torch
from mighty.mighty_models.dqn import DQN, IQN
from mighty.mighty_models.networks import CNN, MLP


class TestDQN:
    def test_init(self):
        dqn = DQN(num_actions=4, obs_size=3)
        assert dqn.num_actions == 4, "Num_actions should be 4"
        assert dqn.dueling is False, "Dueling should be False"
        assert isinstance(
            dqn.feature_extractor, MLP
        ), "Default feature extractor should be an instance of MLP"
        assert isinstance(
            dqn.head, torch.nn.Sequential
        ), "Head should be a nn.Sequential"
        assert isinstance(
            dqn.value, torch.nn.Linear
        ), "Value layer should be a nn.Linear"
        assert isinstance(
            dqn.advantage, torch.nn.Linear
        ), "Advantage layer should be a nn.Linear"
        dummy_input = torch.rand((1, 3))
        assert dqn(dummy_input).shape == (1, 4), "Output should have shape (1, 4)"

        head_kwargs = {"hidden_sizes": [32, 32]}
        feature_extractor_kwargs = {
            "architecture": "cnn",
            "n_convolutions": 2,
            "out_channels": [32, 64],
            "sizes": [8, 4],
            "activation": "relu",
        }
        dqn = DQN(
            num_actions=2,
            obs_size=(64, 64, 3),
            dueling=True,
            head_kwargs=head_kwargs,
            feature_extractor_kwargs=feature_extractor_kwargs,
        )
        assert dqn.num_actions == 2, "Num_actions should be 4"
        assert dqn.dueling is True, "Dueling should be True"
        assert isinstance(
            dqn.feature_extractor, CNN
        ), "Feature extractor should be a CNN"
        assert (
            len(dqn.feature_extractor.cnn) == 5
        ), "Feature extractor should have 2 convolutions"
        assert (
            dqn.feature_extractor.cnn[0].out_channels == 32
        ), "First convolution should have 32 output channels"
        assert dqn.head[0].out_features == 32, "Head layer 1 should have hidden size 32"
        assert dqn.head[1].out_features == 32, "Head layer 2 should have hidden size 32"
        assert dqn.value.out_features == 1, "Value layer should have 1 output feature"
        assert (
            dqn.advantage.out_features == 2
        ), "Advantage layer should have 2 output features"
        dummy_input = torch.rand((5, 64, 64, 3))
        assert dqn(dummy_input).shape == (5, 2), "Output should have shape (1, 2)"

    def test_forward(self):
        dqn = DQN(num_actions=4, obs_size=3)
        dummy_input = torch.rand((50, 3))
        dqn_pred = dqn(dummy_input)
        assert dqn_pred.shape == (50, 4), "Output should have shape (1, 4)"
        calculated_pred = dqn.advantage(dqn.head(dqn.feature_extractor(dummy_input)))
        assert torch.allclose(
            dqn_pred, calculated_pred
        ), """Prediction should be equal to
            advantage(head(feature_extractor(dummy_input)))"""

        dqn = DQN(num_actions=4, obs_size=3, dueling=True)
        dummy_input = torch.rand((50, 3))
        dqn_pred = dqn(dummy_input)
        assert dqn_pred.shape == (50, 4), "Output should have shape (1, 4)"
        calculated_advantage = dqn.advantage(
            dqn.head(dqn.feature_extractor(dummy_input))
        )
        calculated_value = dqn.value(dqn.head(dqn.feature_extractor(dummy_input)))
        calculated_pred = (
            calculated_value
            + calculated_advantage
            - calculated_advantage.mean(dim=1, keepdim=True)
        )
        assert torch.allclose(
            dqn_pred, calculated_pred
        ), "Prediction should be equal to value + advantage - mean_advantage"

    def test_reset_head(self):
        head_kwargs = {"hidden_sizes": [32, 32]}
        feature_extractor_kwargs = {
            "architecture": "cnn",
            "n_convolutions": 2,
            "out_channels": [32, 64],
            "sizes": [8, 4],
            "activation": "relu",
        }
        dqn = DQN(
            num_actions=2,
            obs_size=(64, 64, 3),
            dueling=True,
            head_kwargs=head_kwargs,
            feature_extractor_kwargs=feature_extractor_kwargs,
        )
        dummy_input = torch.rand((5, 64, 64, 3))
        original_features = dqn.feature_extractor(dummy_input)
        original_pred = dqn(dummy_input)
        dqn.reset_head([64])
        new_features = dqn.feature_extractor(dummy_input)
        new_pred = dqn(dummy_input)

        assert torch.allclose(
            original_features, new_features
        ), "Features should be equal"
        assert ~torch.allclose(original_pred, new_pred), "Predictions should differ"

    def test_shrink_weights(self):
        dqn = DQN(num_actions=4, obs_size=3, dueling=True)
        dummy_input = torch.rand((5, 3))
        prev_head_params = deepcopy(list(dqn.head.parameters()))
        prev_adv_params = deepcopy(list(dqn.advantage.parameters()))
        prev_value_params = deepcopy(list(dqn.value.parameters()))
        dqn.shrink_weights(0.5, 0.0)
        for new_param, old_param in zip(
            dqn.head.parameters(), prev_head_params, strict=False
        ):
            assert torch.allclose(
                new_param, old_param * 0.5
            ), "Weights have not been shrunk."
        for new_param, old_param in zip(
            dqn.advantage.parameters(), prev_adv_params, strict=False
        ):
            assert torch.allclose(
                new_param, old_param * 0.5
            ), "Advantage weights have not been shrunk."
        for new_param, old_param in zip(
            dqn.value.parameters(), prev_value_params, strict=False
        ):
            assert torch.allclose(
                new_param, old_param * 0.5
            ), "Value weights have not been shrunk."

    def test_get_state(self):
        dqn = DQN(num_actions=4, obs_size=3, dueling=True)
        state_dict = dqn.state_dict()
        assert isinstance(state_dict, dict), "State dict should be a dictionary"
        assert "feature_extractor.layers.0.weight" in state_dict
        assert "head.0.weight" in state_dict
        assert "advantage.weight" in state_dict
        assert "value.weight" in state_dict

    def test_set_state(self):
        dummy_input = torch.rand((5, 3))
        dqn = DQN(num_actions=4, obs_size=3, dueling=True)
        baseline_pred = dqn(dummy_input)
        state_dict = dqn.state_dict()
        dqn2 = DQN(num_actions=4, obs_size=3, dueling=True)
        original_pred = dqn2(dummy_input)
        assert ~torch.allclose(
            baseline_pred, original_pred
        ), "Predictions should be different before loading"
        for p1, p2 in zip(dqn.parameters(), dqn2.parameters(), strict=False):
            assert ~torch.allclose(
                p1, p2
            ), "Parameters should be different before loading"
        dqn2.load_state_dict(state_dict)
        new_pred = dqn2(dummy_input)
        assert torch.allclose(
            baseline_pred, new_pred
        ), "Predictions should be equal after loading"
        for p1, p2 in zip(dqn.parameters(), dqn2.parameters(), strict=False):
            assert torch.allclose(p1, p2), "Parameters should be equal after loading"


class TestIQN:
    # TODO: test loading, it doesn't work
    def get_iqn(self):
        return IQN(4)


class TestSPRQN:
    pass
