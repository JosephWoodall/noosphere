"""
Tests for the real-EEG evaluation pipeline.

Fast tests (no network / no download):
  - Intent mapping math
  - Channel selection logic
  - encode_trials shape contract

Integration test (uses locally cached PhysionetMI data, marked slow):
  - load_trials returns correct shape and labels
  - Full encode → classify pipeline on 1 subject
"""

from __future__ import annotations

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Intent mapping
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntentMapping:
    def setup_method(self):
        from v2_digital_self_replication.data.intent_mapping import (
            CLASS_TO_INTENT, CLASSES, INTENT_MATRIX,
            accuracy, confusion_matrix,
            labels_to_intents, predict_class, predict_classes,
        )
        self.CLASS_TO_INTENT  = CLASS_TO_INTENT
        self.CLASSES          = CLASSES
        self.INTENT_MATRIX    = INTENT_MATRIX
        self.accuracy         = accuracy
        self.confusion_matrix = confusion_matrix
        self.labels_to_intents = labels_to_intents
        self.predict_class    = predict_class
        self.predict_classes  = predict_classes

    def test_intent_vectors_are_unit_bounded(self):
        for cls, vec in self.CLASS_TO_INTENT.items():
            assert np.all(np.abs(vec) <= 1.0), f"{cls} intent out of [-1, 1]"

    def test_rest_is_zero(self):
        assert np.all(self.CLASS_TO_INTENT["rest"] == 0.0)

    def test_left_right_are_antisymmetric_on_yaw(self):
        l = self.CLASS_TO_INTENT["left_hand"]
        r = self.CLASS_TO_INTENT["right_hand"]
        assert l[0] < 0 and r[0] > 0
        assert abs(l[0]) == abs(r[0])

    def test_predict_class_exact_recovery(self):
        for cls, vec in self.CLASS_TO_INTENT.items():
            pred = self.predict_class(vec)
            assert pred == cls, f"predict_class({cls}) returned {pred}"

    def test_predict_classes_vectorised(self):
        vecs  = np.stack(list(self.CLASS_TO_INTENT.values()))
        preds = self.predict_classes(vecs)
        assert preds == list(self.CLASS_TO_INTENT.keys())

    def test_predict_class_noisy_still_correct(self):
        rng = np.random.default_rng(0)
        for cls, vec in self.CLASS_TO_INTENT.items():
            if cls == "rest":
                continue   # rest is zero — noise can push it anywhere
            noisy = vec + rng.normal(0, 0.1, 6).astype(np.float32)
            pred  = self.predict_class(noisy)
            assert pred == cls, f"Noisy predict_class({cls}) returned {pred}"

    def test_labels_to_intents_shape(self):
        labels = ["left_hand", "right_hand", "rest"]
        intents = self.labels_to_intents(labels)
        assert intents.shape == (3, 6)
        np.testing.assert_array_equal(intents[2], 0.0)

    def test_labels_to_intents_unknown_label(self):
        intents = self.labels_to_intents(["unknown_class"])
        np.testing.assert_array_equal(intents[0], 0.0)

    def test_accuracy_perfect(self):
        labels = ["left_hand", "right_hand", "feet"]
        assert self.accuracy(labels, labels) == pytest.approx(1.0)

    def test_accuracy_zero(self):
        true  = ["left_hand", "left_hand"]
        pred  = ["right_hand", "right_hand"]
        assert self.accuracy(true, pred) == pytest.approx(0.0)

    def test_accuracy_half(self):
        true = ["left_hand", "right_hand"]
        pred = ["left_hand", "left_hand"]
        assert self.accuracy(true, pred) == pytest.approx(0.5)

    def test_confusion_matrix_diagonal(self):
        labels = ["left_hand", "right_hand", "feet"]
        cm     = self.confusion_matrix(labels, labels, classes=labels)
        np.testing.assert_array_equal(cm, np.eye(3, dtype=int))

    def test_intent_matrix_shape(self):
        n = len(self.CLASSES)
        assert self.INTENT_MATRIX.shape == (n, 6)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Channel selection contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestChannelSelection:
    def test_all_21_present_in_physionet(self):
        from v2_digital_self_replication.data.real_eeg import TARGET_CHANNELS

        # PhysionetMI channel list (verified from live dataset)
        physionet_channels = [
            'FC5','FC3','FC1','FCz','FC2','FC4','FC6',
            'C5','C3','C1','Cz','C2','C4','C6',
            'CP5','CP3','CP1','CPz','CP2','CP4','CP6',
            'Fp1','Fpz','Fp2','AF7','AF3','AFz','AF4','AF8',
            'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
            'FT7','FT8','T7','T8','T9','T10','TP7','TP8',
            'P7','P5','P3','P1','Pz','P2','P4','P6','P8',
            'PO7','PO3','POz','PO4','PO8',
            'O1','Oz','O2','Iz','STIM',
        ]
        for ch in TARGET_CHANNELS:
            assert ch in physionet_channels, f"{ch} missing from PhysionetMI"

    def test_target_channels_length(self):
        from v2_digital_self_replication.data.real_eeg import TARGET_CHANNELS
        assert len(TARGET_CHANNELS) == 21

    def test_target_channels_order_matches_config(self):
        from v2_digital_self_replication.data.real_eeg import TARGET_CHANNELS
        from v2_digital_self_replication.config import EEG_CHANNELS_21
        assert TARGET_CHANNELS == EEG_CHANNELS_21


# ═══════════════════════════════════════════════════════════════════════════════
# 3. encode_trials contract (no download — uses synthetic EEG as a stand-in)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncodeTrials:
    """
    encode_trials() processes (n_trials, T, 21) arrays through the twin.
    We use random arrays to test shape contracts without loading real data.
    """

    @pytest.fixture
    def twin(self):
        from v2_digital_self_replication.agent.digital_twin import DigitalTwin
        t = DigitalTwin()
        t.reset_state()
        return t

    def test_output_shapes(self, twin):
        from v2_digital_self_replication.data.real_eeg import encode_trials
        n_trials, T = 4, 256
        eeg = np.random.randn(n_trials, T, 21).astype(np.float32)
        latents, dof = encode_trials(twin, eeg)
        assert latents.shape == (n_trials, twin.cfg.encoder.d_model)
        assert dof.shape     == (n_trials, 6)

    def test_latents_are_finite(self, twin):
        from v2_digital_self_replication.data.real_eeg import encode_trials
        eeg = np.random.randn(3, 128, 21).astype(np.float32)
        latents, _ = encode_trials(twin, eeg)
        assert np.all(np.isfinite(latents))

    def test_dof_in_valid_range(self, twin):
        from v2_digital_self_replication.data.real_eeg import encode_trials
        eeg = np.random.randn(3, 128, 21).astype(np.float32)
        _, dof = encode_trials(twin, eeg)
        # DOF values should be bounded (tanh output through decoder)
        assert np.all(np.abs(dof) <= 2.0)

    def test_different_trials_give_different_latents(self, twin):
        from v2_digital_self_replication.data.real_eeg import encode_trials
        rng = np.random.default_rng(42)
        eeg = rng.standard_normal((2, 256, 21)).astype(np.float32)
        eeg[0] *= 10.0   # very different amplitude
        latents, _ = encode_trials(twin, eeg)
        assert not np.allclose(latents[0], latents[1])

    def test_tail_samples_parameter(self, twin):
        from v2_digital_self_replication.data.real_eeg import encode_trials
        eeg = np.random.randn(2, 256, 21).astype(np.float32)
        l16, _ = encode_trials(twin, eeg, tail_samples=16)
        l64, _ = encode_trials(twin, eeg, tail_samples=64)
        # Shapes must match regardless of tail_samples
        assert l16.shape == l64.shape


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Integration test — requires cached PhysionetMI data
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPhysionetIntegration:
    """
    These tests download / use cached PhysionetMI data.
    Run with:  pytest -m slow   or   pytest --run-slow
    """

    def test_load_trials_shape(self):
        from v2_digital_self_replication.data.real_eeg import load_trials
        eeg, labels = load_trials(
            subjects=[1],
            classes=["left_hand", "right_hand"],
            tmin=0.5, tmax=2.5,
        )
        T_expected = int((2.5 - 0.5) * 256) + 1  # at 256 Hz
        assert eeg.ndim == 3
        assert eeg.shape[2] == 21
        assert abs(eeg.shape[1] - T_expected) <= 5   # allow ±5 for resampling
        assert len(labels) == eeg.shape[0]

    def test_load_trials_labels_are_valid(self):
        from v2_digital_self_replication.data.real_eeg import load_trials, VALID_CLASSES
        _, labels = load_trials(
            subjects=[1],
            classes=["left_hand", "right_hand"],
            tmin=0.5, tmax=2.5,
        )
        for lbl in labels:
            assert lbl in VALID_CLASSES, f"Unexpected label: {lbl}"

    def test_load_trials_eeg_is_finite(self):
        from v2_digital_self_replication.data.real_eeg import load_trials
        eeg, _ = load_trials(subjects=[1], classes=["left_hand", "rest"],
                             tmin=0.5, tmax=2.5)
        assert np.all(np.isfinite(eeg))

    def test_full_pipeline_runs(self):
        """Zero-shot pipeline produces valid accuracy on 1 subject."""
        from v2_digital_self_replication.agent.digital_twin import DigitalTwin
        from v2_digital_self_replication.data.real_eeg import encode_trials, load_trials
        from v2_digital_self_replication.data.intent_mapping import (
            accuracy, predict_classes,
        )

        eeg, labels = load_trials(
            subjects=[1],
            classes=["left_hand", "right_hand"],
            tmin=1.0, tmax=3.0,
        )
        twin = DigitalTwin()
        twin.reset_state()
        _, dof = encode_trials(twin, eeg)
        preds  = predict_classes(dof)
        acc    = accuracy(labels, preds)

        # Basic sanity: accuracy is a valid probability
        assert 0.0 <= acc <= 1.0
        # Shapes are consistent
        assert len(preds) == len(labels)
