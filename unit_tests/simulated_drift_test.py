import numpy as np
from PIL import Image

import drift_detection
import simulated_drift


def _write_image(path, color):
    image = Image.new("RGB", (8, 8), color=color)
    image.save(path)
    return path


def test_severe_shift_changes_brightness_or_contrast():
    rng = np.random.default_rng(42)
    image = Image.new("RGB", (8, 8), color=(90, 110, 130))

    shifted = simulated_drift.apply_simulated_shift(image, "severe", rng)

    clean_features = drift_detection.extract_image_features(image, "clean.png")
    shifted_features = drift_detection.extract_image_features(shifted, "shifted.png")

    brightness_changed = not np.isclose(
        clean_features["brightness"],
        shifted_features["brightness"],
    )
    contrast_changed = not np.isclose(
        clean_features["contrast"],
        shifted_features["contrast"],
    )
    assert brightness_changed or contrast_changed


def test_grayscale_shift_makes_channel_means_equal():
    rng = np.random.default_rng(42)
    image = Image.new("RGB", (8, 8), color=(40, 120, 220))

    shifted = simulated_drift.apply_simulated_shift(image, "grayscale", rng)
    features = drift_detection.extract_image_features(shifted, "shifted.png")

    assert np.isclose(features["mean_r"], features["mean_g"])
    assert np.isclose(features["mean_g"], features["mean_b"])


def test_build_simulated_drift_frames_keeps_equal_row_counts(tmp_path):
    paths = [
        _write_image(tmp_path / "one.png", (40, 80, 120)),
        _write_image(tmp_path / "two.png", (120, 80, 40)),
    ]

    reference_data, current_data = simulated_drift.build_simulated_drift_frames(
        image_paths=paths,
        shift_name="dark",
        seed=42,
    )

    assert len(reference_data) == 2
    assert len(current_data) == 2
    assert current_data["brightness"].mean() < reference_data["brightness"].mean()
