import h5py

from hay_single_compartment import SimulationConfig, generate_dataset, validate_dataset
from hay_single_compartment.dataset import Normalization, SequenceWindowDataset


def test_small_dataset_roundtrip(tmp_path):
    path = tmp_path / "tiny.h5"
    config = SimulationConfig(
        duration_ms=5.0,
        warmup_ms=1.0,
        train_trajectories=2,
        validation_trajectories=1,
        test_trajectories=1,
    )
    report = generate_dataset(path, config)
    assert report["valid"]
    assert validate_dataset(path)["valid"]
    assert path.with_suffix(".manifest.json").exists()
    with h5py.File(path, "r") as handle:
        assert handle["train/states"].shape == (2, 51, 17)
        assert handle["test/inputs"].shape == (1, 50, 4)
        seeds = [
            int(seed)
            for split in ("train", "validation", "test")
            for seed in handle[f"{split}/trajectory_seeds"][...]
        ]
        assert len(seeds) == len(set(seeds))

    normalization = Normalization.from_h5(path)
    windows = SequenceWindowDataset(path, "train", normalization, sequence_length=10, stride=10)
    features, target = windows[0]
    assert features.shape == (10, 21)
    assert target.shape == (10, 17)
