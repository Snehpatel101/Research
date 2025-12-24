import numpy as np

from src.stages.ga_optimize.fitness import calculate_fitness


def test_calculate_fitness_penalizes_imbalanced_labels():
    labels = np.array([1] * 90 + [-1] * 1 + [0] * 9, dtype=np.int8)
    bars_to_hit = np.ones_like(labels, dtype=np.int32)
    mae = np.zeros_like(labels, dtype=np.float32)
    mfe = np.zeros_like(labels, dtype=np.float32)

    fitness = calculate_fitness(
        labels=labels,
        bars_to_hit=bars_to_hit,
        mae=mae,
        mfe=mfe,
        horizon=5,
        atr_mean=1.0,
        symbol="MES",
    )

    assert fitness < -800.0
