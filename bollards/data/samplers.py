import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler

from bollards.constants import LABEL_COL


def make_sampler(df: pd.DataFrame, generator=None, alpha: float = 0.5) -> WeightedRandomSampler:
    labels = df[LABEL_COL].astype(int)
    counts = labels.value_counts().to_dict()
    alpha = float(alpha)
    weights = labels.map(lambda y: (1.0 / counts[int(y)]) ** alpha).astype(np.float32).to_numpy()
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
