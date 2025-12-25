import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler

from bollards.constants import LABEL_COL


def make_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    counts = df[LABEL_COL].value_counts().to_dict()
    weights = df[LABEL_COL].map(lambda y: 1.0 / counts[int(y)]).astype(np.float32).to_numpy()
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
