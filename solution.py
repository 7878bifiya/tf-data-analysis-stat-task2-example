import pandas as pd
import numpy as np
from scipy.stats import t


chat_id = 541133397 # Ваш chat ID, не меняйте название переменной

def solution(confidence: float, measurements: np.array) -> tuple:
    n = len(measurements)
    mean = np.mean(measurements)
    std = np.std(measurements, ddof=1)
    t_crit = t.ppf((1 + confidence) / 2, n - 1)
    interval = t_crit * (std / np.sqrt(n))
    return (mean - interval, mean + interval)
