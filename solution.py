import pandas as pd
import numpy as np

from scipy.stats import norm, expon, t


chat_id = 541133397 # Ваш chat ID, не меняйте название переменной


def solution(alpha: float, x: np.array) -> tuple:
    n = len(x)
    t_alpha = t.ppf(1-alpha/2, n-1)
    z_alpha = norm.ppf(1-alpha/2)
    t_dist = t.rvs(n-1, size=100000)
    z_dist = norm.rvs(size=100000)
    e_dist = expon.rvs(scale=0.5, size=(100000, n))
    a_dist = np.zeros((100000,))
    for i in range(100000):
        s = x + e_dist[i,:]
        t = np.repeat(20, n)
        while np.any(t==0):
            t[t==0] = expon.rvs(scale=0.5, size=np.sum(t==0))
        a_dist[i] = 2*np.mean(s)/np.mean(t)**2
    a_mean = np.mean(a_dist)
    a_std = np.std(a_dist)
    a_left = a_mean - t_alpha*a_std/np.sqrt(n)
    a_right = a_mean + t_alpha*a_std/np.sqrt(n)
    z_left = a_mean - z_alpha*a_std/np.sqrt(n)
    z_right = a_mean + z_alpha*a_std/np.sqrt(n)
    return (a_left, a_right), (z_left, z_right)
