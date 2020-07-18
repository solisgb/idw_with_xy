# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 12:03:01 2020

@author: solis
"""
import numpy as np
from numba import jit


def sort1(values0, ii, values):
    """
    void(float[:], int[:], float[:])
    sorts array values0 by integer in ii
    """
    for i in range(ii.size):
        values[i] = values0[ii[i]]


def idw1(p, dist, values, tiny):
    """
    float(float, float[:], float[:], float)
    interpolates 1 point
    """
    if dist[0] <= tiny:
        return values[0]
    dist_pow = np.power(dist, 2)
    x1 = np.sum(values/dist_pow)
    x2 = np.sum(1/dist_pow)
    return x1 / x2


@jit(nopython=True)
def sortn(values0, ii, values):
    """
    void(float[:], int[:,:], float[:,:])
    sorts array values0 by integer in ii
    """
    for i in range(ii.shape[0]):  # for each point
        for j in range(ii.shape[1]):
            values[i][j] = values0[ii[i, j]]


@jit(nopython=True)
def idwn(p, dist, values, tiny, zi):
    """
    void(float, float[:,:], float[:,:], float, float[:])
    interpolates several points
    """
    for i in range(dist.shape[0]):  # for each point
        if dist[i, 0] <= tiny:
            zi[i] = values[i, 0]
        dist_pow = np.power(dist[i], 2)
        x1 = np.sum(values[i]/dist_pow)
        x2 = np.sum(1/dist_pow)
        zi[i] = x1 / x2
