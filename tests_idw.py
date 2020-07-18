# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:48:21 2020

@author: solis
"""
import numpy as np
from scipy import spatial
import idw


def test01(xit1, datat1, n):
    xi = np.array(xit1, np.float32)
    zi = np.empty((xi.shape[0]))
    data = np.array(datat1, np.float32)

    tree = spatial.cKDTree(data[:,[0,1]])
    dist, ii = tree.query(xi, k=min(data.shape[0],n))
    values0 = data[:,2]
    values = np.empty((dist.shape), np.float32)

    idw.sortn(values0, ii, values)
    idw.idwn(2., dist, values, 0.05, zi)
    print(zi)


def test02(xit1, datat1, n):
    xi = np.array(xit1, np.float32)
    zi = np.empty((xi.shape[0]))
    data = np.array(datat1, np.float32)

    tree = spatial.cKDTree(data[:,[0,1]])
    dist, ii = tree.query(xi, k=min(data.shape[0],n))
    values0 = data[:,2]
    values = np.empty((dist.shape), np.float32)

    idw.sort1(values0, ii, values)
    zi = idw.idw1(2., dist, values, 0.05)
    print(zi)


if __name__ == "__main__":
    xit1 =[[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
    datat1 = [[0,0,0],[0,1,0],[1,0,0],[1,1,0],[0.5,0.5,1]]
    npoints_nearest = 7

    # several points
    test01(xit1, datat1, npoints_nearest)
    # particular case -one point-
    test01([xit1[0]], datat1, npoints_nearest)
    # another way for one point
    test02(xit1[0], datat1, npoints_nearest)
    for point in xit1:
        test02(point, datat1, npoints_nearest)

