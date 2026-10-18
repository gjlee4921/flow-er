"""A disc-shaped probability map gives a polygon close to the circle."""

import numpy as np

from flow_er.ml.unet.polygon import EDGE_OUT, polygon_area, prob_to_polygon


def test_disc_gives_circle():
    yy, xx = np.mgrid[0:101, 0:101]
    r = np.hypot(xx - 40, yy - 60)
    prob = 1 / (1 + np.exp(r - 20))          # 0.5 exactly at radius 20 pixels
    xy = prob_to_polygon(prob)
    assert xy is not None and len(xy) >= 8
    radius = np.hypot(xy[:, 0] * 100 - 40, xy[:, 1] * 100 - 60)
    assert np.abs(radius - 20).max() < 1.0
    assert abs(polygon_area(xy * 100) - np.pi * 20**2) / (np.pi * 20**2) < 0.03


def test_edge_gate_closes_and_empty_map_gives_none():
    prob = np.zeros((101, 101))
    prob[:, 70:] = 1.0                       # touches three edges
    xy = prob_to_polygon(prob)
    assert xy is not None
    assert xy[:, 0].max() == EDGE_OUT[1] and abs(xy[:, 0].min() - 0.695) < 0.01   # past the edge
    assert prob_to_polygon(np.zeros((101, 101))) is None
