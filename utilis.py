# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy.spatial import distance_matrix


#%% Data preprocessing
def norm_data(X, Y, normal=None):
    """
    Nomalize the data points to zero mean and unit covariance
    """
    x = X.copy()
    y = Y.copy()
    x = np.float64(x)
    y = np.float64(y)
    N = x.shape[0]
    M = y.shape[0]
    
    if normal is None:
        # Zero mean
        normal = {}
        normal['xm'] = np.mean(x, axis=0)
        normal['ym'] = np.mean(y, axis=0)
        x -= normal['xm']
        y -= normal['ym']
        # Unit covariance
        normal['xscale'] = np.sqrt(np.sum(x**2) / N)
        normal['yscale'] = np.sqrt(np.sum(y**2) / M)
        x /= normal['xscale']
        y /= normal['yscale']

    else:
        x -= normal['xm']
        y -= normal['ym']
        x /= normal['xscale']
        y /= normal['yscale']
    
    return x, y, normal


def convert_point_to_pixel(image, points):
    """
    Convert the representation from point coordinates to pixel indices.
    """
    pts = points.copy()
    image_h, image_w = image.shape[0:2]
    pts = np.round(pts)  # Round and convert to int
    pixels = pts.copy()
    pixels[:, 0] = pts[:, 1]
    pixels[:, 1] = pts[:, 0]
    # Ensure the pixel indices are within the image bounds
    pixels[:, 0] = np.clip(pixels[:, 0], 0, image_h - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, image_w - 1)
    return pixels


#%% Feature point detection
def canny_edge_detector(image, scale_high=2, scale_low=0.95):
    """
    Apply Canny edge detector with Otsu's method to determine parameters
    
    Return: binary image matrix, 255 represents edge, while 0 represents non-edge
    """
    high_thres, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_thres *= scale_high
    low_thres = scale_low * high_thres
    edge_map = cv2.Canny(image, low_thres, high_thres)
    return edge_map


def edge_points(edge_map, nsamp=100, rand_seed=1234):
    """
    Select keypoints from edges using Jitendra's sampling method.
    """
    points = np.column_stack(np.where(edge_map > 0))
    N = len(points)
    k = 3
    Nstart = min(k * nsamp, N)

    ind0 = np.random.default_rng(seed=rand_seed).permutation(N)
    ind0 = ind0[:Nstart]
    points = points[ind0]
    d2 = distance_matrix(points, points)**2
    np.fill_diagonal(d2, np.inf)

    while len(d2) > nsamp:
        # Find the closest pair
        J = np.argmin(np.min(d2, axis=1))
        # Remove one of the points
        points = np.delete(points, J, axis=0)
        d2 = np.delete(d2, J, axis=0)
        d2 = np.delete(d2, J, axis=1)
        
    return points


#%% Image visualization
def create_checkerboard(image1, image2, checkerboard_size=(4, 4)):
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
    
    if image1.shape != image2.shape:
        raise ValueError("Two images should have the same size.")
    
    row_div = image1.shape[0] // checkerboard_size[0]
    col_div = image1.shape[1] // checkerboard_size[1]
    checkerboard = np.zeros_like(image1)
    for i in range(checkerboard_size[0]):
        for j in range(checkerboard_size[1]):
            if (i + j) % 2:
                checkerboard[i*row_div:(i+1)*row_div, j*col_div:(j+1)*col_div] = image1[i*row_div:(i+1)*row_div, j*col_div:(j+1)*col_div]
            else:
                checkerboard[i*row_div:(i+1)*row_div, j*col_div:(j+1)*col_div] = image2[i*row_div:(i+1)*row_div, j*col_div:(j+1)*col_div]
    return checkerboard


def create_overlap_edge(edge1, edge2):
    rgb_image = np.ones((*edge1.shape, 3), dtype=np.uint8) * 255

    for i in range(edge1.shape[0]):
        for j in range(edge1.shape[1]):
            if edge1[i, j] == 255 and edge2[i, j] == 255:
                rgb_image[i, j] = [255, 0, 255]  # purple
            elif edge1[i, j] == 255:
                rgb_image[i, j] = [0, 0, 255]  # blue
            elif edge2[i, j] == 255:
                rgb_image[i, j] = [255, 0, 0]  # red
                
    return rgb_image


def plot_points_on_image(original_image, points, radius=2, color=(255, 0, 0), thickness=-1):
    if len(original_image.shape) == 2:
        original_image = np.stack([original_image] * 3, axis=-1)
    overlayed_image = np.copy(original_image)
    points = np.round(points)
    points = points.astype(np.int32)
    for i, j in points:
        cv2.circle(overlayed_image, (j, i), radius=radius, color=color, thickness=thickness)
    return overlayed_image