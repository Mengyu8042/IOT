# -*- coding: utf-8 -*-
"""
Implementation of the proposed iterative optimal transport method
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import ot


def IOTreg(X, Y, 
           degree=2, 
           reg_m=5e-3, lambda_=1e-3,
           outer_iter=100, inner_iter=100, stop_thr=1e-6,
           log=False, verbose=False):
    '''
    Transformation function estimation using the iterative optimal transport method.

    Parameters
    ----------
    X : moving feature points
    Y : fixed feature points
    degree : degree of polynomials
    reg_m : marginal relaxation parameter for UOT
    lambda_ : regularization parameter for transformation function
    outer_iter : maximun number of outer iterations
    inner_iter : maximun number of inner iterations
    stop_thr : stop threshold for outer iterations

    Returns
    -------
    T: transformation function coefficients

    '''
    N1 = X.shape[0]
    N2 = Y.shape[0]
    a = np.ones(N1) / N1
    b = np.ones(N2) / N2
    Xa = create_polynomial_matrix(X, degree)
    
    # Initilize the transformation function
    if degree == 1:
        param = np.array([1, 0, 0, 1, 0, 0])
    else:
        param = np.concatenate((np.array([1, 0, 0, 1]), np.zeros(4 * degree)))

    # Initilize the warped moving points and the transport plan
    V = X.copy()
    plan = np.ones((N1, N2)) / (N1 * N2)
    
    if log:
        log = {'err_plan': [], 'err_trans': [], 'plan': [], 'trans': []}
    
    for i in range(outer_iter):
        plan_prev = plan.copy()
        param_prev = param.copy()
        
        # Update the transport plan
        M = sqeuclidean_dist(V, Y)
        plan = ot.unbalanced.mm_unbalanced(a, b, M, reg_m=reg_m,
                                            numItermax=inner_iter, stopThr=1e-6)
        
        # Update the transformation function
        result = minimize(lambda x: costfun(x, X, Xa, Y, plan, lambda_), 
                          param, method='BFGS', options={'maxiter': inner_iter})
        param = result.x
        
        if degree == 1:
            T = param.reshape(3, 2)
        else:
            T = param.reshape(2 * degree + 2, 2)
        V =  Xa @ T

        err1 = np.sqrt(np.sum((plan - plan_prev) ** 2))
        err2 = np.sqrt(np.sum((param - param_prev) ** 2))
        if log:
            log['err_plan'].append(err1)
            log['err_trans'].append(err2)
            log['plan'].append(plan)
            log['trans'].append(T)
        if verbose:
            print('{:5d}|{:8e}|{:8e}|'.format(i, err1, err2))
        if np.max([err1, err2]) < stop_thr:
            break
        
    if log:
        dist = uot_distance_given_plan(a, b, V, Y, plan, reg_m)
        log['dist'] = dist
        return T, log
    else:
        return T
    
    
def IOTtrans(im1, im2, T, normal):
    '''
    Moving image transformation given the transformation function.

    Parameters
    ----------
    im1 : moving image
    im2 : fixed image
    T : transformation function (output of the IOTreg function)
    normal : normalization factors

    Returns
    -------
    im1_trans: warped moving image

    '''
    rows, cols = im1.shape[:2]
    degree = int((T.shape[0] - 2) / 2) if T.shape[0] > 3 else 1

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(rows), np.arange(cols))
    pixels = np.stack([x.ravel(), y.ravel()], axis=1)

    # Normalize and transform all pixels at once
    normalized_pixels = (pixels - normal['xm']) / normal['xscale']
    poly_matrix = create_polynomial_matrix(normalized_pixels, degree)
    pixels_trans = poly_matrix @ T
    pixels_trans = pixels_trans * normal['yscale'] + normal['ym']
    
    im1_trans = np.zeros_like(im2)
    for ii, pixel in enumerate(pixels):
        pixel_trans = pixels_trans[ii]
        x = pixel_trans[0]
        y = pixel_trans[1]
        x_round = np.round(x)
        y_round = np.round(y)
        if 0 <= x_round <= im2.shape[0] - 1 and 0 <= y_round <= im2.shape[1] - 1:
            im1_trans[int(x_round), int(y_round)] = im1[pixel[0], pixel[1]]
    
    if len(im1.shape) > 2:
        for ii in range(im1.shape[2]):
            idx_zero = np.where(im1_trans[:, :, ii] == 0)
            for jj in range(len(idx_zero[0])):
                x, y = idx_zero[0][jj], idx_zero[1][jj]
                if 1 <= x <= im1_trans.shape[0]-2 and 1 <= y <= im1_trans.shape[1]-2:
                    temp = np.array([im1_trans[x-1, y, ii], im1_trans[x+1, y, ii], 
                                     im1_trans[x, y-1, ii], im1_trans[x, y+1, ii]]).astype(np.float32)
                    if np.sum(temp > 0) >= 3:
                        im1_trans[x, y, ii] = int(np.mean(temp))
    
    return im1_trans


def add_column_ones(X):
    Xa = np.ones((X.shape[0], X.shape[1] + 1))
    Xa[:, :-1] = X
    return Xa


def create_polynomial_matrix(X, degree=2):
    # Add interaction terms
    if degree > 1:
        X = np.hstack((X, (X[:, 0] * X[:, 1])[:, None]))
        # Add polynomial terms
        for d in range(2, degree + 1):
            X = np.hstack((X, 
                           np.power(X[:, 0], d)[:, None],
                           np.power(X[:, 1], d)[:, None]))
    return add_column_ones(X)


def sqeuclidean_dist(Xs, Xt):
    if len(Xs.shape) == 1:
        Xs = Xs[:, None]
    if len(Xt.shape) == 1:
        Xt = Xt[:, None]
        
    F = cdist(Xs, Xt)**2
    return F


def costfun(param, X, Xa, Y, W, lambda_):
    T = param.reshape(Xa.shape[1], 2)
    Z = Xa @ T
    # Cost computation
    F = sqeuclidean_dist(Z, Y)
    E = np.sum(W * F)
    # Regularization
    E += lambda_ * np.mean((Z - X) ** 2)
    
    return E


def uot_distance_given_plan(a, b, X, Y, plan, reg_m):
    M = sqeuclidean_dist(X, Y)
    dist = np.sum(M * plan)
    a_star = np.sum(plan, axis=1) + 1e-15
    b_star = np.sum(plan, axis=0) + 1e-15
    kl1 = sum(np.multiply(a_star, np.log(a_star) - np.log(a)) - a_star + a)
    kl2 = sum(np.multiply(b_star, np.log(b_star) - np.log(b)) - b_star + b)
    dist += reg_m * kl1
    dist += reg_m * kl2
    return dist