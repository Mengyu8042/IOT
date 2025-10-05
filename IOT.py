# -*- coding: utf-8 -*-
"""
Implementation of the proposed iterative optimal transport method
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import ot


def IOTreg(
    X, Y,
    degree=2,
    reg_m=5e-3, lambda_=1e-3,
    outer_iter=100, inner_iter=100, stop_thr=1e-6,
    log=False, verbose=False,
):
    """
    Implementation of Iterative Optimal Transport.

    Parameters
    ----------
    X : (N1, d) moving keypoints (d=2 or 3)
    Y : (N2, d) fixed keypoints
    degree : int >= 1, polynomial degree
    reg_m : float, UOT marginal relaxation
    lambda_ : float, transformation regularization
    outer_iter, inner_iter, stop_thr : optimization hyper-params
    log, verbose : debug flags

    Returns
    -------
    T : (P, d) polynomial coefficient matrix,
        where P = number of polynomial basis terms produced by create_polynomial_matrix.
    (optionally) log dict
    """
    N1 = X.shape[0]
    N2 = Y.shape[0]
    d = X.shape[1]
    a = np.ones(N1) / N1
    b = np.ones(N2) / N2
    Xa = create_polynomial_matrix(X, degree) # (N1, P)
    P = Xa.shape[1]
    
    # ---- Initialize T (identity mapping) ----
    # Feature layout: [x1..xd, interactions, powers, 1]
    # Set first d rows to identity, others (including bias row) to 0
    T0 = np.zeros((P, d), dtype=float)
    T0[:d, :d] = np.eye(d)
    param = T0.ravel()

    # Init warped points & transport plan
    V = X.copy()
    plan = np.ones((N1, N2)) / (N1 * N2)
    
    if log:
        log = {'err_plan': [], 'err_trans': [], 'plan': [], 'trans': []}
    
    for i in range(outer_iter):
        plan_prev = plan.copy()
        param_prev = param.copy()
        
        # ---- Update transport plan (UOT) ----
        M = sqeuclidean_dist(V, Y)                       # (N1, N2)
        plan = ot.unbalanced.mm_unbalanced(
            a, b, M, reg_m=reg_m, numItermax=inner_iter, stopThr=1e-6
        )
        
        # ---- Update transformation T ----
        res = minimize(
            lambda p: costfun(p, X, Xa, Y, plan, lambda_),
            param, method='BFGS', options={'maxiter': inner_iter}
        )
        param = res.x
        T = param.reshape(P, d)

        # update warped points
        V = Xa @ T                                       # (N1, d)

        # ---- Stopping ----
        err1 = np.sqrt(np.sum((plan - plan_prev) ** 2))
        err2 = np.sqrt(np.sum((param - param_prev) ** 2))
        if log:
            log['err_plan'].append(err1)
            log['err_trans'].append(err2)
            log['plan'].append(plan)
            log['trans'].append(T)
        if verbose:
            print(f"{i:5d}|{err1:8.3e}|{err2:8.3e}|")
        if max(err1, err2) < stop_thr:
            break
        
    if log:
        dist = uot_distance_given_plan(a, b, V, Y, plan, reg_m)
        log['dist'] = dist
        return T, log
    else:
        return T
    
    
def IOTtrans(im1, im2, T, degree, normal):
    """
    Warp a moving image (im1) to the fixed image space (im2) using
    a forward polynomial transform T estimated by IOT.

    Parameters
    ----------
    im1 : np.ndarray
        Moving image. 2D grayscale (H,W), 2D color (H,W,C with C<=4), or 3D volume (D,H,W).
    im2 : np.ndarray
        Fixed image (used for output shape). 
    T : np.ndarray
        (P, d) polynomial coefficients (output of IOTreg)
    degree : int >= 1, polynomial degree
    normal : dict 
        Normalization parameters with 'xm','xscale','ym','yscale'

    Returns
    -------
    out : ndarray
        Warped image, same shape as im2 (and same #channels if 2D color).
    """
    def is_2d(a):
        return (a.ndim == 2) or (a.ndim == 3 and a.shape[-1] <= 4)

    if is_2d(im1):
        # =========================
        #           2D
        # =========================
        if im1.ndim == 2:
            src = im1[..., None]
        else:
            src = im1

        Hs, Ws, Cs = src.shape
        Ht, Wt = im2.shape[:2]
        out = np.zeros((Ht, Wt, Cs), dtype=src.dtype)

        # Build a source grid (row, col) in im1 coordinates
        rr, cc = np.meshgrid(np.arange(Hs), np.arange(Ws), indexing='ij')  # (Hs,Ws)
        grid_src = np.stack([rr.ravel(), cc.ravel()], axis=1)              # (N,2)

        # Normalize source coords -> apply polynomial -> map to target coords
        grid_n = (grid_src - normal['xm']) / normal['xscale']              # (N,2)
        Phi = create_polynomial_matrix(grid_n, degree)                     # (N,P)
        grid_tgt = Phi @ T                                                 # (N,2) normalized
        grid_tgt = grid_tgt * normal['yscale'] + normal['ym']              # (N,2) in target

        # Round to nearest integer target indices
        tgt_r = np.rint(grid_tgt[:, 0]).astype(int)  # rows (y)
        tgt_c = np.rint(grid_tgt[:, 1]).astype(int)  # cols (x)

        # Valid mask within target image bounds
        valid = (tgt_r >= 0) & (tgt_r < Ht) & (tgt_c >= 0) & (tgt_c < Wt)
        src_r = rr.ravel()[valid]
        src_c = cc.ravel()[valid]
        tr = tgt_r[valid]
        tc = tgt_c[valid]

        # Splat per channel
        for ch in range(Cs):
            out[tr, tc, ch] = src[src_r, src_c, ch]

        # If target is strictly gray (2D), drop channel axis
        if im2.ndim == 2 and Cs == 1:
            out = out[..., 0]

        return out
        
    else:
        # =========================
        #           3D
        # =========================
        Ds, Hs, Ws = im1.shape
        Dt, Ht, Wt = im2.shape

        out = np.zeros_like(im2)

        # Build a source grid (z,y,x) in im1 coordinates
        zz, yy, xx = np.meshgrid(np.arange(Ds), np.arange(Hs), np.arange(Ws), indexing='ij')
        grid_src = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)  # (N,3)

        # Normalize -> polynomial -> target coords
        grid_n = (grid_src - normal['xm']) / normal['xscale']              # (N,3)
        Phi = create_polynomial_matrix(grid_n, degree)                     # (N,P)
        grid_tgt = Phi @ T                                                 # (N,3) normalized
        grid_tgt = grid_tgt * normal['yscale'] + normal['ym']              # (N,3) target

        # Round to nearest integer target indices
        tz = np.rint(grid_tgt[:, 0]).astype(int)   # z
        ty = np.rint(grid_tgt[:, 1]).astype(int)   # y
        tx = np.rint(grid_tgt[:, 2]).astype(int)   # x

        # Valid mask within target volume bounds
        valid = (tz >= 0) & (tz < Dt) & (ty >= 0) & (ty < Ht) & (tx >= 0) & (tx < Wt)
        sz = zz.ravel()[valid]
        sy = yy.ravel()[valid]
        sx = xx.ravel()[valid]
        tz = tz[valid]
        ty = ty[valid]
        tx = tx[valid]

        # Splat scalar voxels
        out[tz, ty, tx] = im1[sz, sy, sx]

        return out


def create_polynomial_matrix(X, degree):
    """
    Build a polynomial feature matrix for dimension d (2 or 3).
    Column order:
    [x1..xd,  xi*xj (i<j),  xi^p (p=2..degree for each i),  1]
    2D case: (x, y, xy, x^2, y^2, x^3, y^3, ..., 1)
    """
    N, d = X.shape

    feats = [X]  # x1..xd
    if degree > 1:
        # pairwise interactions xi*xj
        inter = []
        for i in range(d):
            for j in range(i + 1, d):
                inter.append((X[:, i] * X[:, j])[:, None])
        if len(inter) > 0:
            feats.append(np.hstack(inter))
        # powers xi^p, p=2..degree
        pow_terms = []
        for p in range(2, degree + 1):
            pow_terms.append(np.stack([X[:, i] ** p for i in range(d)], axis=1))
        if len(pow_terms) > 0:
            feats.append(np.hstack(pow_terms))

    # constant 1
    feats.append(np.ones((N, 1), dtype=float))

    return np.hstack(feats)  # (N, P)


def sqeuclidean_dist(Xs, Xt):
    Xs = np.atleast_2d(Xs)
    Xt = np.atleast_2d(Xt)
    return cdist(Xs, Xt) ** 2


def costfun(param, X, Xa, Y, W, lambda_):
    P, d = Xa.shape[1], X.shape[1]
    T = param.reshape(P, d)
    Z = Xa @ T                             # (N1,d)
    F = sqeuclidean_dist(Z, Y)            # (N1,N2)
    E = np.sum(W * F)
    # regularization
    E += lambda_ * np.mean((Z - X) ** 2)
    return E


def uot_distance_given_plan(a, b, X, Y, plan, reg_m):
    M = sqeuclidean_dist(X, Y)
    dist = np.sum(M * plan)
    a_star = np.sum(plan, axis=1) + 1e-15
    b_star = np.sum(plan, axis=0) + 1e-15
    kl1 = sum(np.multiply(a_star, np.log(a_star) - np.log(a)) - a_star + a)
    kl2 = sum(np.multiply(b_star, np.log(b_star) - np.log(b)) - b_star + b)
    return dist + reg_m * (kl1 + kl2)