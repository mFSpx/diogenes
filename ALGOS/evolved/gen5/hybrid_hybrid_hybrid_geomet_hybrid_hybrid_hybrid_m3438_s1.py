# DARWIN HAMMER — match 3438, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m155_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_distri_m1385_s3.py (gen4)
# born: 2026-05-29T23:50:15Z

import numpy as np

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def geometric_product(a, b):
    result = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            result[blade_res] = result.get(blade_res, 0.0) + sign * coef_a * coef_b
    return {blade: c for blade, c in result.items() if abs(c) > 1e-12}

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)

def ssim(x, y, dynamic_range=255.0, k1=0.01, k2=0.03):
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) *
                                                             (sigma_x ** 2 + sigma_y ** 2 + c2))

def bayesian_update(prior, likelihood):
    unnorm = {}
    for blade, p in prior.items():
        l = likelihood.get(blade, 0.0)
        unnorm[blade] = p * l
    total = sum(unnorm.values())
    if total == 0.0:
        n = len(unnorm)
        return {blade: 1.0 / n for blade in unnorm}
    return {blade: v / total for blade, v in unnorm.items()}

def vector_to_multivector(v):
    return {frozenset({int(i)}): float(coeff) for i, coeff in enumerate(v)}

def multivector_to_vector(mv, dim):
    vec = np.zeros(dim, dtype=float)
    for blade, coeff in mv.items():
        if len(blade) == 1:
            idx = next(iter(blade))
            if idx < dim:
                vec[idx] = coeff
    return vec

def hybrid_step(W, x, prior_mv, target=None, lr=0.1):
    pred = W @ x
    pred_mv = vector_to_multivector(pred)
    likelihood_mv = geometric_product(prior_mv, pred_mv)
    sim = ssim(pred, x if target is None else target)
    sim = max(0.0, min(1.0, sim))
    scaled_likelihood = {blade: coeff * sim for blade, coeff in likelihood_mv.items()}
    posterior_mv = bayesian_update(prior_mv, scaled_likelihood)
    posterior_confidence = max(posterior_mv.values())
    ttt_g = ttt_grad(W, x, target)
    W_update = lr * posterior_confidence * ttt_g
    W_new = W - W_update
    return W_new, posterior_mv