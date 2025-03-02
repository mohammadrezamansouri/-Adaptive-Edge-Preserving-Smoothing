import numpy as np
from scipy.ndimage import convolve

def ubf(img, s):
    """
    Applies a uniform box filter with symmetric padding to the image.
    Works for both grayscale and color images.
    """
    kernel = np.ones((s, s))
    if img.ndim == 2:  # Grayscale image
        return convolve(img, kernel, mode='reflect')
    elif img.ndim == 3:  # Color image
        out = np.empty_like(img)
        for c in range(img.shape[2]):
            out[..., c] = convolve(img[..., c], kernel, mode='reflect')
        return out 

def ISESFilter(x, s, p, e):
    """
    Applies the ISES filter for adaptive edge-preserving smoothing.

    Parameters:
    x : Input image (grayscale or color)
    s : Patch size (must be an odd number)
    p : Edge sharpening parameter
    e : Regularization parameter for contrast smoothing

    Returns:
    z : Filtered output image
    """
    if s % 2 == 0:
        raise ValueError("Patch size 's' must be an odd number.")
    
    # Compute the mean (mu) within each patch
    mu = ubf(x, s) / (s**2)

    # Compute the variance within each patch
    if x.ndim == 3:  # Color image
        x2 = x**2
        mu_x2 = ubf(x2, s) / (s**2)
        variance = np.mean(mu_x2 - mu**2, axis=2)
    else:  # Grayscale image
        x2 = x**2
        mu_x2 = ubf(x2, s) / (s**2)
        variance = mu_x2 - mu**2

    # Compute weights using variance (with a max limit)
    w = np.minimum(1.0 / ((variance + e)**p), 1e15)

    # Sum of weights within each patch
    w_sum = ubf(w, s)

    # Compute the final filtered output
    if x.ndim == 3:
        weighted_mu = ubf(w[..., np.newaxis] * mu, s)
        z = weighted_mu / w_sum[..., np.newaxis]
    else:
        weighted_mu = ubf(w * mu, s)
        z = weighted_mu / w_sum
        
    return z 
