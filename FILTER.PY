import numpy as np
import matplotlib.pyplot as plt
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

def compute_ises_weights(x, s, p, e):
    """
    Computes the ISES filter weights (w) without actually filtering the image.
    """
    if s % 2 == 0:
        raise ValueError("Patch size 's' must be an odd number.")
    
    mu = ubf(x, s) / (s**2)

    if x.ndim == 3:  # Color image (میانگین کانال‌ها)
        x2 = x**2
        mu_x2 = ubf(x2, s) / (s**2)
        variance = np.mean(mu_x2 - mu**2, axis=2)
    else:  # Grayscale
        x2 = x**2
        mu_x2 = ubf(x2, s) / (s**2)
        variance = mu_x2 - mu**2

    # Compute adaptive weights
    w = np.minimum(1.0 / ((variance + e)**p), 1e15)
    return w

def visualize_ises_weights(img, s=5, p=1, e=0.01):
    """
    Computes and visualizes ISES weights for a given image.
    """
    if img.ndim == 3:  # تبدیل به grayscale برای راحتی کار
        img_gray = np.mean(img, axis=2)
    else:
        img_gray = img

    w = compute_ises_weights(img_gray, s, p, e)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original Image (Grayscale)')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(w, cmap='hot')
    plt.colorbar(label='ISES Weight')
    plt.title(f'ISES Filter Weights (s={s}, p={p}, e={e})')
    plt.axis('off')

    plt.show()

# مثال تست:
from skimage.data import camera
image = camera()  # تصویر تست

visualize_ises_weights(image, s=7, p=2, e=0.01)
