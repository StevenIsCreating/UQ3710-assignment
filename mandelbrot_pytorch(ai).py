# mandelbrot_pytorch.py
import numpy as np
import matplotlib.pyplot as plt
import torch

# ---- Device configuration ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ---- Create complex plane grid ----
# Range [-2, 1] × [-1.3, 1.3], step size 0.005
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

# Convert to PyTorch tensors
x = torch.tensor(X, dtype=torch.float32, device=device)
y = torch.tensor(Y, dtype=torch.float32, device=device)

# Complex grid c
c = torch.complex(x, y)

# Initialize z and ns
z = torch.zeros_like(c)
ns = torch.zeros(x.shape, dtype=torch.float32, device=device)

# ---- Mandelbrot iteration ----
max_iter = 200
for i in range(max_iter):
    z = z * z + c
    mask = torch.abs(z) < 4.0
    ns += mask

# ---- Visualization function ----
def processFractal(a):
    """Convert iteration counts to a colored image"""
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([
        10 + 20 * np.cos(a_cyclic),
        30 + 50 * np.sin(a_cyclic),
        155 - 80 * np.cos(a_cyclic)
    ], 2)
    img[a == a.max()] = 0
    img = np.uint8(np.clip(img, 0, 255))
    return img

# ---- Plot ----
plt.figure(figsize=(12, 8))
plt.imshow(processFractal(ns.cpu().numpy()))
plt.axis('off')
plt.tight_layout()
plt.show()

