import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


x_center, y_center = -0.75, 0.0
zoom = 1.5
step = 0.001
max_iter = 500


x_range = zoom
y_range = zoom * 0.75 
Y, X = np.mgrid[
    y_center - y_range : y_center + y_range : step,
    x_center - x_range : x_center + x_range : step
]

x = torch.tensor(X, dtype=torch.float32, device=device)
y = torch.tensor(Y, dtype=torch.float32, device=device)
c = torch.complex(x, y)

z = torch.zeros_like(c)
ns = torch.zeros(x.shape, dtype=torch.float32, device=device)

for _ in range(max_iter):
    z = z * z + c
    mask = torch.abs(z) < 4.0
    ns += mask

def processFractal(a):
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([
        10 + 20 * np.cos(a_cyclic),
        30 + 50 * np.sin(a_cyclic),
        155 - 80 * np.cos(a_cyclic)
    ], 2)
    img[a == a.max()] = 0
    return np.uint8(np.clip(img, 0, 255))

plt.figure(figsize=(12, 8))
plt.imshow(processFractal(ns.cpu().numpy()))
plt.axis('off')
plt.tight_layout()
plt.show()
