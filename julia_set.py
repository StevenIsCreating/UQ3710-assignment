import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


c_fixed = complex(-0.7, 0.27015) 
step = 0.005
max_iter = 300

Y, X = np.mgrid[-1.3:1.3:step, -2:2:step]
x = torch.tensor(X, dtype=torch.float32, device=device)
y = torch.tensor(Y, dtype=torch.float32, device=device)
z = torch.complex(x, y)
c = torch.tensor(c_fixed, dtype=torch.complex64, device=device)

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

