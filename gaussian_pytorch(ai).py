# gaussian_pytorch.py
import numpy as np
import matplotlib.pyplot as plt
import torch

# ---- Device ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ---- Grid via NumPy, then to torch ----
X, Y = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]
x = torch.tensor(X, dtype=torch.float32, device=device)
y = torch.tensor(Y, dtype=torch.float32, device=device)

# ---- Gaussian in torch ----
z = torch.exp(- (x**2 + y**2) / 2.0)

# ---- Plot (move to CPU for matplotlib) ----
plt.imshow(z.detach().cpu().numpy(), origin='lower')
plt.title('2D Gaussian (PyTorch)')
plt.axis('off')
plt.tight_layout()

# show
plt.show()
