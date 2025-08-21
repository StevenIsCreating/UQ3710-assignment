import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="PyTorch Cantor Set renderer")
parser.add_argument("--width", type=int, default=2048, help="number of samples in [0,1)")
parser.add_argument("--iters", type=int, default=12,   help="Cantor construction iterations (rows)")
parser.add_argument("--outfile", type=str, default="cantor.png", help="output image file")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


W = args.width
K = args.iters

x = torch.linspace(
    0.0, 1.0, steps=W + 1, device=device, dtype=torch.float32, requires_grad=False
)[:-1]  


pow3 = (3.0 ** torch.arange(0, K, device=device, dtype=torch.float32)).unsqueeze(1) 
T = torch.floor(x.unsqueeze(0) * pow3) % 3.0 
removed = T.eq(1.0)                        


ever_removed_up_to_row = removed.float().cumsum(dim=0).gt(0)  
kept = ~ever_removed_up_to_row                                 


img = kept.float()  


fractal_dim = math.log(2.0) / math.log(3.0)
print(f"Cantor set fractal dimension D = log(2)/log(3) = {fractal_dim:.6f}")


plt.figure(figsize=(12, 6))
plt.imshow(img.cpu().numpy(), aspect="auto", interpolation="nearest", origin="upper")
plt.xlabel("x in [0, 1)")
plt.ylabel("iteration (top → bottom)")
plt.title(f"Cantor Set — width={W}, iters={K}, D≈{fractal_dim:.4f}")
plt.tight_layout()
plt.show()

