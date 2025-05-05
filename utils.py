import wandb
import torch
import numpy as np
import matplotlib
import matplotlib.cm as cm 

def scalar_to_rgb(array, cmap=cm.rainbow):
    norm = matplotlib.colors.Normalize(vmin=array.min(), vmax=array.max())
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    img = (mapper.to_rgba(array)[:, :, :3] * 255).astype(np.uint8)
    return img


def log_things(images, shifted_images, tauOmega, Omega):

    Phi_sorted, Lambda_sorted, M = Omega
    
    Learned_Operator = M.diag() @ Phi_sorted @ Lambda_sorted.diag() @ Phi_sorted.T
    Learned_Operator = scalar_to_rgb(Learned_Operator.detach().cpu().numpy())

    Learned_Eigenvectors = Phi_sorted.reshape(16, 16, 16, 16).permute(2, 0, 3, 1).reshape(256, 256)
    Learned_Eigenvectors = scalar_to_rgb(Learned_Eigenvectors.detach().cpu().numpy(), cmap=cm.Greys)
    Learned_Eigenvectors = Learned_Eigenvectors.repeat(4, axis=0).repeat(4, axis=1)

    tauOmega = scalar_to_rgb(tauOmega[0].detach().cpu().numpy())

    grid_images = torch.cat([
        torch.cat([images[0], images[1], images[2]], dim=2),  # First row: original images
        torch.cat([shifted_images[0], shifted_images[1], shifted_images[2]], dim=2)  # Second row: shifted images
    ], dim=1).detach().cpu().permute(1, 2, 0).numpy()

    wandb.log({
        "grid_images": wandb.Image(grid_images, caption="Top: Original, Bottom: Shifted"),
        "operator_matrix": wandb.Image(Learned_Operator, caption="Learned Operator"),
        "Learned_Eigenvectors": wandb.Image(Learned_Eigenvectors, caption="Learned Eigenvectors"),
        "tauOmega": wandb.Image(tauOmega, caption="tauOmega"),
    })