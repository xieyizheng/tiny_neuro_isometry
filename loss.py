# https://github.com/vsitzmann/neural-isometries/blob/3d47289a6aa16be76ca158fadfd30518601ea977/nn/losses.py
import torch
from core import delta_mask
def multiplicity_loss(eigenvalues):
    # Create similarity matrix G
    G = delta_mask(eigenvalues)
    
    # Create degree matrix D
    D = torch.diag_embed(G.sum(dim=-1))
    
    # Create Laplacian matrix L = D - G
    L = D - G
    
    # Compute normalized Frobenius norm
    norm = torch.sqrt(torch.sum(L**2, dim=(-2,-1)) + 1e-8) / L.shape[-1]
    return torch.mean(norm)

def reconstruction_loss(A, B, tauOmega, Omega):
    B_pred = torch.einsum('...ij,...jk,...lk,...lm->...im', 
                        Omega[0][None,...], 
                        tauOmega,
                        Omega[0][None,...], 
                        Omega[2][None,...,None] * A)
        
    A_pred = torch.einsum('...ij,...jk,...lk,...lm->...im',
                        Omega[0][None,...],
                        tauOmega.transpose(-2,-1), 
                        Omega[0][None,...],
                        Omega[2][None,...,None] * B)
    
    loss_recon = torch.mean(torch.abs(A - A_pred)) + torch.mean(torch.abs(B - B_pred))
    return loss_recon
