import os
import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from dataset import ShiftedImageDataset  # Import our custom dataset
from core import OperatorIso
from loss import multiplicity_loss, reconstruction_loss
from utils import log_things
# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --- Init W&B ---
wandb.init(project="tiny_niso")

# --- Transforms and Dataset ---
transform = transforms.Compose([
    transforms.Resize((16, 16)),
    transforms.ToTensor(),
])

# Create base dataset then wrap it in ShiftedImageDataset
base_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
dataset = ShiftedImageDataset(base_dataset, image_size=(16, 16))
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, 
                       collate_fn=ShiftedImageDataset.collate_fn)

# --- Model ---
model = OperatorIso(op_dim=256, spatial_dim=256, clustered_init=True, device=device)

optimizer = optim.AdamW(model.parameters(), lr=5e-4)

# --- Training Loop ---
for epoch in range(10000):
    for step, (images, shifted_images, labels) in enumerate(tqdm(dataloader)):
        # Stack of images with the same shift
        images = images.to(device) # [S, 3, 16, 16]
        shifted_images = shifted_images.to(device) # [S, 3, 16, 16]
        
        A = images.permute(2,3,0,1) # [16, 16, S, 3]
        B = shifted_images.permute(2,3,0,1) # [16, 16, S, 3]

        A = A.reshape(1, 256, -1) # [1, 256, S*3]
        B = B.reshape(1, 256, -1) # [1, 256, S*3]

        # model input A, B : (batch × spatial_dim × channels) tensors
        tauOmega, Omega = model(A, B)

        loss_recon = reconstruction_loss(A, B, tauOmega, Omega)
        loss_mult = multiplicity_loss(Omega[1])  # Omega[1] contains eigenvalues
        
        loss_total = loss_recon + 0.1 * loss_mult

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if step % 10 == 0:
            wandb.log({"loss_total": loss_total.item()})
            wandb.log({"loss_recon": loss_recon.item()})
            wandb.log({"loss_mult": loss_mult.item()})

        if step % 200 == 0:
            log_things(images, shifted_images, tauOmega, model._get_Omega())

        if epoch % 20 == 0 and step == 0:
            os.makedirs(f"wandb/latest-run/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"wandb/latest-run/checkpoints/model_{epoch}_{step}.pth")
