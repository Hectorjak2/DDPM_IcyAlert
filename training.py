import torch
from torch.optim import Adam

from utils import CARRA2, visualize_sample
from models.ddpm import DDPM
from models.unet import Unet

device = "mps"
ddpm = DDPM(timesteps=1000, device=device)

timesteps = 100
batch_size = 4

train_ds = CARRA2("siconc", device, time_slice=slice("2012-01", "2013-12"), TEST=True, batch_dim=False)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

def train(): 

    #define the model
    model = Unet(image_size=128)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.1f}M")

    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    epochs = 10

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)

            optimizer.zero_grad()

            # sample t from U(0,T)
            t = torch.randint(0, timesteps, (batch.shape[0],), device=device).long()
            
            loss = ddpm.compute_loss(model, batch, t)

            print(f"Epoch: {epoch}, step: {step} -- Loss: {loss.item():.3f}")

            loss.backward()
            optimizer.step() 

train()