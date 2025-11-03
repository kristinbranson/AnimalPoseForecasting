# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2
    
import os
import cv2
import glob
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np
import time

from sandbox.kp2im import load_mouse_data, BasicDataset, MouseDataset
# -



data_path = '/groups/branson/home/eyjolfsdottire/data/diffusion_mice/190625_m165836silpb_no_odor_m165837_f0164992'

# +
full_dataset = MouseDataset(data_path)
n_train = int(len(full_dataset) * 0.9)

diffusion_train_dataset = MouseDataset(data_path, end_idx = n_train)
diffusion_val_dataset = MouseDataset(data_path, start_idx = n_train)
# -

len(diffusion_train_dataset), len(diffusion_val_dataset)

t0 = time.time()
diffusion_val_dataset.filter(skip_large=True, skip_mouse_id=0)
dt = time.time() - t0
print(dt)

t0 = time.time()
diffusion_train_dataset.filter(skip_large=True, skip_mouse_id=0)
dt = time.time() - t0
print(dt)

# +
diffusion_train_dataset.set_step(10)
diffusion_train_dataset.set_scalefactor(0.5)

diffusion_val_dataset.set_step(10)
diffusion_val_dataset.set_scalefactor(0.5)
# -

len(diffusion_train_dataset), len(diffusion_val_dataset)

diffusion_train_dataset.set_step(2)
diffusion_val_dataset.set_step(2)

# +
idx = np.random.randint(len(diffusion_train_dataset))
sample = diffusion_train_dataset[idx]

img, kp_img, bg_img, kpts = sample
kps = np.array(np.where(kp_img.detach().cpu().numpy() == 1)).T[:, 1:][:, ::-1]
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.imshow(img[0].detach().cpu().numpy(), cmap='gray')
y, x = np.where(kp_img.sum(0) > 0)
# plt.imshow(kp_img[0].detach().cpu().numpy(), alpha=0.5, cmap='hot')
plt.plot(x[None, :], y[None, :], '.')
plt.subplot(1,2,2)
plt.imshow(bg_img[0].detach().cpu().numpy(), cmap='gray')
plt.show()
# -

len(train_dataloader)

# +
from torch.utils.data import DataLoader

batch_size = 8
train_dataloader = DataLoader(diffusion_train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(diffusion_val_dataset, batch_size=batch_size, shuffle=False)
# -

pred_noise = False

# +
from diffusers import DDPMScheduler

if pred_noise:
    pred_type = 'epsilon'
else:
    pred_type = 'sample'
n_time_steps = 100
noise_scheduler = DDPMScheduler(num_train_timesteps=n_time_steps, beta_schedule="squaredcos_cap_v2", prediction_type=pred_type)

# +
from tqdm import tqdm
import torch
import torch.nn as nn
from apf.diffusion_models import KeypointConditionedUnet


n_epochs = 10

n_items = len(train_dataloader)
n_keypoints = kp_img.shape[0]
resolution = kp_img.shape[-1]
net = KeypointConditionedUnet(n_keypoints=n_keypoints, img_resolution=resolution, include_bg_img=True).to(device)
# -

use_hm_loss = False

losses = []
val_losses = []
best_val_loss = 1000
best_val_net = None

# +
import copy

n_epochs = 5

loss_fn = nn.MSELoss()

opt = torch.optim.Adam(net.parameters(), lr=1e-3)

n_val_items = len(val_dataloader)
for epoch in range(n_epochs):
    net.train()
    for sample in tqdm(train_dataloader):

        x = sample[0]
        y = sample[1]
        bg = sample[2]
        
        # Get some data and prepare the corrupted version
        x = x.to(device) * 2 - 1  # Data on the GPU (mapped to (-1, 1))
        y = y.to(device)
        bg = bg.to(device) * 2 - 1
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, n_time_steps, (x.shape[0],)).long().to(device)
        # timesteps[:] = timesteps[0]
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
        pred = net(noisy_x, timesteps, y, bg)  # Note that we pass in the labels y

        # Calculate the loss
        if pred_noise:
            loss = loss_fn(pred, noise)  # How close is the output to the noise
        else:
            loss = loss_fn(pred, x)  # How close is the output to the noise

        if use_hm_loss: # and timesteps.min() > 800:
            # Apply keypoint classifier to the output image
            if pred_noise:
                pred_im = noisy_x - pred
            else:
                pred_im = pred
            # pred_hm = im2kp_net.forward((pred_im + 1) / 2)
            pred_hm = im2kp_net.forward(0.8 * (1 - (pred_im + 1) / 2))
            
            # Compute loss on the predicted keypoint heatmap
            gt_hm = sample[2]
            hm_loss = hm_loss_fn(pred_hm, gt_hm)

            loss = loss + hm_loss * 0.1 #* timesteps.min() / 999
        
        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()        
        
        # Store the loss for later
        losses.append(loss.item())

    net.eval()
    for sample in tqdm(val_dataloader):
        x = sample[0]
        y = sample[1]
        bg = sample[2]
        
        # Get some data and prepare the corrupted version
        x = x.to(device) * 2 - 1  # Data on the GPU (mapped to (-1, 1))
        y = y.to(device)
        bg = bg.to(device) * 2 - 1
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, n_time_steps, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
        pred = net(noisy_x, timesteps, y, bg)  # Note that we pass in the labels y

        # Calculate the loss
        if pred_noise:
            loss = loss_fn(pred, noise)  # How close is the output to the noise
        else:
            loss = loss_fn(pred, x)  # How close is the output to the noise

        if use_hm_loss: # and timesteps.min() > 800:
            # Apply keypoint classifier to the output image
            if pred_noise:
                pred_im = noisy_x - pred
            else:
                pred_im = pred
            # pred_hm = im2kp_net.forward((pred_im + 1) / 2)
            pred_hm = im2kp_net.forward(0.8 * (1 - (pred_im + 1) / 2))
            
            # Compute loss on the predicted keypoint heatmap
            gt_hm = sample[2]
            hm_loss = hm_loss_fn(pred_hm, gt_hm)

            loss = loss + hm_loss * 0.1 #* timesteps.min() / 999
        
        # Store the loss for later
        val_losses.append(loss.item())        
    
    # Print out the average of the last 100 loss values to get an idea of progress:    
    avg_loss = sum(losses[-n_items:]) / n_items
    avg_val_loss = sum(val_losses[-n_val_items:]) / n_val_items

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_net = copy.deepcopy(net)
    
    print(f"Finished epoch {epoch}. Mean train loss: {avg_loss:05f}, mean val loss: {avg_val_loss:05f}")

# View the loss curve
plt.plot(losses)
plt.plot(val_losses)

# +
# torch.save(net.state_dict(), 'mouse_net_full_res_bg_input_subset_1_epoch.pth')
# -

plt.plot(losses)
plt.plot(val_losses)

n_items = len(train_dataloader)
n_val_items = len(val_dataloader)

# +
n_items
n_val_items

tmp = [np.mean(losses[i*n_items:(i+1)*n_items]) for i in range(len(losses) // n_items)]
tmp_val = [np.mean(val_losses[i*n_val_items:(i+1)*n_val_items]) for i in range(len(val_losses) // n_val_items)]
plt.plot(tmp)
plt.plot(tmp_val)
plt.legend(['training', 'validation'])
plt.xlabel('epoch')
plt.title('mean combined loss')
# -

latest_net = net
net = best_val_net

net = net.eval()

i = 0 #n_train
x_gt = []
y = []
bg = []
n_val = 49
while len(x_gt) < n_val:
    img, kpt_img, bg_img, _ = diffusion_val_dataset[i]
    # if img.shape[-1] == 296:
    x_gt.append(img[None, :, :, :])
    y.append(kpt_img[None, :, :, :])
    bg.append(bg_img[None, :, :, :])
    i += 1

592 / 2 / 2 / 2

x_gt = torch.concat(x_gt, axis=0).to(device)
y = torch.concat(y, axis=0).to(device)
bg = torch.concat(bg, axis=0).to(device)
x.shape, x_gt.shape, y.shape, bg.shape

# +
n_val = 49
width = x_gt.shape[-1]
x = torch.randn(n_val, 1, width, width).to(device)
# x_gt = t_labels[n_train:n_train+n_val].to(device)
# y = t_inputs[n_train:n_train+n_val].to(device)


intermediates = []
# Sampling loop
for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

    # Get model pred
    with torch.no_grad():
        residual = net(x, t, y, bg)  # Again, note that we pass in our labels y

    # Update sample with step
    out = noise_scheduler.step(residual, t, x)
    x = out.prev_sample
    intermediates.append(out.pred_original_sample)
x = out.pred_original_sample    
# -

n_items * 8

# Show the results
plt.figure(figsize=(15, 9))
n_train = 0
for i in range(x.shape[0]):
    # r, c = keypoints.reshape([-1, 21, 2])[n_train + i].T
    _, c, r = np.where(y[i].detach().cpu().numpy() == 1)
    plt.subplot(7, 7, i+1)
    pred_im = x[i, 0].detach().cpu().numpy() / 2 + 0.5
    gt_im = x_gt[i, 0].detach().cpu().numpy()
    
    im = np.concatenate([gt_im, pred_im], axis=1)
    plt.imshow(im, cmap='gray')
    # plt.plot(r[None, :], c[None, :], '.r', markersize=2)
    # plt.plot(r[None, :] + width, c[None, :], '.r', markersize=2)
    plt.axis('off')
    plt.clim([0, 1])
    plt.title(str(n_train + i), fontsize=10)
plt.show()

# +

# i = np.random.randint(n_val)
for i in range(49):

    plt.figure(figsize=(15, 7))
    
    _, c, r = np.where(y[i].detach().cpu().numpy() == 1)
    
    pred_im = x[i, 0].detach().cpu().numpy() / 2 + 0.5
    gt_im = x_gt[i, 0].detach().cpu().numpy()
    
    im = np.concatenate([gt_im, pred_im], axis=1)
    plt.imshow(im, cmap='gray')
    # plt.plot(r[None, :], c[None, :], '.r', markersize=2)
    # plt.plot(r[None, :] + width, c[None, :], '.r', markersize=2)
    plt.axis('off')
    plt.clim([0, 1])
    plt.title(str(n_train + i), fontsize=10)
    
    plt.show()
# -

len(diffusion_val_dataset)

diffusion_val_dataset.set_step(2) # 80

4076/49

i = 0 #n_train
x_gt = []
y = []
bg = []
n_val = 49
while len(x_gt) < n_val:
    img, kpt_img, bg_img, _ = diffusion_val_dataset[i]
    # if img.shape[-1] == 296:
    x_gt.append(img[None, :, :, :])
    y.append(kpt_img[None, :, :, :])
    bg.append(bg_img[None, :, :, :])
    i += 1

plt.figure(figsize=(15, 15))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.imshow(x_gt[i][0, 0], cmap='gray')
    plt.title(i)
    plt.axis('off')
plt.show()


