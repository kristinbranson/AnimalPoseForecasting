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
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from sandbox.kp2im import (
    BasicDataset, load_data, add_bg_to_img, center_img, uncenter_img, 
    KeypointConditionedUnet, train_kp2im, make_img, load_fly_movie
)
from sandbox.im2kp import device, load_im2kp_datasets, init_unet, train_unet, make_heatmap_from_coco, heatmap2keypoints
# -

# # im2kp

# ### Load data

im_width = 120
im2kp_train_dataset, im2kp_test_dataset = load_im2kp_datasets(
    dataurl="https://research.janelia.org/bransonlab/multifly/multifly_v_1_0.zip", 
    outzipfile="multifly.zip",
    overwrite=False,
    crop_width=im_width,
)

# Visualize a sample from the training set
sample = im2kp_train_dataset[0]
img = sample['image']
hmaps = sample['heatmaps']
kps = sample['keypoints']
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.imshow(img[0], cmap='gray')
plt.plot(kps[None, :, 0], kps[None, :, 1], '.')
plt.subplot(1,2,2)
plt.imshow(img[0], cmap='gray')
plt.imshow(hmaps[19], alpha=0.5, cmap='hot')
plt.show()

# ### Train im2kp model

# Initialize the network
im2kp_net = init_unet((im_width, im_width, 1), im2kp_train_dataset)

# Load existing model
checkpoint_path = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/PoseEstimationNets/UNet20251009T073335/Final_epoch10.pth"
im2kp_net.load_state_dict(torch.load(checkpoint_path))

# +
# OR train a new model

# Make a checkpoint directory
now = datetime.now()
timestamp = now.strftime('%Y%m%dT%H%M%S')
savedir = 'PoseEstimationNets'
if not os.path.exists(savedir):
  os.mkdir(savedir)
checkpointdir = os.path.join(savedir, 'UNet'+timestamp)
os.mkdir(checkpointdir)

# Train the network
im2kp_net, train_dataloader, val_dataloader, epoch_losses, savefile = \
    train_unet(im2kp_train_dataset, im2kp_net, checkpointdir, nepochs=10)
savefile
# -

# ### Test on one sample

sample = im2kp_test_dataset[10]
im2kp_net.eval()
out = im2kp_net.forward(sample['image'][None, :, :, :].to(device))

pred_hmaps = out[0].detach().cpu().numpy()
kpts = heatmap2keypoints(out[0].detach().cpu().numpy())
plt.imshow(sample['image'][0, :, :].detach().cpu().numpy(), cmap='gray')
plt.imshow(pred_hmaps[19], alpha=0.5, cmap='hot')
x, y = kpts.T
plt.plot(x[None, :], y[None, :], '.')
plt.show()

# # kp2im

# ### Load data

data_path = "/groups/branson/bransonlab/flydisco_data"
keypoints, keypoint_imgs, masks, imgs = load_data(data_path, im_radius=im_width//2, n_movies=6)
n_keypoints = keypoints.shape[-2]
im_width = imgs.shape[-1]

inputs = (keypoint_imgs).reshape([-1] + list(keypoint_imgs.shape[-3:])).astype(np.float32)
labels = imgs.reshape([-1, 1, im_width, im_width])

# +
n_train = 20000
use_im2kp = True

if use_im2kp:
    hmaps = np.array([make_heatmap_from_coco(kps, im_width, im2kp_train_dataset) 
                      for kps in keypoints.reshape([-1, n_keypoints, 2])])
    kp2im_train_dataset = BasicDataset(labels[:n_train], inputs[:n_train], hmaps[:n_train])
    kp2im_val_dataset = BasicDataset(labels[n_train:], inputs[n_train:], hmaps[n_train:])
else:
    kp2im_train_dataset = BasicDataset(labels[:n_train], inputs[:n_train])
    kp2im_val_dataset = BasicDataset(labels[n_train:], inputs[n_train:])
    im2kp_net = None

# +
# Visualize a sample from the training set
sample = kp2im_train_dataset[0]

img = sample[0]
kp_img = sample[1]
kps = np.array(np.where(kp_img == 1)).T[:, 1:][:, ::-1]
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.imshow(img[0], cmap='gray')
plt.imshow(kp_img[19], alpha=0.5, cmap='hot')
if use_im2kp:
    plt.subplot(1,2,2)
    plt.imshow(img[0], cmap='gray')    
    hmaps = sample[2]
    plt.imshow(hmaps[19], alpha=0.5, cmap='hot')
plt.show()
# -

# ### Test im2kp on this data

if use_im2kp:
    img = add_bg_to_img(kp2im_train_dataset[60][0])
    out = im2kp_net.forward(torch.from_numpy(img[None, :, :, :]).to(device))
    
    pred_hmaps = out[0].detach().cpu().numpy()
    kpts = heatmap2keypoints(out[0].detach().cpu().numpy())
    plt.imshow(img[0, :, :], cmap='gray')
    plt.imshow(pred_hmaps[19], alpha=0.5, cmap='hot')
    x, y = kpts.T
    plt.plot(x[None, :], y[None, :], '.')
    plt.show()

# ### Train a kp2im model

# Wrap data into loaders
train_dataloader = DataLoader(kp2im_train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(kp2im_val_dataset, batch_size=8, shuffle=False)

# +
# Create a noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_schedule="squaredcos_cap_v2",
    prediction_type='sample',  # use 'epsilon' for noise, 'sample' for image
)

# Initialize network
net = KeypointConditionedUnet(n_keypoints=n_keypoints, img_resolution=im_width).to(device)
losses = None

# +
# save net
# optionally load net
# -

n_epochs = 10
net, losses = train_kp2im(
    train_dataloader, val_dataloader,
    noise_scheduler, net, losses=losses,
    n_epochs=n_epochs, learning_rate=1e-3,
    im2kp_net=im2kp_net, hm_loss_weight=0.1,
)

plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.plot(np.array(losses['train_img_loss']).reshape(n_epochs, len(train_dataloader)).mean(1))
plt.plot(np.array(losses['val_img_loss']).reshape(n_epochs, len(val_dataloader)).mean(1))
plt.legend(['train', 'val'])
plt.title('image loss')
if use_im2kp:
    plt.subplot(1, 2, 2)
    plt.plot(np.array(losses['train_hm_loss']).reshape(n_epochs, len(train_dataloader)).mean(1))
    plt.plot(np.array(losses['val_hm_loss']).reshape(n_epochs, len(val_dataloader)).mean(1))
    plt.title('keypoint loss')
plt.show()

# ### Run kp2im on some validation samples

# +
n_val = 49
x = torch.randn(n_val, 1, im_width, im_width).to(device)
x_gt = torch.from_numpy(labels[n_train:n_train+n_val]).to(device)
y = torch.from_numpy(inputs[n_train:n_train+n_val]).to(device)

# Sampling loop
net = net.eval()
for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

    # Get model pred
    with torch.no_grad():
        residual = net(x, t, y)  # Again, note that we pass in our labels y

    # Update sample with step
    out = noise_scheduler.step(residual, t, x)
    x = out.prev_sample
x = out.pred_original_sample    
# -

# Show the results
plt.figure(figsize=(12, 12))
n_train = 0
for i in range(x.shape[0]):
    # r, c = keypoints.reshape([-1, 21, 2])[n_train + i].T
    _, c, r = np.where(y[i].detach().cpu().numpy() == 1)
    plt.subplot(10, 5, i+1)
    pred_im = uncenter_img(x[i, 0].detach().cpu().numpy())
    gt_im = x_gt[i, 0].detach().cpu().numpy()
    im = add_bg_to_img(np.concatenate([gt_im, pred_im], axis=1))
    plt.imshow(im, cmap='gray')
    # plt.plot(r[None, :], c[None, :], '.r', markersize=2)
    # plt.plot(r[None, :] + im_width, c[None, :], '.r', markersize=2)
    plt.axis('off')
    plt.clim([0, 1])
    plt.title(str(n_train + i), fontsize=10)
plt.show()

# ### Generate a full image of flies in a chamber

# Load fly data
movies = os.listdir(data_path)
movie, apt_data, bg_img = load_fly_movie(os.path.join(data_path, movies[6627+11]))
apt_trk = np.array(apt_data['pTrk'])

# +
frame_idx = np.random.randint(apt_trk.shape[-1])
keypoints = apt_trk[:, :, :, frame_idx]
new_img, pred = make_img(bg_img, keypoints, net, noise_scheduler, width=im_width)

plt.figure()
plt.imshow(new_img, cmap='gray')
plt.show()
# -


