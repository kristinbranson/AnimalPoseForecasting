import numpy as np
from tqdm import tqdm
import os
import scipy
import mat73
from scipy.io import loadmat
import copy
from scipy.spatial import ConvexHull
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation
from skimage import morphology
from skimage.measure import label
import cv2
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from diffusers import UNet2DModel

from sandbox.im2kp import heatmap2keypoints

# Add the flymovie code to path
sys.path.append("/groups/branson/home/eyjolfsdottire/code/")
from flymovie.movies import Movie


""" Model """


class KeypointConditionedUnet(nn.Module):
    def __init__(self, n_keypoints, img_resolution, layers_per_block=2, include_bg_img=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=n_keypoints, out_channels=n_keypoints, kernel_size=15, padding=7)
        self.conv2 = nn.Conv2d(in_channels=n_keypoints, out_channels=n_keypoints, kernel_size=15, padding=7)
        self.conv3 = nn.Conv2d(in_channels=n_keypoints, out_channels=n_keypoints, kernel_size=15, padding=7)

        n_imgs = 1 + include_bg_img

        self.model = UNet2DModel(
            sample_size=img_resolution,  # the target image resolution
            in_channels=n_imgs + n_keypoints,  # Additional input channels for kpt cond.
            out_channels=1,  # the number of output channels
            layers_per_block=layers_per_block,  # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
        )

    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, kptimg, bg_img=None):
        # Alternative way of encoding this is to have each channel embed the x, y coordinate of the
        #    keypoint (at each pixel), so that all the pixels have access to that value

        # First convolve the keypoint input image a few times
        for conv in [self.conv1, self.conv2, self.conv3]:
            kptimg = conv(kptimg)
        kptimg = F.sigmoid(kptimg)

        # Feed this to the UNet alongside the timestep and return the prediction
        if bg_img is None:
            return self.model(torch.cat((x, kptimg), 1), t).sample
        return self.model(torch.cat((x, bg_img, kptimg), 1), t).sample


""" Data """


def get_rotmat(kpt, mid_idx=5, vec_ids=(0, 6)):
    # center around kpt 5
    if type(mid_idx) is list or type(mid_idx) is np.ndarray:
        center = kpt[mid_idx].mean(0)
    else:
        center = kpt[mid_idx]
    # rotate along kpts 0-6
    pt1 = vec_ids[0]
    pt2 = vec_ids[1]
    vec = np.array(kpt[pt1] - kpt[pt2])
    vec = vec / np.linalg.norm(vec)
    angle = np.rad2deg(np.arctan2(vec[0], -vec[1]))
    return cv2.getRotationMatrix2D(center, angle, 1)


def rotate_image(img, rotmat):
    height, width = img.shape
    im_type = img.dtype
    return cv2.warpAffine(img.astype(np.float32), rotmat, (width, height)).astype(im_type)


def rotate_keypoints(kpt, rotmat):
    return np.dot(kpt, rotmat[:2, :2].T) + rotmat[:2, 2]


def get_cropbox(kpt, buf=49.5, mid_idx=5):
    if type(mid_idx) is list or type(mid_idx) is np.ndarray:
        midx, midy = kpt[mid_idx].mean(0)
    else:
        midx, midy = kpt[mid_idx]
    minx, maxx, miny, maxy = np.round(np.array([midx - buf, midx + buf + 1, midy - buf, midy + buf + 1])).astype(int)
    return [minx, maxx, miny, maxy]


def crop_image(img, cropbox, fill_value=0.5):
    minx, maxx, miny, maxy = cropbox

    out_of_bounds = (minx < 0 or miny < 0 or maxx > img.shape[1] or maxy > img.shape[0])
    if not out_of_bounds:
        return img[miny:maxy, minx:maxx], out_of_bounds

    width = maxx - minx
    x0 = y0 = 0
    x1 = y1 = width
    if minx < 0:
        x0 -= minx
        minx = 0
    if miny < 0:
        y0 -= miny
        miny = 0
    if maxx > img.shape[1]:
        diff = maxx - img.shape[1]
        x1 -= diff
        maxx = img.shape[1]
    if maxy > img.shape[0]:
        diff = maxy - img.shape[0]
        y1 -= diff
        maxy = img.shape[1]

    # print('Handling out of bounds')
    cropimg = np.ones((width, width)) * fill_value
    try:
        cropimg[y0:y1, x0:x1] = img[miny:maxy, minx:maxx]
    except:
        print(f"width={width}, [{x0}, {x1}, {y0}, {y1}], [{minx}, {maxx}, {miny}, {maxy}], img shape={img.shape}")
        cropimg[y0:y1, x0:x1] = img[miny:maxy, minx:maxx]

    return cropimg, out_of_bounds


def crop_keypoints(kpt, cropbox):
    minx, maxx, miny, maxy = cropbox
    return kpt - np.array([minx, miny])[None, :]


def get_hull_mask(points, width):
    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]
    hull_mask = np.zeros((width, width), dtype=np.uint8)
    hull_vertices_int = hull_vertices.astype(np.int32)
    polygons = [hull_vertices_int]
    cv2.fillPoly(hull_mask, polygons, 255)
    for _ in range(2):
        hull_mask = morphology.binary_dilation(hull_mask)
    return hull_mask > 0


class BasicDataset(Dataset):
    def __init__(self, inputs, labels, heatmaps=None):
        self.inputs = inputs
        self.labels = labels
        self.heatmaps = heatmaps

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.heatmaps is None:
            return self.inputs[idx], self.labels[idx]
        return self.inputs[idx], self.labels[idx], self.heatmaps[idx]


class MouseDataset(Dataset):
    def __init__(self, data_path, start_idx=0, end_idx=None):
        self.data_path = data_path
        img_paths = glob.glob(os.path.join(data_path, '*_img.npz'))
        self.count = len(img_paths)
        if end_idx is None:
            end_idx = self.count - 1
        self.start_idx = start_idx
        self.count = end_idx - self.start_idx + 1
        self.valid_ids = np.arange(self.count)
        self.step = 1
        self.valid_keypoints = None
        self.scalefactor = 1.0

    def filter(self, skip_large=False, skip_mouse_id=None):
        sizes = np.zeros(self.count, int)
        has_mouse = np.zeros((2, self.count), bool)
        for idx in range(self.count):
            use_idx = self.start_idx + idx
            sizes[idx] = np.load(os.path.join(self.data_path, f'{use_idx}_img.npz'))['arr_0'].shape[0]
            kpts = np.load(os.path.join(self.data_path, f'{use_idx}_kpt.npy'))
            has_mouse[0, idx] = ~np.isnan(kpts[0, 0])
            has_mouse[1, idx] = ~np.isnan(kpts[-1, 0])
        valid = np.ones(self.count, bool)
        if skip_large:
            valid = valid * (sizes == sizes.min())
        if skip_mouse_id is not None:
            valid = valid * (~has_mouse[skip_mouse_id])
            use_mouse_id = 1 - skip_mouse_id
            self.valid_keypoints = np.arange(4) + 4 * use_mouse_id
        self.valid_ids = np.where(valid)[0]

    def set_step(self, step):
        self.step = step

    def set_scalefactor(self, scalefactor):
        self.scalefactor = scalefactor

    def __len__(self):
        # return self.count
        return len(self.valid_ids[::self.step])

    def __getitem__(self, idx):
        # use_idx = self.start_idx + idx
        use_idx = self.start_idx + self.valid_ids[::self.step][idx]
        img = np.load(os.path.join(self.data_path, f'{use_idx}_img.npz'))['arr_0'].astype(np.float32) / 255
        bg_img = np.load(os.path.join(self.data_path, f'{use_idx}_bg.npz'))['arr_0'].astype(np.float32) / 255
        kpts = np.load(os.path.join(self.data_path, f'{use_idx}_kpt.npy'))
        if self.valid_keypoints is not None:
            kpts = kpts[self.valid_keypoints]

        if self.scalefactor != 1.0:
            img = cv2.resize(img, dsize=None, fx=self.scalefactor, fy=self.scalefactor, interpolation=cv2.INTER_AREA)
            bg_img = cv2.resize(bg_img, dsize=None, fx=self.scalefactor, fy=self.scalefactor, interpolation=cv2.INTER_AREA)
            kpts = kpts * self.scalefactor

        n_kpts = kpts.shape[0]
        height, width = img.shape
        kpt_img = np.zeros((n_kpts, height, width), np.float32)
        x, y = np.round(kpts).astype(np.long).T
        valid = ~np.isnan(x)
        x = x[valid].clip(0, width - 1)
        y = y[valid].clip(0, height - 1)
        kpt_img[np.arange(n_kpts), y, x] = 1

        # return img, kpt_img, bg_img, kpts
        return torch.from_numpy(img[None, :, :]), torch.from_numpy(kpt_img), torch.from_numpy(bg_img[None, :, :]), torch.from_numpy(kpts)


def get_clean_img(img, bg_img, arena_mask, kpts, fly_id):
    bg_subtr = bg_img - img
    bg_subtr = bg_subtr / bg_img
    bg_subtr[arena_mask] = 0

    body = bg_subtr > .65
    fg = bg_subtr > .08

    labeled_image = label(body)
    # x, y = np.round(kpts[fly_id, 5, :]).astype(int)
    # keep_label = labeled_image[y, x]
    # mask_out = np.logical_and(labeled_image > 0, labeled_image != keep_label)

    label_per_fly = np.zeros(len(kpts))
    for fly_i in range(len(kpts)):
        x, y = np.round(kpts[fly_i, 5, :]).astype(int)
        label_per_fly[fly_i] = labeled_image[y, x]
    mask_out = labeled_image == -1
    for other_fly_i in range(len(kpts)):
        if other_fly_i == fly_id:
            continue
        if label_per_fly[other_fly_i] == label_per_fly[fly_id]:
            continue
        mask_out = np.logical_or(mask_out, labeled_image == label_per_fly[other_fly_i])

    mask_out = binary_dilation(mask_out, iterations=2)
    mask_out = np.logical_and(mask_out, fg)

    fg[mask_out] = 0
    labeled_image_fg = label(fg)
    x, y = np.round(kpts[fly_id, 5, :]).astype(int)
    keep_label = labeled_image_fg[y, x]
    mask_out2 = np.logical_and(labeled_image_fg > 0, labeled_image_fg != keep_label)
    fg[mask_out2] = 0

    mask_out = np.logical_or(mask_out, mask_out2)
    bg_subtr[mask_out] = 0
    return bg_subtr, labeled_image_fg == keep_label


def load_fly_movie(movie_path):
    bg_path = os.path.join(movie_path, 'movie-bg.mat')
    bg_img = scipy.io.loadmat(bg_path)['bg'][0][0][0]

    apt_path = os.path.join(movie_path, 'apt.trk')
    apt_data = mat73.loadmat(apt_path)

    ufmf_file_name = os.path.join(movie_path, 'movie.ufmf')
    movie = Movie(ufmf_file_name)
    return movie, apt_data, bg_img


def load_data(data_path, start_idx=6627, n_movies=10, im_radius=96):
    buf = im_radius - 0.5
    width = int(buf * 2 + 1)

    folders = os.listdir(data_path)

    movie_data = []
    for i in range(n_movies):
        idx = start_idx + i
        movie_data.append(load_fly_movie(os.path.join(data_path, folders[idx])))

    all_kpts = []
    all_kpt_imgs = []
    all_imgs = []
    all_masks = []

    for movie_id, (movie, apt_data, bg_img) in enumerate(movie_data):
        apt_trk = np.array(apt_data['pTrk'])
        n_flies, n_kpts, _, n_frames = apt_trk.shape
        frame_ids = np.arange(0, n_frames, 100)
        n_use_frames = len(frame_ids)
        total_flies = len(frame_ids) * n_flies

        keypoints = np.zeros((n_flies, n_use_frames, n_kpts, 2), np.float32)
        keypoint_imgs = np.zeros((n_flies, n_use_frames, n_kpts, width, width), np.long)
        imgs = np.zeros((n_flies, n_use_frames, width, width), np.float32)
        masks = np.zeros((n_flies, n_use_frames, width, width), np.long)

        for frame_i, apt_frame_id in enumerate(frame_ids):
            frame_id = apt_data['pTrkFrm'][apt_frame_id] - 1
            frame, stamp = movie.get_frame(frame_id)
            img = frame.astype(np.float64) / 255

            # fg = bg_img - img > 0.05

            arena_mask = bg_img < 0.7
            # arena_mask = binary_fill_holes(arena_mask) # NOTE: This will fill the big hole if fully in the image
            I, J = np.where(bg_img > -1)
            radius = bg_img.shape[0] / 2
            inds = (I - radius) ** 2 + (J - radius) ** 2 > (radius * 1.1) ** 2
            arena_mask[I[inds], J[inds]] = 1

            kpts = apt_trk[..., apt_frame_id]
            for fly_i, kpt in enumerate(kpts):

                fly_img, fly_fg = get_clean_img(img, bg_img, arena_mask, kpts, fly_i)

                rotmat = get_rotmat(kpt)
                rot_img = rotate_image(fly_img, rotmat)
                rot_mask = rotate_image(fly_fg, rotmat)
                rot_kpt = rotate_keypoints(kpt, rotmat)

                cropbox = get_cropbox(rot_kpt, buf)
                crop_img, out_of_bounds = crop_image(rot_img, cropbox, fill_value=0)
                crop_mask, _ = crop_image(rot_mask, cropbox, fill_value=0)
                crop_mask = crop_mask > 0.5
                crop_kpt = crop_keypoints(rot_kpt, cropbox)

                keypoints[fly_i, frame_i] = crop_kpt
                x, y = np.round(crop_kpt).astype(np.long).T.clip(0, width - 1)
                keypoint_imgs[fly_i, frame_i, np.arange(n_kpts), y, x] = 1

                # bg_crop, _ = crop_image(rotate_image(bg_img, rotmat), cropbox)
                # labeled_image = label(crop_mask)
                # keep_label = labeled_image[im_radius, im_radius]
                # mask_out = np.logical_and(labeled_image > 0, labeled_image != keep_label)
                # # hull_mask = get_hull_mask(crop_kpt, crop_img.shape[-1])
                # # mask_out = ~hull_mask
                # crop_img[mask_out] = bg_crop[mask_out]
                # crop_mask[mask_out] = 0
                # # crop_mask = labeled_image == keep_label

                imgs[fly_i, frame_i] = crop_img
                masks[fly_i, frame_i] = crop_mask

        all_kpts.append(keypoints)
        all_kpt_imgs.append(keypoint_imgs)
        all_imgs.append(imgs)
        all_masks.append(masks)

    keypoints = np.concatenate(all_kpts, axis=0)
    keypoint_imgs = np.concatenate(all_kpt_imgs, axis=0)
    imgs = np.concatenate(all_imgs, axis=0)
    masks = np.concatenate(all_masks)

    return keypoints, keypoint_imgs, masks, imgs


def load_mouse_data(save_dir, movie_path, main_mouse_id=1, max_frames=None, frame_step=1, im_radius=200, scalefactor=None):
    if scalefactor is not None:
        im_radius *= scalefactor

    buf = im_radius - 0.5
    width = int(buf * 2 + 1)

    moviename = os.path.split(os.path.split(movie_path)[0])[-1]
    save_path = os.path.join(save_dir, moviename)
    os.mkdir(save_path)
    apt_path = movie_path.replace('.mjpg', '.trk')
    apt_data = loadmat(apt_path)
    apt_trk = np.array([trk for trk in apt_data['pTrk'][0]])
    n_mice, n_kpts, _, n_frames = apt_trk.shape

    if max_frames is None:
        max_frames = n_frames

    movie = Movie(movie_path)
    frame, _ = movie.get_frame(0)

    if scalefactor is not None:
        frame = cv2.resize(frame, dsize=None, fx=scalefactor, fy=scalefactor, interpolation=cv2.INTER_AREA)
        apt_trk = apt_trk * scalefactor

    # Compute background image from first 1000 frames
    # TODO: Need a better way to compute bg image without mice, since they can stay put for 1000 frames
    bg_chunk = 1000
    chunk_imgs = np.zeros((bg_chunk, frame.shape[0], frame.shape[1]), dtype=np.uint8)
    for i in range(bg_chunk):
        frame, _ = movie.get_frame(i)
        if scalefactor is not None:
            frame = cv2.resize(frame, dsize=None, fx=scalefactor, fy=scalefactor, interpolation=cv2.INTER_AREA)
        chunk_imgs[i] = frame
    bg_img = np.nanmedian(chunk_imgs, axis=0)  # .astype(np.float32) / 255

    # keypoints = np.zeros((n_mice, max_frames, n_kpts, 2), np.float32)
    # keypoint_imgs = np.zeros((n_mice, max_frames, n_kpts, width, width), np.long)
    # imgs = np.zeros((n_mice, max_frames, width, width), np.float32)

    # bg_imgs = []
    # imgs = []
    # kpt_imgs = []
    # keypoints = []
    count = 0

    for frame_i in tqdm(range(0, max_frames, frame_step)):
        # mod_frame_i = np.mod(frame_i, bg_chunk)
        # # Front load next bg_chunk of images and compute background image
        # if mod_frame_i == 0:
        #     # Load a chunk of images
        #     max_frame_i = min(frame_i + bg_chunk, n_frames)
        #     for chunk_frame_i in range(frame_i, max_frame_i):
        #         frame, _ = movie.get_frame(chunk_frame_i)
        #         if scalefactor is not None:
        #             frame = cv2.resize(frame, dsize=None, fx=scalefactor, fy=scalefactor, interpolation=cv2.INTER_AREA)
        #         chunk_imgs[chunk_frame_i - frame_i] = frame
        #     # Compute background image
        #     # TODO: Need a better way to compute bg image without mice, since they can stay put for 1000 frames
        #     if frame_i == 0:
        #         bg_img = np.nanmedian(chunk_imgs, axis=0) #.astype(np.float32) / 255
        #
        # img = chunk_imgs[mod_frame_i] #.astype(np.float32) / 255

        img, _ = movie.get_frame(frame_i)
        if scalefactor is not None:
            img = cv2.resize(img, dsize=None, fx=scalefactor, fy=scalefactor, interpolation=cv2.INTER_AREA)

        # Subtract background and threshold to find foregrounds
        mask = np.abs(img - bg_img) > 20
        mask = binary_fill_holes(mask)
        mask_label = label(mask)

        # Determine whether mice share a foreground, in which case we use a bigger window and include both mice
        pt1 = np.nanmean(apt_trk[0, :, :, frame_i], 0)
        pt2 = np.nanmean(apt_trk[1, :, :, frame_i], 0)
        if any(np.isnan(pt1)) or any(np.isnan(pt2)):
            print(f"Frame {frame_i} has all Nan keypoints, skipping...")
            continue
        x1, y1 = pt1.astype(int)
        x2, y2 = pt2.astype(int)
        label1 = mask_label[y1, x1]
        label2 = mask_label[y2, x2]
        label_ids = [label1, label2]
        n_kpts = apt_trk.shape[1]
        touching = label1 == label2

        mid_idx = n_kpts * main_mouse_id + np.arange(2)
        vec_ids = n_kpts * main_mouse_id + np.arange(2)
        kpts = apt_trk[:, :, :, frame_i].reshape(-1, 2)

        rotmat = get_rotmat(kpts, mid_idx=mid_idx, vec_ids=vec_ids)
        rot_img = rotate_image(img, rotmat)
        rot_bg_img = rotate_image(bg_img, rotmat)
        rot_kpts = rotate_keypoints(kpts, rotmat)

        if touching:
            cropbox = get_cropbox(rot_kpts, (im_radius * 1.5)-0.5, mid_idx=mid_idx)
            try:
                crop_img, out_of_bounds = crop_image(rot_img, cropbox, fill_value=0)
            except:
                print(rot_kpts)
                print(im_radius)
                print(mid_idx)
                print(cropbox)
                tmp = 1/0
            crop_bg_img, _ = crop_image(rot_bg_img, cropbox, fill_value=0)
            crop_kpts = crop_keypoints(rot_kpts, cropbox)

            # kpt_img = np.zeros((n_kpts * 2, crop_img.shape[0], crop_img.shape[1]), np.long)
            # x, y = np.round(crop_kpts).astype(np.long).T.clip(0, width - 1)
            # kpt_img[np.arange(2 * n_kpts), y, x] = 1

            np.savez_compressed(os.path.join(save_path, f'{count}_img.npz'), crop_img)
            np.savez_compressed(os.path.join(save_path, f'{count}_bg.npz'), crop_bg_img)
            np.save(os.path.join(save_path, f'{count}_kpt.npy'), crop_kpts)
            count += 1

            # imgs.append(crop_img)
            # kpt_imgs.append(kpt_img)
            # bg_imgs.append(crop_bg_img)
            # keypoints.append(crop_kpts)
        else:
            rot_mask_label = rotate_image(mask_label, rotmat)
            for mouse_id, label_id in enumerate(label_ids):
                other_id = np.mod(mouse_id + 1, 2)
                curr_mouse_kpt_ids = np.arange(n_kpts) + n_kpts * mouse_id
                mid_idx = n_kpts * mouse_id + np.arange(2)
                if not mouse_id == main_mouse_id:
                    use_img = img
                    use_bg_img = bg_img
                    use_kpts = kpts
                    use_mask_label = mask_label
                else:
                    use_img = rot_img
                    use_bg_img = rot_bg_img
                    use_kpts = rot_kpts
                    use_mask_label = rot_mask_label

                # Make a copy of the image and mask out the other mouse
                other_pixels = use_mask_label == label_ids[other_id]
                masked_img = copy.deepcopy(use_img)
                masked_img[other_pixels] = use_bg_img[other_pixels]
                masked_kpts = np.ones((n_kpts * 2, 2)) * np.nan
                masked_kpts[curr_mouse_kpt_ids] = use_kpts[curr_mouse_kpt_ids]

                cropbox = get_cropbox(masked_kpts, buf, mid_idx=mid_idx)
                crop_img, out_of_bounds = crop_image(masked_img, cropbox, fill_value=0)
                crop_bg_img, _ = crop_image(use_bg_img, cropbox, fill_value=0)
                crop_kpts = crop_keypoints(masked_kpts, cropbox)

                # plt.imshow(rot_img, cmap='gray')
                # plt.title(cropbox)
                # plt.show()
                # plt.imshow(masked_rot_img, cmap='gray')
                # plt.show()

                # kpt_img = np.zeros((n_kpts * 2, crop_img.shape[0], crop_img.shape[1]), np.long)
                # x, y = np.round(crop_kpts).astype(np.long).T.clip(0, width - 1)
                # y = y[curr_mouse_kpt_ids]
                # x = x[curr_mouse_kpt_ids]
                # kpt_img[np.arange(n_kpts), y, x] = 1

                np.savez_compressed(os.path.join(save_path, f'{count}_img.npz'), crop_img)
                np.savez_compressed(os.path.join(save_path, f'{count}_bg.npz'), crop_bg_img)
                np.save(os.path.join(save_path, f'{count}_kpt.npy'), crop_kpts)
                count += 1

                # imgs.append(crop_img)
                # kpt_imgs.append(kpt_img)
                # bg_imgs.append(crop_bg_img)
                # keypoints.append(crop_kpts)

    return save_path, count #keypoints, kpt_imgs, imgs, bg_imgs


""" Train """


def train_kp2im(
        train_dataloader, val_dataloader,
        noise_scheduler, net, losses=None,
        n_epochs=10, learning_rate=1e-3,
        im2kp_net=None, hm_loss_weight=0.1,
):
    device = next(net.parameters()).device

    # Define loss functions
    img_loss_fn = nn.MSELoss()
    hm_loss_fn = nn.BCEWithLogitsLoss()
    use_hm_loss = im2kp_net is not None
    if use_hm_loss:
        im2kp_net.eval()

    # Initialize losses if not given
    if losses is None:
        losses = {'train_img_loss': [], 'train_hm_loss': []}
        if use_hm_loss:
            losses['val_img_loss'] = []
            losses['val_hm_loss'] = []
    net.train()

    # Set up optimizer
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Extract noise scheduler info
    n_noise_timesteps = len(noise_scheduler.timesteps)
    pred_noise = noise_scheduler.prediction_type == 'epsilon'

    # Helper function to run the network (so we don't need to repeat this for train and val
    def _run_on_sample(sample):
        # Get some data and prepare the corrupted version
        x = sample[0].to(device) * 2 - 1  # Data on the GPU (mapped to (-1, 1))
        y = sample[1].to(device)
        noise = torch.randn_like(x)
        # TODO: use the same timestep for the whole batch
        timesteps = torch.randint(0, n_noise_timesteps, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
        pred = net(noisy_x, timesteps, y)  # Note that we pass in the labels y

        # Calculate the loss
        if pred_noise:
            img_loss = img_loss_fn(pred, noise)  # How close is the output to the noise
        else:
            img_loss = img_loss_fn(pred, x)  # How close is the output to the noise

        hm_loss = 0
        if im2kp_net is not None:
            # Apply im2kp net to the output image
            if pred_noise:
                pred_im = noisy_x - pred
            else:
                pred_im = pred
            pred_hm = im2kp_net.forward(add_bg_to_img(uncenter_img(pred_im)))

            # Compute loss on the predicted keypoint heatmap
            gt_hm = sample[2].to(device)
            hm_loss = hm_loss_fn(pred_hm, gt_hm)

        # TODO: weigh hm_loss as a function of noise level?
        loss = img_loss + hm_loss * hm_loss_weight

        return loss, img_loss, hm_loss

    # Helper function to print results (so we don't need to repeat this for train and val
    def _print_latest_avg_loss(train_val, n_items):
        avg_img_loss = sum(losses[f'{train_val}_img_loss'][-n_items:]) / n_items
        print_str = f"Avg loss {train_val}: img={avg_img_loss:05f}"
        if use_hm_loss:
            avg_hm_loss = sum(losses[f'{train_val}_img_loss'][-n_items:]) / n_items
            print_str += f", hm={avg_hm_loss:05f}"
        print(print_str)

    # Train the network
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1} / {n_epochs}")
        for sample in tqdm(train_dataloader):
            # Run model on sample
            loss, img_loss, hm_loss = _run_on_sample(sample)

            # Backprop and update the params:
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Store the loss
            losses['train_img_loss'].append(img_loss.item())
            if use_hm_loss:
                losses['train_hm_loss'].append(hm_loss.item())

        # Print out the average loss
        _print_latest_avg_loss(train_val='train', n_items=len(train_dataloader))

        # Run validation
        for sample in tqdm(val_dataloader):
            net.eval()
            loss, img_loss, hm_loss = _run_on_sample(sample)
            losses['val_img_loss'].append(img_loss.item())
            if use_hm_loss:
                losses['val_hm_loss'].append(hm_loss.item())

        # Print out the average loss
        _print_latest_avg_loss(train_val='val', n_items=len(val_dataloader))

    return net, losses


""" Synthesize"""


def center_img(img):
    return img * 2 - 1


def uncenter_img(img):
    return (img + 1) / 2


def add_bg_to_img(bg_subtr):
    # This is a hack that makes the fly body and background roughly the same intensity as in a real image.
    return 0.8 * (1 - bg_subtr)


def make_img(
        bg_img, keypoints, net, noise_scheduler,
        width=100, init_x=None, init_t_idx=0, init_x_w=0.5, im2kp_net=None):
    """ Makes an image from a keypoints of multiple animals in a frame.

    By providing init_x (and the corresponding init_t_idx and init_x_w) one can initialize
    the noise from a previous frame in hopes that it will enforce consistency across frames
    when generating a video.

    By providing im2kp_net, keypoints detected in the generated image are used to shift the
    image when pasted on the bg_img such that the new fly is properly centered, in hopes of
    removing jitter when generating a video.

    Inputs:
        bg_img: Image on which animals will be pasted
        keypoints: 2d keypoints of the tracked animals n_animals x n_keypoints x 2
        net: KeypointConditionedUnet
        noise_scheduler: Noise scheduler associated with the trained net
        width: Width of the images that net expects
        init_x: If provided, use this initialize noise of fies to be diffused. n_animals x 1 x width x width
        init_t_idx: Noise level t to be assigned to init_x.
        init_x_w: Weight of init_x compared when added to noise.
        im2kp_net: Network used to detect keypoints in the generated fly images, to shift them if needed.

    Returns
        pred_img: Predicted image with individual fly images pasted on top of the background.
        pred: Predicted image for the flies in their reference frame. Can be used as x_init to
              the next frame when generating a video.
    """
    device = next(net.parameters()).device

    n_flies, n_kpts = keypoints.shape[:2]
    keypoint_imgs = np.zeros((n_flies, n_kpts, width, width), np.float32)
    buf = width / 2 - 0.5

    angles = []
    cropboxes = []
    for fly_id in range(keypoints.shape[0]):
        kpt = keypoints[fly_id]

        # Extract rotation and center from the keypoints
        center = kpt[5]
        vec = np.array(kpt[0] - kpt[6])
        vec = vec / np.linalg.norm(vec)
        angle = np.rad2deg(np.arctan2(vec[0], -vec[1]))
        angles.append(angle)
        rotmat = cv2.getRotationMatrix2D(center, angle, 1)

        rot_kpt = rotate_keypoints(kpt, rotmat)
        cropbox = get_cropbox(rot_kpt, buf)
        cropboxes.append(cropbox)
        crop_kpt = crop_keypoints(rot_kpt, cropbox)

        # Create rotated and cropped keypoint images
        x, y = np.round(crop_kpt).astype(np.long).T.clip(0, width - 1)
        keypoint_imgs[fly_id, np.arange(n_kpts), y, x] = 1

    # Apply model to keypoint image to generate an image
    x = torch.randn(n_flies, 1, width, width).to(device)
    if init_x is not None:
        x = init_x * init_x_w + x * (1-init_x_w)
        timesteps = noise_scheduler.timesteps[init_t_idx:]
    else:
        timesteps = noise_scheduler.timesteps
    y = torch.from_numpy(keypoint_imgs).to(device)
    for i, t in enumerate(timesteps):
        with torch.no_grad():
            residual = net(x, t, y)
        out = noise_scheduler.step(residual, t, x)
        x = out.prev_sample
    pred = out.pred_original_sample
    # pred_adj = .8 * (1-(pred + 1) / 2)
    pred_adj = add_bg_to_img(uncenter_img(pred))
    pred_imgs = pred_adj.detach().cpu().numpy()

    if im2kp_net is not None:
        pred_keypoints = im2kp_net(pred_adj)

    # Rotate the image and special paste it in the center location over the bg_img
    pred_img = copy.deepcopy(bg_img)

    for fly_id in range(n_flies):
        crop_img = pred_imgs[fly_id][0]

        rotmat = cv2.getRotationMatrix2D((buf, buf), -angles[fly_id], 1)
        new_crop_img = cv2.warpAffine(crop_img.astype(np.float32), rotmat, (width, width), borderValue=1.0)

        minx, maxx, miny, maxy = cropboxes[fly_id]

        if im2kp_net is not None:
            kpts = heatmap2keypoints(pred_keypoints[fly_id].detach().cpu().numpy())
            kpts = rotate_keypoints(kpts, rotmat)
            x, y = np.round(kpts[5]).astype(int)
            minx = minx - x + width//2
            miny = miny - y + width//2

        thresh = np.median([crop_img[1, 1], crop_img[-1, -1], crop_img[-1, 1], crop_img[1, -1]]) - 0.02
        I, J = np.where(new_crop_img < thresh)
        pred_img[I + miny, J + minx] = new_crop_img[I, J]

    return pred_img, pred


def make_video(video_path, bg_img, all_keypoints, net, noise_scheduler, width=100, im2kp_net=None):
    # Make an image individually for each frame
    frames = [make_img(bg_img, keypoints, net, noise_scheduler, width, im2kp_net=im2kp_net)
              for keypoints in all_keypoints]

    # Turn it into a video
    height, width = frames[0].shape
    video = cv2.VideoWriter(video_path, 0, 1, (width, height))

    for frame in frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

    print(f"Video successfully saved to {video_path}")
