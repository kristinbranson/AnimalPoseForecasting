import numpy as np
from tqdm import tqdm
import os
import json
import urllib
import zipfile
import cv2
import matplotlib
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset

print('CUDA available: %d' % torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_im2kp_datasets(dataurl: str, outzipfile: str, overwrite: bool = False, crop_width=120):
    datadir, trainannfile, testannfile = fetch_data(dataurl, outzipfile, overwrite=overwrite)
    train_dataset = COCODataset(trainannfile, datadir=datadir, width=crop_width)
    test_dataset = COCODataset(testannfile, datadir=datadir, width=crop_width)
    return train_dataset, test_dataset


def fetch_data(dataurl, outzipfile, overwrite=False):
    # download zip file
    if not os.path.exists(outzipfile) or overwrite:
        print(f'Downloading from {dataurl} to {outzipfile}...')
        urllib.request.urlretrieve(dataurl, outzipfile)
    else:
        print(f'{outzipfile} already exists, skipping download')

    # Extract the zip file
    with zipfile.ZipFile(outzipfile, 'r') as zip_ref:
        filelist = zip_ref.namelist()
        datadir = filelist[0]
        if overwrite or (not os.path.exists(datadir)):
            print(f'Unzipping {outzipfile}')
            zip_ref.extractall('.')
        else:
            print(f'Data directory {datadir} already exists, skipping unzipping')

    # Locations of the training and test data
    trainannfile = os.path.join(datadir, 'train_annotations.json')
    testannfile = os.path.join(datadir, 'test_annotations.json')
    return datadir, trainannfile, testannfile


def heatmap2image(hm, cmap='jet', colors=None):
    """
    Creates and returns an image visualization from keypoint heatmaps. Each
    keypoint is colored according to the input cmap/colors.

    Inputs:
        hm: nkeypoints x height x width ndarray, dtype=float in the range 0 to 1.
            hm[p,i,j] is a score indicating how likely it is that the pth keypoint
            is at pixel location (i,j).
        cmap: string. Name of colormapx for defining colors of keypoint points.
            Used only if colors is None. Default: 'jet'
        colors: list of length nkeypoints.
            colors[p] is an ndarray of size (4,) indicating the color to use for the
            pth keypoint. colors is the output of matplotlib's colormap functions.
            Default: None

    Output:
        im: height x width x 3 ndarray. Image representation of the input heatmap keypoints.
    """
    hm = np.maximum(0., np.minimum(1., hm))
    im = np.zeros((hm.shape[1], hm.shape[2], 3))
    if colors is None:
        if isinstance(cmap, str):
            # cmap = matplotlib.cm.get_cmap(cmap)
            cmap = matplotlib.colormaps[cmap]
        colornorm = matplotlib.colors.Normalize(vmin=0, vmax=hm.shape[0])
        colors = cmap(colornorm(np.arange(hm.shape[0])))
    for i in range(hm.shape[0]):
        color = colors[i]
        for c in range(3):
            im[..., c] = im[..., c] + (color[c] * .7 + .3) * hm[i, ...]
    im = np.minimum(1., im)
    return im


def heatmap2keypoints(hms):
    idx = np.argmax(hms.reshape(hms.shape[:-2] + (hms.shape[-2] * hms.shape[-1],)), axis=-1)
    locs = np.zeros(hms.shape[:-2] + (2,))
    locs[..., 1], locs[..., 0] = np.unravel_index(idx, hms.shape[-2:])
    return locs


class COCODataset(torch.utils.data.Dataset):
    """
  Torch Dataset based on the COCO keypoint file format with more
  options implemented.
  """

    def __init__(self, annfile, datadir=None, label_sigma=3., label_filter_r=1, transform=None,
                 keypoints=None, movs=None, tgts=None, maxsamples=np.inf, width=None):
        """
    Inputs:
    annfile: string
    Path to json file containing annotations.
    datadir: string
    Path to directory containing images. If None, images are assumed to be in
    the working directory.
    Default: None
    label_sigma: scalar float
    Standard deviation in pixels of Gaussian to be used to make the keypoint
    heatmap.
    Default: 3.
    transform: None
    Not used currently
    keypoints: ndarray (or list, something used for indexing into ndarray)
    Indices of keypoints available to use in this dataset. Reducing the
    keypoints used can make training faster and require less memory, and is
    useful for testing code. If None, all keypoints are used.
    Default: None
    """

        # read in the annotations from the json file
        with open(annfile) as f:
            self.ann = json.load(f)
        # where the images are
        self.datadir = datadir

        # keypoints to use
        self.nkeypoints_all = len(self.ann['annotations'][0]['keypoints']) // 3
        if keypoints is None:
            self.nkeypoints = self.nkeypoints_all
        else:
            self.nkeypoints = len(keypoints)
        self.keypoints = keypoints

        # for data augmentation/rescaling
        self.transform = transform

        # limit data to specified movies or flies
        n = len(self.ann['annotations'])
        canselect = np.ones(n, dtype=bool)
        if movs is not None:
            m = np.array(list(map(lambda x: x['mov'], self.ann['annotations'])))
            canselect = np.logical_and(canselect, np.isin(m, movs))
        if tgts is not None:
            t = np.array(list(map(lambda x: x['tgt'], self.ann['annotations'])))
            canselect = np.logical_and(canselect, np.isin(t, tgts))
        idx = np.where(canselect)[0]

        if maxsamples is None:
            maxsamples = np.inf
        nsubsample = int(np.minimum(maxsamples, len(idx)))
        if nsubsample < n:
            idx = np.random.choice(idx, nsubsample)

        if len(idx) < n:
            self.ann['annotations'] = [self.ann['annotations'][i] for i in idx]
            self.ann['images'] = [self.ann['images'][i] for i in idx]

        # output will be heatmap images, one per keypoint, with Gaussian values
        # around the keypoint location -- precompute some stuff for that
        self.label_filter = None
        self.label_filter_r = label_filter_r
        self.label_filter_d = 2 * label_filter_r + 1
        self.label_sigma = label_sigma
        self.init_label_filter()
        self.width = width

    def __len__(self):
        """
    Overloaded len function.
    This must be defined in every Torch Dataset and must take only self
    as input.
    Returns the number of examples in the dataset.
    """
        return len(self.ann['annotations'])

    def __getitem__(self, item):
        """
    Overloaded getitem function.
    This must be defined in every Torch Dataset and must take only self
    and item as input. It returns example number item.
    item: scalar integer.
    The output example is a dict with the following fields:
    image: torch float32 tensor of size ncolors x height x width
    keypoints: nkeypoints x 2 float ndarray
    heatmaps: torch float32 tensor of size nkeypoints x height x width
    id: scalar integer, contains item
    """

        # keypoint locations
        # these are stored as x-coordinate, y-coordinate, and a flag indicating visibility.
        # visibility: v = 0: not labeled, v = 1: labeled but not visible, v = 2: labeled and visible
        locs = np.reshape(self.ann['annotations'][item]['keypoints'], [self.nkeypoints_all, 3])
        locs = locs[:, :2]
        if self.keypoints is not None:
            locs = locs[self.keypoints, :]

        # read in the image for training example item
        # and convert to a torch tensor
        filename = self.ann['images'][item]['file_name']
        if self.datadir is not None:
            filename = os.path.join(self.datadir, filename)
        assert os.path.exists(filename)
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            im, locs = self.transform(im, locs)

        im = torch.from_numpy(im)
        # convert to float32 in the range 0. to 1.
        if im.dtype == float:
            pass
        elif im.dtype == torch.uint8:
            im = im.float() / 255.
        elif im.dtype == torch.uint16:
            im = im.float() / 65535.
        else:
            print('Cannot handle im type ' + str(im.dtype))
            raise TypeError

        imsz = im.shape
        # convert to a tensor of size ncolors x h x w
        if im.dim() == 3:
            im = torch.transpose(im, [2, 0, 1])  # now 3 x h x w
        else:
            im = torch.unsqueeze(im, 0)  # now 1 x h x w

        # create heatmap target prediction
        heatmaps = self.make_heatmap_target(locs, imsz)

        # TODO: Crop images (and keypoints) to the size that I want
        if self.width is not None:
            buf = (im.shape[-1] - self.width) // 2
            im = im[..., buf:buf + self.width, buf:buf + self.width]
            heatmaps = heatmaps[..., buf:buf + self.width, buf:buf + self.width]
            locs = locs - buf

        # return a dict with the following fields:
        # image: torch float32 tensor of size ncolors x height x width
        # keypoints: nkeypoints x 2 float ndarray
        # heatmaps: torch float32 tensor of size nkeypoints x height x width
        # id: scalar integer, contains item
        features = {'image': im,
                    'keypoints': locs.astype(np.float32),
                    'heatmaps': heatmaps,
                    'id': item}

        return features

    def init_label_filter(self):
        """
    init_label_filter(self)
    Helper function
    Create a Gaussian filter for the heatmap target output
    """
        # radius of the filter
        self.label_filter_r = max(int(round(3 * self.label_sigma)), 1)
        # diameter of the filter
        self.label_filter_d = 2 * self.label_filter_r + 1

        # allocate
        self.label_filter = np.zeros([self.label_filter_d, self.label_filter_d])
        # set the middle pixel to 1.
        self.label_filter[self.label_filter_r, self.label_filter_r] = 1.
        # blur with a Gaussian
        self.label_filter = cv2.GaussianBlur(self.label_filter, (self.label_filter_d, self.label_filter_d),
                                             self.label_sigma)
        # normalize
        self.label_filter = self.label_filter / np.max(self.label_filter)
        # convert to torch tensor
        self.label_filter = torch.from_numpy(self.label_filter)

    def make_heatmap_target(self, locs, imsz):
        """
        make_heatmap_target(self,locs,imsz):
        Helper function
        Creates the heatmap tensor of size imsz corresponding to keypoint locations locs

        Inputs:
            locs: nkeypoints x 2 ndarray, Locations of keypoints
            imsz: image shape

        Returns:
            target: torch tensor of size nkeypoints x imsz[0] x imsz[1], Heatmaps corresponding to locs
        """
        label_filter_r = self.label_filter_r
        label_filter_d = self.label_filter_d
        label_filter = self.label_filter

        # allocate the tensor
        target = torch.zeros((locs.shape[0], imsz[0], imsz[1]), dtype=torch.float32)
        # loop through keypoints
        for i in range(locs.shape[0]):
            # location of this keypoint to the nearest pixel
            x = int(np.round(locs[i, 0]))  # losing sub-pixel accuracy
            y = int(np.round(locs[i, 1]))
            # edges of the Gaussian filter to place, minding border of image
            x0 = np.maximum(0, x - label_filter_r)
            x1 = np.minimum(imsz[1] - 1, x + label_filter_r)
            y0 = np.maximum(0, y - label_filter_r)
            y1 = np.minimum(imsz[0] - 1, y + label_filter_r)
            # crop filter if it goes outside of the image
            fil_x0 = label_filter_r - (x - x0)
            fil_x1 = label_filter_d - (label_filter_r - (x1 - x))
            fil_y0 = label_filter_r - (y - y0)
            fil_y1 = label_filter_d - (label_filter_r - (y1 - y))
            # copy the filter to the relevant part of the heatmap image
            target[i, y0:y1 + 1, x0:x1 + 1] = label_filter[fil_y0:fil_y1 + 1, fil_x0:fil_x1 + 1]
        return target

    @staticmethod
    def get_image(d, i=None):
        """
    static function, used for visualization
    COCODataset.get_image(d,i=None)
    Returns an image usable with plt.imshow()
    Inputs:
    d: if i is None, item from a COCODataset.
    if i is a scalar, d is a batch of examples from a COCO Dataset returned
    by a DataLoader.
    i: Index of example into the batch d, or None if d is a single example
    Returns the ith image from the patch as an ndarray plottable with
    plt.imshow()
    """
        if i is None:
            im = np.squeeze(np.transpose(d['image'].cpu().numpy(), (1, 2, 0)), axis=2)
        else:
            im = np.squeeze(np.transpose(d['image'][i, ...].cpu().numpy(), (1, 2, 0)), axis=2)
        return im

    @staticmethod
    def get_keypoints(d, i=None):
        """
    static helper function
    COCODataset.get_keypoints(d,i=None)
    Returns a nkeypoints x 2 ndarray indicating keypoint locations.
    Inputs:
    d: if i is None, item from a COCODataset.
    if i is a scalar, batch of examples from a COCO Dataset returned
    by a DataLoader.
    i: Index of example into the batch d, or None if d is a single example
    """
        if i is None:
            locs = d['keypoints']
        else:
            locs = d['keypoints'][i]
        return locs

    @staticmethod
    def get_heatmap_image(d, i=None, cmap='jet', colors=None):
        """
    static function, used for visualization
    COCODataset.get_heatmap_image(d,i=None)
    Returns an image visualization of heatmaps usable with plt.imshow()
    Inputs:
    d: if i is None, item from a COCODataset.
    if i is a scalar, batch of examples from a COCO Dataset returned
    by a DataLoader.
    i: Index of example into the batch d, or None if d is a single example
    Returns the ith heatmap from the patch as an ndarray plottable with
    plt.imshow()
    cmap: string.
    Name of colormap for defining colors of keypoint points. Used only if colors
    is None.
    Default: 'jet'
    colors: list of length nkeypoints.
    colors[p] is an ndarray of size (4,) indicating the color to use for the
    pth keypoint. colors is the output of matplotlib's colormap functions.
    Default: None
    Output:
    im: height x width x 3 ndarray
    Image representation of the input heatmap keypoints.
    """
        if i is None:
            hm = d['heatmaps']
        else:
            hm = d['heatmaps'][i, ...]
        hm = hm.cpu().numpy()
        im = heatmap2image(hm, cmap=cmap, colors=colors)
        return im


def make_heatmap_from_coco(kps, width, coco_dataset, buffer=50):
    hm_large = coco_dataset.make_heatmap_target(kps + buffer, (width+buffer*2, width+buffer*2))
    return hm_large[:, buffer:-buffer, buffer:-buffer]


# Copy-paste & modify from https://github.com/milesial/Pytorch-UNet

# The UNet is defined modularly.
# It is a series of downsampling layers defined by the module Down
# followed by upsampling layers defined by the module Up. The output is
# a convolutional layer with an output channel for each keypoint, defined by
# the module OutConv.
# Each down and up layer is actually two convolutional layers with
# a ReLU nonlinearity and batch normalization, defined by the module
# DoubleConv.
# The Down module consists of a 2x2 max pool layer followed by the DoubleConv
# module.
# The Up module consists of an upsampling, either defined via bilinear
# interpolation (bilinear=True), or a learned convolutional transpose, followed
# by a DoubleConv module.
# The Output layer is a single 2-D convolutional layer with no nonlinearity.
# It does not include a nonlinearity because this final nonlinearity is part of
# the network loss function.

# Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# ReLU: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
# MaxPool2d: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
# BatchNorm2d: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
# ConvTranspose2d: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

class DoubleConv(nn.Module):
    """
  (convolution => [BN] => ReLU) * 2
  """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# copy-pasted and modified from unet_model.py

class UNet(nn.Module):
    def __init__(self, n_channels, n_keypoints, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_keypoints = n_keypoints
        self.bilinear = bilinear
        self.nchannels_inc = 8

        # define the layers

        # number of channels in the first layer
        nchannels_inc = self.nchannels_inc
        # increase the number of channels by a factor of 2 each layer
        nchannels_down1 = nchannels_inc * 2
        nchannels_down2 = nchannels_down1 * 2
        nchannels_down3 = nchannels_down2 * 2
        # decrease the number of channels by a factor of 2 each layer
        nchannels_up1 = nchannels_down3 // 2
        nchannels_up2 = nchannels_up1 // 2
        nchannels_up3 = nchannels_up2 // 2

        if bilinear:
            factor = 2
        else:
            factor = 1

        self.layer_inc = DoubleConv(n_channels, nchannels_inc)

        self.layer_down1 = Down(nchannels_inc, nchannels_down1)
        self.layer_down2 = Down(nchannels_down1, nchannels_down2)
        self.layer_down3 = Down(nchannels_down2, nchannels_down3 // factor)

        self.layer_up1 = Up(nchannels_down3, nchannels_up1 // factor, bilinear)
        self.layer_up2 = Up(nchannels_up1, nchannels_up2 // factor, bilinear)
        self.layer_up3 = Up(nchannels_up2, nchannels_up3 // factor, bilinear)

        self.layer_outc = OutConv(nchannels_up3 // factor, self.n_keypoints)

    def forward(self, x, verbose=False, return_intermediate=False):

        # return_intermediate and verbose options are to visualize
        # the intermediate outputs

        if return_intermediate:
            intermediate = []

        x1 = self.layer_inc(x)
        if verbose: print('inc: shape = ' + str(x1.shape))
        x2 = self.layer_down1(x1)
        if verbose: print('down1: shape = ' + str(x2.shape))
        x3 = self.layer_down2(x2)
        if verbose: print('down2: shape = ' + str(x3.shape))
        x4 = self.layer_down3(x3)
        if verbose: print('down3: shape = ' + str(x4.shape))

        if return_intermediate:
            intermediate.append(x1.clone())
            intermediate.append(x2.clone())
            intermediate.append(x3.clone())
            intermediate.append(x4.clone())

        x = self.layer_up1(x4, x3)
        if verbose: print('up1: shape = ' + str(x.shape))
        if return_intermediate:
            intermediate.append(x.clone())

        x = self.layer_up2(x, x2)
        if verbose: print('up2: shape = ' + str(x.shape))
        if return_intermediate:
            intermediate.append(x.clone())

        x = self.layer_up3(x, x1)
        if verbose: print('up3: shape = ' + str(x.shape))
        if return_intermediate:
            intermediate.append(x.clone())

        logits = self.layer_outc(x)
        if verbose: print('outc: shape = ' + str(logits.shape))
        if return_intermediate:
            intermediate.append(logits.clone())

        if return_intermediate:
            return intermediate
        else:
            return logits

    def output(self, x, verbose=False):
        return torch.sigmoid(self.forward(x, verbose=verbose))

    def __str__(self):
        s = ''
        s += 'inc: ' + str(self.layer_inc) + '\n'
        s += 'down1: ' + str(self.layer_down1) + '\n'
        s += 'down2: ' + str(self.layer_down2) + '\n'
        s += 'down3: ' + str(self.layer_down3) + '\n'
        s += 'up1: ' + str(self.layer_up1) + '\n'
        s += 'up2: ' + str(self.layer_up2) + '\n'
        s += 'up3: ' + str(self.layer_up3) + '\n'
        s += 'outc: ' + str(self.layer_outc) + '\n'
        return s

    def __repr__(self):
        return str(self)


def init_unet(imsize, train_dataset):
    net = UNet(n_channels=imsize[-1], n_keypoints=train_dataset.nkeypoints)
    net.to(device=device)  # have to be careful about what is done on the CPU vs GPU
    return net


def train_unet(train_dataset,
               net,
               checkpointdir,
               loadepoch=0,
               batchsize=8,
               nepochs=None,
               learning_rate=0.001,
               weight_decay=1e-8,
               momentum=0.9,
               nepochs_per_save=None,
               val_frac=0.1
               ):
    """
  Following https://github.com/milesial/Pytorch-UNet/blob/master/train.py

  Inputs:
  train_dataset: Required. COCODataset instance defining training data.
  net: Required. Initialized unet.
  checkpointdir: Required. Location to store networks to.
  batchsize=8: Number of images per batch -- amount of required memory for
  training will increase linearly in batchsize
  nepochs=None: Number of times to cycle through all the data during training.
  If None, will be computed as int(np.round(25000./len(train_dataset)))
  learning_rate=0.001: Initial learning rate
  weight_decay = 1e-8: How learning rate decays over time
  momentum = 0.9: how much to use previous gradient direction
  nepochs_per_save = None: how often to save the network. If None, set to
  int(np.round(10000./len(train_dataset)))
  val_frac = 0.1: what fraction of data to use for validation
  Outputs:
  net: Trained unet.
  train_dataloader: Training data loader
  val_dataloader: Validation data loader
  epoch_losses: Training loss at each epoch.
  savefile: Location network was saved to.
  """

    if nepochs is None:
        nepochs = int(np.round(60000. / len(train_dataset)))
    if nepochs_per_save is None:
        nepochs_per_save = int(np.round(20000. / len(train_dataset)))

    # split into train and validation datasets
    n_val = int(len(train_dataset) * val_frac)
    n_train = len(train_dataset) - n_val
    train, val = torch.utils.data.random_split(train_dataset, [n_train, n_val])
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batchsize, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batchsize, shuffle=False)

    # gradient descent flavor
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    # Following https://github.com/milesial/Pytorch-UNet
    # Use binary cross entropy loss combined with sigmoid output activation function.
    # We combine here for numerical improvements
    criterion = nn.BCEWithLogitsLoss()

    # store loss per epoch
    epoch_losses = np.zeros(nepochs)
    epoch_losses[:] = np.nan

    # when we last saved the network
    saveepoch = None

    # how many gradient descent updates we have made
    iters = loadepoch * len(train_dataloader)

    # loop through entire training data set nepochs times
    for epoch in range(loadepoch, nepochs):
        net.train()  # put in train mode (affects batchnorm)
        epoch_loss = 0
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{nepochs}', unit='img') as pbar:
            # loop through each batch in the training data
            for batch in train_dataloader:
                # compute the loss
                imgs = batch['image']
                imgs = imgs.to(device=device, dtype=torch.float32)  # transfer to GPU
                hm_labels = batch['heatmaps']
                hm_labels = hm_labels.to(device=device, dtype=torch.float32)  # transfer to GPU
                hm_preds = net(imgs)  # evaluate network on batch
                loss = criterion(hm_preds, hm_labels)  # compute loss
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # gradient descent
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                iters += 1
                pbar.update(imgs.shape[0])

        print('loss (epoch) = %f' % epoch_loss)
        epoch_losses[epoch] = epoch_loss

        # save checkpoint networks every now and then
        if epoch % nepochs_per_save == 0:
            print('Saving network state at epoch %d' % (epoch + 1))
            # only keep around the last two epochs for space purposes
            if saveepoch is not None:
                savefile0 = os.path.join(checkpointdir, f'CP_latest_epoch{saveepoch + 1}.pth')
                savefile1 = os.path.join(checkpointdir, f'CP_prev_epoch{saveepoch + 1}.pth')
                if os.path.exists(savefile0):
                    try:
                        os.rename(savefile0, savefile1)
                    except:
                        print('Failed to rename checkpoint file %s to %s' % (savefile0, savefile1))
            saveepoch = epoch
            savefile = os.path.join(checkpointdir, f'CP_latest_epoch{saveepoch + 1}.pth')
            torch.save(net.state_dict(), os.path.join(checkpointdir, f'CP_latest_epoch{epoch + 1}.pth'))

    savefile = os.path.join(checkpointdir, f'Final_epoch{epoch + 1}.pth')
    torch.save(net.state_dict(), savefile)

    return net, train_dataloader, val_dataloader, epoch_losses, savefile

