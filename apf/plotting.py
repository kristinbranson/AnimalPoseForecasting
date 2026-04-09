
import cv2
import numpy as np

# Get a cropped frame of the fly at the beginning of the thing
def get_rotmat(kpt, mid_idx=7, vec_ids=(5, 6)):
    # center around kpt 5
    if type(mid_idx) is list or type(mid_idx) is np.ndarray:
        center = kpt[mid_idx].mean(0)
    else:
        center = kpt[mid_idx]
    # rotate along kpts 0-6
    pt1 = vec_ids[0]
    pt2 = vec_ids[1]
    # vec = np.array(kpt[pt1] - kpt[pt2])
    # vec = vec / np.linalg.norm(vec)


    fthorax = np.array(kpt[pt1] + kpt[pt2] ) /2
    vec = np.array(fthorax - center)

    angle = np.rad2deg(np.arctan2(vec[0], -vec[1]))

    # bthorax = Xkp[keypointnames.index('base_thorax')]
    # lthorax = Xkp[keypointnames.index('left_front_thorax')]
    # rthorax = Xkp[keypointnames.index('right_front_thorax')]
    # fthorax = (lthorax + rthorax) / 2.
    # vec = fthorax - bthorax
    # thorax_theta = mod2pi(np.arctan2(vec[1, ...], vec[0, ...]) - np.pi / 2)
    # Xn = rotate_2d_points(Xkp - fthorax[np.newaxis, ...], thorax_theta)

    # return cv2.getRotationMatrix2D(center, angle, 1)

    return cv2.getRotationMatrix2D(fthorax, angle, 1)


def rotate_image(img, rotmat):
    height, width = img.shape
    im_type = img.dtype
    print(im_type)
    return cv2.warpAffine(img.astype(np.float32), rotmat, (width, height)).astype(im_type)


def get_cropbox(kpt, buf=49.5, mid_idx=[5, 6]):
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


def rotate_keypoints(kpt, rotmat):
    return np.dot(kpt, rotmat[:2, :2].T) + rotmat[:2, 2]


def crop_keypoints(kpt, cropbox):
    minx, maxx, miny, maxy = cropbox
    return kpt - np.array([minx, miny])[None, :]
