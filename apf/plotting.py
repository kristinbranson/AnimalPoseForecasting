
import cv2
import numpy as np
import io
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple
import subprocess
import os


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


def arrays_to_video(
        frames: List[np.ndarray],
        output_path: str,
        fps: float = 30.0,
        size: Tuple[int, int] = None,
        codec: str = "mp4v",
        color_format: str = "gray",
) -> None:
    """
    Write a list of numpy image arrays to a video file.

    Args:
        frames:       List of numpy arrays. Each frame can be:
                        - (H, W, 3)  — color image
                        - (H, W)     — grayscale image (auto-converted to BGR)
                      All frames must have the same spatial dimensions.
        output_path:  Destination file path (e.g. "output.mp4", "clip.avi").
        fps:          Frames per second for the output video. Default: 30.
        size:         (width, height) to resize every frame to. If None, the
                      size of the first frame is used.
        codec:        FourCC codec string. Common options:
                        "mp4v"  — .mp4 (default)
                        "XVID"  — .avi
                        "MJPG"  — .avi (Motion JPEG)
                        "avc1"  — .mp4 (H.264, platform-dependent)
        color_format: "rgb"  — input frames are RGB   (converted to BGR for OpenCV)
                      "bgr"  — input frames are already BGR (no conversion)
                      "gray" — input frames are grayscale

    Raises:
        ValueError: if `frames` is empty or frames have inconsistent shapes.
        IOError:    if the VideoWriter fails to open.
    """
    if not frames:
        raise ValueError("The frames list is empty.")

    # ------------------------------------------------------------------ #
    # Determine output frame size
    # ------------------------------------------------------------------ #
    first = frames[0]
    if first.ndim == 2:
        h, w = first.shape
    elif first.ndim == 3:
        h, w = first.shape[:2]
    else:
        raise ValueError(f"Unsupported frame shape: {first.shape}. Expected (H,W) or (H,W,C).")

    out_w, out_h = size if size else (w, h)

    # ------------------------------------------------------------------ #
    # Set up VideoWriter
    # ------------------------------------------------------------------ #
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    if not out.isOpened():
        raise IOError(
            f"cv2.VideoWriter failed to open '{output_path}'. "
            "Check that the codec is supported and the output directory exists."
        )

    # ------------------------------------------------------------------ #
    # Write frames
    # ------------------------------------------------------------------ #
    for i, frame in enumerate(frames):
        frame = np.asarray(frame)

        # --- normalise dtype to uint8 ---
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                # assume [0.0, 1.0] → scale to [0, 255]
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # --- handle grayscale ---
        if frame.ndim == 2 or color_format == "gray":
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY if color_format == "rgb" else cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # --- convert RGB → BGR (OpenCV native) ---
        elif color_format == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # --- resize if needed ---
        if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        out.write(frame)

    out.release()
    print(f"Video saved → {Path(output_path).resolve()}  ({len(frames)} frames @ {fps} fps)")


def convert_video_to_mp4(input_path):
    """
        Runs ffmpeg -i train_10_wingext_500_30fps.avi -c:v libx264 -pix_fmt yuv420p train_10_wingext_500_30fps.mp4
    """
    name, ext = os.path.splitext(input_path)
    subprocess.run(["ffmpeg", "-i", input_path, "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{name}.mp4"])


def fig_to_numpy(fig):
    """Convert a matplotlib figure to an RGB numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    img_array = plt.imread(buf)  # shape: (H, W, 4) RGBA, float32 in [0, 1]
    buf.close()

    rgb = (img_array[:, :, :3] * 255).astype(np.uint8)  # drop alpha, scale to uint8
    return rgb


def distinct_colors(n, cmap="rainbow"):
    cmap = plt.get_cmap(cmap)
    return np.array([cmap(i / n) for i in range(n)])  # shape: (n, 4), RGBA


def plot_array_as_table(data, row_labels, col_labels, title='Data Table', pdf=None):
    """
    Save a 2D array as a table in the PDF.

    Parameters:
    - pdf: PdfPages object
    - data: 2D numpy array or list of lists
    - row_labels: list of labels for rows
    - col_labels: list of labels for columns
    - title: title for the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    ax.axis('tight')
    ax.axis('off')

    # Convert data to list of lists if it's a numpy array
    if hasattr(data, 'tolist'):
        table_data = data.tolist()
    else:
        table_data = data

    # Add row labels to the left of each row
    table_data_with_labels = [[row_labels[i]] + row for i, row in enumerate(table_data)]

    # Add column labels as the first row
    full_table = [[''] + col_labels] + table_data_with_labels

    # Create the table
    table = ax.table(cellText=full_table, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Make header row bold
    for i in range(len(full_table[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Make first column (row labels) bold
    for i in range(1, len(full_table)):
        table[(i, 0)].set_facecolor('#E8E8E8')
        table[(i, 0)].set_text_props(weight='bold')

    if pdf is not None:
        pdf.savefig(fig)
        plt.close()
    else:
        plt.show()
