import numpy as np
from flyllm.features import kp2feat, feat2kp
from flyllm.config import keypointnames


def test_kp2feat2kp():
    # Generate some keypoints (copied from dataset)
    Xkp = np.array([
        [19.181751 , -6.3604794],
        [19.08275  , -6.242819 ],
        [21.333103 , -3.4102457],
        [20.753012 , -3.5140944],
        [21.338167 , -3.9927166],
        [21.229656 , -4.081912 ],
        [20.680365 , -3.6455448],
        [20.379992 , -4.592538 ],
        [19.779501 , -5.4755116],
        [20.21491  , -4.112506 ],
        [19.940474 , -4.0738897],
        [20.900364 , -4.6502776],
        [21.06569  , -4.922514 ],
        [20.867718 , -3.1095588],
        [19.36815  , -3.8049867],
        [19.08863  , -4.8691125],
        [20.227793 , -6.3686438],
        [21.490784 , -5.2183347],
        [21.599771 , -3.387719 ]
    ])

    # Convert them to feat and back to keypoints
    Xfeat, scale_perfly, flyid = kp2feat(Xkp, return_scale=True)
    Xkp_new = feat2kp(Xfeat, scale_perfly, flyid)

    # Verify that they are the same as original keypoints
    dists = np.linalg.norm(Xkp - Xkp_new, axis=1)
    max_dist = dists.max()
    max_dist_idx = np.argwhere(dists == max_dist)[0][0]
    assert max_dist < 0.02, f"Distance for keypoint '{keypointnames[max_dist_idx]}' is too large: {max_dist}"


if __name__ == "__main__":
    test_kp2feat2kp()