import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

from apf.io import read_config, load_and_filter_data
from apf.dataset import Zscore, Discretize, Data, Dataset, Operation, Fusion, Subset, FutureAsInput, Identity, apply_opers_from_data
from apf.training import train
from apf.utils import function_args_from_config, modrange, rotate_2d_points, set_invalid_ends
from apf.models import initialize_model
from apf.data import debug_less_data
from apf.simulation import simulate

from flyllm.config import DEFAULTCONFIGFILE, posenames, featrelative, keypointnames, featangle
from flyllm.features import (
    featglobal, get_sensory_feature_idx, kp2feat, compute_sensory_wrapper, compute_global_velocity,
    compute_relpose_velocity, compute_scale_perfly, compute_noise_params, feat2kp
)
from flyllm.simulation import animate_pose


class Sensory(Operation):
    def __init__(self):
        self.idxinfo = None

    def apply(self, Xkp):
        feats = []
        for flyid in range(Xkp.shape[0]):
            feat, idxinfo = compute_sensory_wrapper(Xkp.T, flyid, returnidx=True)
            feats.append(feat.T)
        self.idxinfo = idxinfo
        return np.array(feats)

    def inverse(self, sensory):
        print(f"Operation {self} is not invertible")
        return None


class Pose(Operation):
    """Includes global and local pose

    """
    # def __init__(self):
    #     pass

    def apply(self, Xkp, scale_perfly=None, flyid=None):
        # Compute scale_perfly
        # Store it for inversibility
        self.scale_perfly = scale_perfly
        return kp2feat(Xkp=Xkp.T, scale_perfly=scale_perfly, flyid=flyid).T

    def inverse(self, pose, flyid=None):
        return feat2kp(pose.T, scale_perfly=self.scale_perfly, flyid=flyid).T


class LocalVelocity(Operation):
    """Velocity and it's inverse
    """
    def apply(self, pose, isstart=None):
        pose_velocity = np.moveaxis(compute_relpose_velocity(pose.T), 2, 0)
        if isstart is not None:
            set_invalid_ends(pose_velocity, isstart, dt=1)
        pose_velocity = pose_velocity[0]
        return pose_velocity.T

    def inverse(self, velocity, x0=None):
        """
        Params:
            velocity: n_agents x n_frames x n_features
            x0
        """
        # Note: here we are assuming dt=1
        if x0 is None:
            n_agents, _, n_features = velocity.shape
            x0 = np.zeros((n_agents, n_features))
        velocity = np.concatenate([x0[:, None, :], velocity], axis=1)[:, :-1, :]
        pose = np.cumsum(velocity, axis=1)
        is_angle = np.where(featangle[featrelative])[0]
        for i in is_angle:
            pose[..., i] = modrange(pose[..., i], -np.pi, np.pi)
        return pose


class GlobalVelocity(Operation):
    def __init__(self, tspred: list[int]):
        self.tspred = tspred

    def apply(self, position, isstart=None):
        Xorigin = position[..., :2].T
        Xtheta = position[..., 2].T
        _, n_frames, n_flies = Xorigin.shape
        dXoriginrel, dtheta = compute_global_velocity(Xorigin, Xtheta, self.tspred)
        movement_global = np.concatenate((dXoriginrel[:, [1, 0]], dtheta[:, None, :, :]), axis=1)
        if isstart is not None:
            for movement, dt in zip(movement_global, self.tspred):
                set_invalid_ends(movement, isstart, dt=dt)
        movement_global = movement_global.reshape((-1, n_frames, n_flies))

        return movement_global.T

    def inverse(self, velocity, x0=None):
        """

            velocity: n_agents x n_frames x 3
            x0: n_agents x 3
        """
        # Note: here we are assuming dt=1
        if x0 is None:
            n_agents, _, n_dim = velocity.shape
            x0 = np.zeros((n_agents, n_dim))

        d_theta = np.concatenate([x0[:, None, 2], velocity[:, :, 2]], axis=1)
        theta = modrange(np.cumsum(d_theta, axis=1), -np.pi, np.pi)[:, :-1]

        d_pos_rel = velocity[..., [1, 0]]
        d_pos = rotate_2d_points(d_pos_rel.transpose((1, 2, 0)), -theta.T).transpose((2, 0, 1))
        d_pos = np.concatenate([x0[:, None, :2], d_pos], axis=1)
        pos = np.cumsum(d_pos, axis=1)[:, :-1, :]
        return np.concatenate([pos, theta[:, :, None]], axis=-1)


class Velocity(Operation):
    def __init__(self):
        self.global_inds = np.where(~featrelative)[0]
        self.local_inds = np.where(featrelative)[0]
        self.fusion = Fusion([GlobalVelocity(tspred=[1]), LocalVelocity()], [self.global_inds, self.local_inds])

    def apply(self, pose, isstart=None):
        return self.fusion.apply(pose, kwargs_per_op={'isstart': isstart})

    def inverse(self, velocity, x0=None):
        if x0 is not None:
            kwargs_per_op = [{'x0': x0[..., self.global_inds]}, {'x0': x0[..., self.local_inds]}]
        else:
            kwargs_per_op = None
        return self.fusion.inverse(velocity, kwargs_per_op)


def load_npz_data(infile, config):
    return load_and_filter_data(
        infile,
        config,
        compute_scale_per_agent=compute_scale_perfly,
        compute_noise_params=compute_noise_params,
        keypointnames=keypointnames,
    )


def load_data(config, datafile, debug=False):
    data, scale_perfly = load_npz_data(datafile, config)
    if debug:
        debug_less_data(data, n_frames_per_video=45000, max_n_videos=2)

    # Remove all NaN agents (sometimes the last one is a dummy)
    Xkp = data['X']
    valid = np.sum(~np.isnan(Xkp[0, 0]), axis=-2) > 0
    Xkp = Xkp[..., valid]
    flyids = data['ids'][..., valid]
    isstart = data['isstart'][..., valid]
    isdata = data['isdata'][..., valid]

    return Xkp, flyids, isstart, isdata, scale_perfly


def make_dataset(config, filename, ref_dataset=None, return_all=False, debug=True):
    # Load data
    Xkp, flyids, isstart, isdata, scale_perfly = load_data(config, config[filename], debug=debug)

    # Currently I don't support different ways of computing the features, I should assert that the
    # config asks for what I do there

    # Compute features
    track = Data('keypoints', Xkp.T, [])
    sensory = Sensory()(track)
    pose = Pose()(track, scale_perfly=scale_perfly, flyid=flyids)
    pose.array[isdata.T == 0] = np.nan
    sensory.array[isdata.T == 0] = np.nan
    velocity = Velocity()(pose, isstart=isstart)

    tspred_global = config['tspred_global']
    aux_tspred = [dt for dt in tspred_global if dt > 1]
    if len(aux_tspred) > 0:
        auxiliary = GlobalVelocity(tspred=aux_tspred)(pose, isstart=isstart)
    else:
        auxiliary = None

    # Assemble the dataset
    if ref_dataset is None:
        # velocity = OddRoot(5)(velocity)
        bin_config = {'nbins': config['discretize_nbins'], 'bin_epsilon': config['discretize_epsilon']}

        # Need to zscore before binning, otherwise bin_epsilon values need to be divided by zscore stds
        zscored_velocity = Zscore()(velocity)
        discreteidx = config['discreteidx']
        bin_config['bin_epsilon'] /= zscored_velocity.operations[-1].std[discreteidx]
        continuousidx = np.setdiff1d(np.arange(velocity.array.shape[-1]), discreteidx)
        indices_per_op = [discreteidx, continuousidx]
        dataset = Dataset(
            inputs = {
                'velocity': Zscore()(FutureAsInput(dt=1)(velocity)),
                'pose': Zscore()(Subset(featrelative)(pose)),
                'sensory': Zscore()(sensory),
            },
            labels = {
                'velocity': Fusion([Discretize(**bin_config), Identity()], indices_per_op)(zscored_velocity),
                # 'velocity': Fusion([Discretize(**bin_config), Zscore()], indices_per_op)(velocity),
                # 'auxiliary': Discretize()(auxiliary)
            },
            context_length=config['contextl'],
            isstart=isstart,
        )
    else:
        dataset = Dataset(
            inputs = apply_opers_from_data(ref_dataset.inputs, {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
            labels = apply_opers_from_data(ref_dataset.labels, {'velocity': velocity}), #, 'auxiliary': auxiliary}),
            context_length=config['contextl'],
            isstart=isstart,
        )
    if return_all:
        return dataset, flyids, track, pose, velocity, sensory
    else:
        return dataset


class DatasetVars:
    def __init__(self, dataset):
        self.d_input = dataset.d_input
        self.d_output_continuous = dataset.d_output_continuous
        self.d_output_discrete = dataset.d_output_discrete
        self.d_output = self.d_output_continuous + self.d_output_discrete
        self.discretize_nbins = dataset.n_bins

        self.flatten = False
        self.discretize = self.d_output_discrete > 0

        # Collect indices for the inputs
        inds_per_input = {}
        curr_idx = 0
        for key, data in dataset.inputs.items():
            dim = data.array.shape[-1]
            if key == 'sensory':
                for sensory_key, lim in data.operations[0].idxinfo.items():
                    inds_per_input[sensory_key] = [curr_idx + lim[0], curr_idx + lim[1]]
            else:
                inds_per_input[key] = [curr_idx, curr_idx + dim]
            curr_idx += dim
        self.input_idx = OrderedDict([(key, value) for key, value in inds_per_input.items()])
        self.input_szs = OrderedDict([(key, (value[1] - value[0],)) for key, value in inds_per_input.items()])

    def get_input_shapes(self):
        return self.input_idx, self.input_szs


def initialize_model_wrapper(config, dataset, device):
    dataset_vars = DatasetVars(dataset)
    return initialize_model(config, dataset_vars, device)


def experiment():
    # Read config
    # configfile = "/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/config_fly_llm_default.json"
    configfile = "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_20241125.json"
    config = read_config(
        configfile,
        default_configfile=DEFAULTCONFIGFILE,
        posenames=posenames,
        featglobal=featglobal,
        get_sensory_feature_idx=get_sensory_feature_idx,
    )
    # config['discretize_epsilon'] = [0.013222, 0.013496, 0.02598117, 0.02930942, 0.02496748]

    # Make datasets
    train_dataset, flyids, gt_track, gt_pose, gt_velocity = make_dataset(config, 'intrainfile', return_all=True)
    val_dataset = make_dataset(config, 'invalfile', train_dataset)

    # Wrap into dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=False)

    # Initialize the model
    device = torch.device(config['device'])
    model, criterion = initialize_model_wrapper(config, train_dataset, device)

    # Train the model
    train_args = function_args_from_config(config, train)
    train_args['num_train_epochs'] = 100
    model, best_model, loss_epoch = train(train_dataloader, val_dataloader, model, **train_args)

    # Plot the losses
    idx = np.argmin(loss_epoch['val'])
    print((idx, loss_epoch['val'][idx]))
    plt.figure()
    plt.plot(loss_epoch['train'])
    plt.plot(loss_epoch['val'])
    plt.show()
    plt.plot(idx, loss_epoch['val'][idx], '.g')


    # model_file = "/groups/branson/home/eyjolfsdottire/data/flyllm/model_refactored_latestkb_betterbin.pkl"
    # pickle.dump(model, open(model_file, "wb"))
    # model = pickle.load(open(model_file, "rb"))

    agent_id = 0
    pred_track = simulate(
        train_dataset,
        model,
        gt_track,
        gt_pose,
        flyids,
        track_len=1000,
        burn_in=200,
        max_contextl=512,
        agent_id=agent_id,
        start_frame=1000,
    )


    savevidfile = "/groups/branson/home/eyjolfsdottire/data/flyllm/animation_refactor_kristin_agent0_newbin_pred.gif"
    ani = animate_pose({'Pred': pred_track.T.copy(), 'True': gt_track.T.copy()}, focusflies=[agent_id], savevidfile=savevidfile)
