import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import pickle
from torch.utils.data import DataLoader
from collections import OrderedDict

from apf.io import read_config, load_and_filter_data
from apf.dataset import Zscore, Discretize, Data, Dataset, Operation, Fusion, Subset, FutureAsInput, apply_opers_from_data
from apf.training import train
from apf.utils import function_args_from_config, modrange, rotate_2d_points
from apf.models import initialize_model
from apf.data import debug_less_data
from apf.simulation import simulate

from flyllm.config import DEFAULTCONFIGFILE, posenames, featrelative, keypointnames, featangle
from flyllm.features import (
    featglobal, get_sensory_feature_idx, kp2feat, compute_sensory_wrapper, compute_global_velocity,
    compute_relpose_velocity, compute_scale_perfly, compute_noise_params, feat2kp
)
from flyllm.simulation import animate_pose

from experiments.spatial_infomax import set_invalid_ends


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

    # TODO: How to handle scale_perfly and fly_ids, do I need to provide them as args? that would break the interface
    #  it slightly annoying since scale is only used for one of the pose features (not sure if necessary)
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
    # TODO: Implement
    """
        # velocity and it's inverse
        v = delta(x)
        x_ = x[0] + cumsum(v)
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

# Look at:
#   how much memory a dataset takes
#   how much time it takes to load a chunk
#   mmcv config files (pythorch, python)
# Thought:
#   if I wanted to infer multiple timesteps for global pos, and use those as input
#   for the remaining timesteps, would that fit into this pipeline?
#   I think it could... it would be folder into the main task as we will be doing it
#   in the future as well

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


def make_dataset(config, filename, ref_dataset=None, return_all=False):
    # Load data
    Xkp, flyids, isstart, isdata, scale_perfly = load_data(config, config[filename], debug=True)

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
        discreteidx = config['discreteidx']
        continuousidx = np.setdiff1d(np.arange(velocity.array.shape[-1]), discreteidx)
        indices_per_op = [discreteidx, continuousidx]
        dataset = Dataset(
            inputs = {
                'velocity': Zscore()(FutureAsInput(dt=1)(velocity)),
                'pose': Zscore()(Subset(featrelative)(pose)),
                'sensory': Zscore()(sensory),
            },
            labels = {
                'velocity': Fusion([Discretize(**bin_config), Zscore()], indices_per_op)(velocity),
                # 'auxiliary': Discretize()(auxiliary)
            },
            context_length=config['contextl'],
            isstart=isstart,
        )
    else:
        dataset = Dataset(
            inputs = apply_opers_from_data(ref_dataset.inputs, {'sensory': sensory, 'pose': pose, 'velocity': velocity}),
            labels = apply_opers_from_data(ref_dataset.labels, {'velocity': velocity}), #, 'auxiliary': auxiliary}),
            context_length=config['contextl'],
            isstart=isstart,
        )
    if return_all:
        return dataset, flyids, track, pose, velocity
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
    config['discretize_epsilon'] = [0.013222, 0.013496, 0.02598117, 0.02930942, 0.02496748]

    # Make datasets
    train_dataset, flyids, gt_track, gt_pose, gt_velocity = make_dataset(config, 'intrainfile', return_all=True)
    val_dataset = make_dataset(config, 'invalfile', train_dataset)

    # Wrap into dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=False)

    # Initialize the model
    device = torch.device(config['device'])
    # model_args = function_args_from_config(config, TransformerModel)
    # model = init_model(train_dataset, model_args).to(device)
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

    gt_track = simulate(
        train_dataset,
        model,
        gt_track,
        gt_pose,
        flyids,
        track_len=1000,
        burn_in=200,
        max_contextl=512,
        agent_id=0,
        start_frame=1000,
    )


    savevidfile = "/groups/branson/home/eyjolfsdottire/data/flyllm/animation_refactor_kristin_agent0_newbin_pred.gif"
    ani = animate_pose({'Pred': pred_track.T.copy(), 'True': gt_track.T.copy()}, focusflies=[agent_id], savevidfile=savevidfile)

# def compute_features(data, scale_perfly, tspred_global):
#     # TODO: Take as input isdata (that filters out labels that should note be used)
#     # - for example, might filter out female flies but want to use them for computing sensory info
#     # - look at the number of training examples and make sure that matches what we had before
#
#     # Remove all NaN agents (sometimes the last one is a dummy)
#     Xkp = data['X']
#
#     valid = np.sum(~np.isnan(Xkp[0, 0]), axis=-2) > 0
#     Xkp = Xkp[..., valid]
#     flyids = data['ids'][..., valid]
#     isstart = data['isstart'][..., valid]
#
#     # Compute pose features
#     pose = kp2feat(Xkp=Xkp, scale_perfly=scale_perfly, flyid=flyids)
#     relpose = pose[featrelative, ...]
#     globalpos = pose[featglobal, ...]
#
#     # Compute all sensory features
#     n_flies = Xkp.shape[-1]
#     sensory = np.array([compute_sensory_wrapper(Xkp, flyid, theta_main=globalpos[featthetaglobal, :, flyid])[0].T
#                         for flyid in range(n_flies)]).T
#
#     # Compute global movement
#     Xorigin = globalpos[:2, ...]
#     Xtheta = globalpos[2, ...]
#     _, n_frames, n_flies = globalpos.shape
#     dXoriginrel, dtheta = compute_global_velocity(Xorigin, Xtheta, tspred_global)
#     movement_global = np.concatenate((dXoriginrel[:, [1, 0]], dtheta[:, None, :, :]), axis=1)
#     # movement_global = movement_global.reshape((-1, n_frames, n_flies))
#
#     for movement, tspred in zip(movement_global, tspred_global):
#         set_invalid_ends(movement, isstart, dt=tspred)
#
#     # Compute pose velocity
#     tspred_dct = []
#     relpose_rep = compute_relpose_velocity(relpose, tspred_dct)
#     # relpose_rep = compute_relpose_tspred(relpose, tspred_dct, discreteidx=discreteidx)
#     relpose_rep = np.moveaxis(relpose_rep, 2, 0)
#
#     set_invalid_ends(relpose_rep, isstart, dt=1)
#
#     # relpose_rep = relpose_rep.reshape((-1, n_frames, n_flies))
#     relpose_rep = relpose_rep[0]  # we are only using one tspred_dct here (empty, what does that refer to)
#
#     return relpose, globalpos, sensory, movement_global, relpose_rep, Xkp, isstart, flyids
#
#
# def simulate(
#         model,
#         train_dataset,
#         global_vel_out_operation,
#         relpose_vel_out_operation,
#         global_vel_in_operation,
#         relpose_vel_in_operation,
#         relpose_in_operation,
#         sensory_in_operation,
#         Xkp,
#         globalpos,
#         relpose,
#         flyids,
#         scale_perfly,
#         track_len: int = 1000,
#         burn_in: int = 200,
#         max_contextl: int = 512,
#         agent_id: int = 0,
#         start_frame: int = 100
# ):
#     # TODO: Look at the current flyllm.prediction.predict_iterative function in Kristin's branch
#
#     # Extract ground truth
#     gt_chunk = train_dataset.get_chunk(start_frame=start_frame, duration=track_len, agent_id=agent_id)
#     gt_input = gt_chunk['input']
#     gt_track = Xkp[..., start_frame:start_frame + track_len, :]
#     gt_pos = globalpos[..., start_frame:start_frame + track_len, agent_id]
#     gt_pose = relpose[..., start_frame:start_frame + track_len, agent_id]
#     flyid = np.unique(flyids[start_frame:start_frame + track_len, agent_id])
#     assert len(flyid) == 1, f"Too many flyids: {flyid}"
#
#     # Initialize model input
#     device = next(model.parameters()).device
#     model_input = torch.zeros((1, track_len, gt_input.shape[-1]))
#     curr_frame = burn_in
#     model_input[0, :curr_frame, :] = gt_input[:curr_frame]
#     model_input = model_input.to(device)
#
#     # Initialize track
#     track = copy.deepcopy(gt_track)
#     track[:, curr_frame:, agent_id] = np.nan
#     pose = np.concatenate([gt_pos, gt_pose], axis=0)
#     pose[:, curr_frame:] = np.nan
#
#     # Simulate
#     model.eval()
#     while curr_frame < track_len:
#         # Make a motion prediction
#         frame0 = 0
#         if max_contextl is not None:
#             frame0 = max(0, curr_frame - max_contextl)
#
#         # Apply model to previous frames
#         pred = model.output(model_input[:, frame0:curr_frame, :])
#
#         # From prediction to pose velocities
#         pos_vel = global_vel_out_operation.inverse(pred['discrete'][:, -1:].detach().cpu().numpy()).flatten()
#         pose_vel = relpose_vel_out_operation.inverse(pred['continuous'][:, -1:].detach().cpu().numpy()).flatten()
#
#         # From pose velocities to pose
#         curr_pose = pose[:, curr_frame - 1]
#         pose[:3, curr_frame] = apply_motion(*curr_pose[:3], *pos_vel)
#         pose[3:, curr_frame] = curr_pose[3:] + pose_vel
#
#         pose[featangle, curr_frame] = modrange(pose[featangle, curr_frame], -np.pi, np.pi)
#
#         # From pose to kp
#         track[..., curr_frame, agent_id] = feat2kp(pose[:, curr_frame], scale_perfly[:, flyid[0]])[..., 0, 0]
#
#         # From kp to observation
#         sensory = compute_sensory_wrapper(track[:, :, curr_frame:curr_frame + 1, :], agent_id,
#                                           theta_main=pose[2, curr_frame:curr_frame + 1])[0].T
#
#         # Now, wrap everything into the input
#         # input: [global_movement_in, relpose_vel_in, relpose_in, sensory_in]
#         curr_in = [
#             global_vel_in_operation.apply(pos_vel[None, None, :]),
#             relpose_vel_in_operation.apply(pose_vel[None, None, :]),
#             relpose_in_operation.apply(curr_pose[None, None, 3:]),
#             sensory_in_operation.apply(sensory[None, :, :])
#         ]
#         curr_in = np.concatenate(curr_in, axis=-1)
#         model_input[:, curr_frame, :] = torch.from_numpy(curr_in.astype(np.float32)).to(device)
#
#         curr_frame += 1
#
#     return track, gt_track
#
#
# def experiment():
#     """
#     """
#     configfile = "/groups/branson/home/bransonk/behavioranalysis/code/MABe2022/config_fly_llm_default.json"
#     config = read_config(
#         configfile,
#         default_configfile=DEFAULTCONFIGFILE,
#         posenames=posenames,
#         featglobal=featglobal,
#         get_sensory_feature_idx=get_sensory_feature_idx,
#     )
#
#     # Load data
#     data, scale_perfly = load_npz_data(config['intrainfile'], config)
#     valdata, val_scale_perfly = load_npz_data(config['invalfile'], config)
#
#     debug_less_data(data, n_frames_per_video=45000, max_n_videos=2)
#     debug_less_data(valdata, n_frames_per_video=45000, max_n_videos=2)
#
#     # Compute features
#     relpose, globalpos, sensory, movement_global, relpose_rep, Xkp, isstart, flyids = \
#         compute_features(data, scale_perfly, config['tspred_global'])
#     val_relpose, val_globalpos, val_sensory, val_movement_global, val_relpose_rep, val_Xkp, val_isstart, val_flyids = \
#         compute_features(valdata, val_scale_perfly, config['tspred_global'])
#
#     # Wrap into data containers
#
#     # Input data
#     relpose_in = Data(raw=relpose.T, operation=Zscore())
#     val_relpose_in = Data(raw=val_relpose.T, operation=relpose_in.operation)
#
#     sensory_in = Data(raw=sensory.T, operation=Zscore())
#     val_sensory_in = Data(raw=val_sensory.T, operation=sensory_in.operation)
#
#     # Output data
#     # TODO: Make this a dict rather than array
#     global_movement_out = []
#     val_global_movement_out = []
#     tspred_global = config['tspred_global']
#     bin_config = {'nbins': config['discretize_nbins'], 'bin_epsilon': config['discretize_epsilon'][:3]}
#     for movement, val_movement, tspred in zip(movement_global, val_movement_global, tspred_global):
#         global_movement_out.append(Data(raw=movement.T, operation=Discretize(**bin_config)))
#         val_global_movement_out.append(Data(raw=val_movement.T, operation=global_movement_out[-1].operation))
#
#     relpose_vel_out = Data(raw=relpose_rep.T, operation=Zscore())
#     val_relpose_vel_out = Data(raw=val_relpose_rep.T, operation=relpose_vel_out.operation)
#
#     # TODO: Make an operation that can specify which dimensions to discretize and which ones to zscore
#
#     # Motion (output) data as inputs
#     global_movement_in = Data(raw=np.roll(movement_global[0].T, shift=tspred_global[0], axis=1), operation=Zscore())
#     val_global_movement_in = Data(raw=np.roll(val_movement_global[0].T, shift=tspred_global[0], axis=1),
#                                   operation=global_movement_in.operation)
#     relpose_vel_in = Data(raw=np.roll(relpose_rep.T, shift=1, axis=1), operation=Zscore())
#     val_relpose_vel_in = Data(raw=np.roll(val_relpose_rep.T, shift=1, axis=1), operation=relpose_vel_in.operation)
#
#     # Wrap into dataset
#     # TODO: Make inputs and labels a dictionary
#     # TODO: Should also take as input isdata
#     train_dataset = Dataset(
#         inputs=[global_movement_in, relpose_vel_in, relpose_in, sensory_in],
#         labels=[relpose_vel_out] + global_movement_out,  # + [relpose_vel_wings_out],
#         isstart=isstart,
#         context_length=config['contextl'],
#     )
#     val_dataset = Dataset(
#         inputs=[val_global_movement_in, val_relpose_vel_in, val_relpose_in, val_sensory_in],
#         labels=[val_relpose_vel_out] + val_global_movement_out,  # + [val_relpose_vel_wings_out],
#         isstart=val_isstart,
#         context_length=config['contextl'],
#     )
#
#     # Wrap into dataloader
#     device = torch.device(config['device'])
#     train_dataloader = to_dataloader(train_dataset, device=device, batch_size=config['batch_size'], shuffle=True)
#     val_dataloader = to_dataloader(val_dataset, device=device, batch_size=config['batch_size'], shuffle=False)
#
#     # Initialize the model
#     model_args = function_args_from_config(config, TransformerModel)
#     model = init_model(train_dataset, model_args).to(device)
#
#     # Train the model
#     train_args = function_args_from_config(config, train)
#     model, best_model, loss_epoch = train(train_dataloader, val_dataloader, model, **train_args)
#
#     # Plot the losses
#     idx = np.argmin(loss_epoch['val'])
#     print((idx, loss_epoch['val'][idx]))
#     plt.figure()
#     plt.plot(loss_epoch['train'])
#     plt.plot(loss_epoch['val'])
#     plt.plot(idx, loss_epoch['val'][idx], '.g')
#     plt.show()
#
#     # Simulate
#     track, gt_track = simulate(
#         model,
#         train_dataset,
#         global_vel_out_operation=global_movement_out[0].operation,
#         relpose_vel_out_operation=relpose_vel_out.operation,
#         global_vel_in_operation=global_movement_in.operation,
#         relpose_vel_in_operation=relpose_vel_in.operation,
#         relpose_in_operation=relpose_in.operation,
#         sensory_in_operation=sensory_in.operation,
#         Xkp=Xkp,
#         globalpos=globalpos,
#         relpose=relpose,
#         flyids=flyids,
#         scale_perfly=scale_perfly,
#         track_len=1000,
#         burn_in=200,
#         max_contextl=512,
#         agent_id=0,
#         start_frame=100
#     )
#
#     savevidfile = "/groups/branson/home/eyjolfsdottire/data/flyllm/animation.gif"
#     ani = animate_pose({'Pred': track.copy(), 'True': gt_track.copy()}, focusflies=[0], savevidfile=savevidfile)
