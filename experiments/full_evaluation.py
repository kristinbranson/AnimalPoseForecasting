import time
import pickle

from apf.evaluation import *
from flyllm.config import ARENA_RADIUS_PX, PXPERMM
from flyllm.config import posenames, keypointnames
from experiments.flyllm import BodyCentricKP
from apf.ml import train_binary_classifier, apply_binary_classifier, classify_real_fake
from apf.models import MLP


def save_and_show_fig(resultsdir, save_name, show_figures):
    plt.savefig(os.path.join(resultsdir, save_name), dpi=150, bbox_inches='tight')
    if show_figures:
        plt.show()
    else:
        plt.close()


def save_pickle(resultsdir, save_name, save_data):
    with open(os.path.join(resultsdir, save_name), 'wb') as f:
        pickle.dump(save_data, f)


def plot_classifier_accuracies(actions, action_stats, train_layers, resultsdir, show_figures):
    best_layer_per_classifier = {}
    for action in actions:
        if not action in action_stats:
            continue
        print(action)

        layer_stats = action_stats[action]

        plt.figure(figsize=(18, 5))
        plt.suptitle(action)

        best_ba = 0
        best_layer = -1
        for i in train_layers:
            stats = layer_stats[i]
            precision, recall, thresholds = stats['PR_curve']
            idx = np.argmin(np.abs(thresholds - 0.5))
            valprecision, valrecall, valthresholds = stats['val_PR_curve']
            validx = np.argmin(np.abs(valthresholds - 0.5))

            plt.subplot(1, 3, 1)
            plt.plot(recall, precision)
            plt.plot(recall[idx], precision[idx], '.k')
            plt.subplot(1, 3, 2)

            f1 = stats['F1s'][-1, 1]
            ba = stats['accuracies'][-1][1]

            if ba > best_ba:
                best_ba = ba
                best_layer = i

            plt.plot(valrecall, valprecision, '.', markersize=2,
                     label=f"layer {i}: F1={f1:.3f}, BA={ba:.3f}")
            plt.plot(valrecall[validx], valprecision[validx], '.k')

            plt.subplot(1, 3, 3)
            plt.plot(stats['losses'], '-o')

        best_layer_per_classifier[action] = best_layer

        plt.subplot(1, 3, 1)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.axis([0, 1, 0, 1])
        plt.title('train')
        plt.subplot(1, 3, 2)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.axis([0, 1, 0, 1])
        plt.title('val')
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.title('training loss')
        plt.xlabel('train iter')

        save_name = f"classifier_stats_action_{action}.png"
        save_and_show_fig(resultsdir, save_name, show_figures)

    return best_layer_per_classifier


def evaluate_model(configfile, modelfile, frames_chunk_for_eval=64, show_figures=False, overwrite_classifiers=False):
    # Verify that simulation and caching has been done for this model
    configname = os.path.split(configfile)[-1].replace('.json', '')
    modelname = os.path.split(modelfile)[-1].replace('.pth', '')

    parentdir = "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/train_data/synthetic"
    sim_savedir = os.path.join(parentdir, f"{configname}_{modelname}")
    assert os.path.exists(sim_savedir), f"data has not been simulated!"

    cache_foldername = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/hidden_cache"
    assert os.path.exists(cache_foldername)
    transformed_paths = {}

    for dataname in ['train_new', 'sim_new', 'val_new']:
        transformed_path = os.path.join(cache_foldername, f"{dataname}__{configname}__{modelname}.npy")
        assert os.path.exists(transformed_path), f"{dataname} has not been cached!"
        transformed_paths[dataname] = transformed_path

    # ==== LOADING ====

    # Load config
    print('Loading config')
    config = read_config(configfile)
    load_config_from_model_file(loadmodelfile=modelfile, config=config, weights_only=False)

    # Swap out test set
    config['invalfilestr'] = config['invalfilestr'].replace('testtrain', 'test2')
    config['invalfile'] = config['invalfile'].replace('testtrain', 'test2')

    # Load dataset
    t0 = time.time()
    print('Loading dataset')
    res = init_datasets(config, needvaldata=True, res={'config': config}, debug_uselessdata=False)
    train_dataset = res['train_dataset']
    train_data = res['train_data']
    contextl = train_dataset.context_length
    val_dataset = res['val_dataset']
    val_data = res['val_data']
    t1 = time.time()
    print(f"loading dataset took {(t1 - t0) / 60} minutes")

    # Load model
    print('Loading model')
    device = torch.device(config['device'])
    model, criterion = initialize_model(config, train_dataset, device)
    _ = load_model(modelfile, model, device)

    # Load simulated data
    print('Loading sim data')
    sim_len = 512
    sim_track, sim_frame, _ = simulate_tracks(
        model, train_dataset, train_data, sim_savedir,
        sim_len=sim_len,
        stride=None,
        simulate_single=False,
        first_sim_frame=512,
        sim_transformed_path=None
    )
    print('Making sim dataset')
    sim_dataset, sim_data = create_dataset_from_ref_data(train_dataset, train_data, sim_track, sim_frame)

    # Load cached data
    d_model = model.d_model
    n_layers = len(model.transformer_encoder.layers)
    n_agents, n_frames = train_dataset.inputs['velocity'].array.shape[:2]
    n_val_agents, n_val_frames = val_dataset.inputs['velocity'].array.shape[:2]
    transformed_data = {}
    for dataname, transformed_path in transformed_paths.items():
        num_agents, num_frames = n_agents, n_frames
        if dataname == 'val_new':
            num_agents, num_frames = n_val_agents, n_val_frames
        transformed_data[dataname] = np.memmap(transformed_path, dtype=np.float32, mode='r',
                                               shape=(n_layers, num_agents, num_frames, d_model))

    # ==== EVALUATING ====

    # Make a results directory
    resultsdir = f"/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/results/{configname}_{modelname}"
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    first_chunk = frames_chunk_for_eval
    valid_frames = {
        'first': np.logical_and(sim_frame > 0, sim_frame <= first_chunk),
        'last': sim_frame > sim_len - first_chunk,
        'all': sim_frame > 0,
    }

    # === Video of simulation ===
    video_name = "simulation_samples"
    if not os.path.exists(os.path.join(resultsdir, f"{video_name}.mp4")):
        make_dataset_video(sim_dataset, sim_data['track'], resultsdir, video_name)
        make_dataset_video(sim_dataset, val_data['track'], resultsdir, video_name.replace('.mp4', '_real.mp4'))

    # # === Feature distribution ===
    # if not os.path.exists(os.path.join(resultsdir, 'all_distributions.pkl')):
    #     # --- MEASURE mean distribution distance (KL divergence or other) ---
    #     print('Plotting feature distributions')
    #     if isinstance(train_data['pose'].operations[0], BodyCentricKP):
    #         bckpnames = []
    #         for name in keypointnames:
    #             # TODO, verify that this is the right order
    #             bckpnames.append(name + '_x')
    #             bckpnames.append(name + '_y')
    #         featnames = posenames[:3] + bckpnames
    #     else:
    #         featnames = posenames
    #
    #     all_distributions = {}
    #     for data_name, use_log in zip(['pose', 'velocity'], [False, True]):
    #         log_str = "log " if use_log else ""
    #
    #         all_frame_distributions = {}
    #         feat_ids = [0, 1, 2, 27, 28] + list(np.arange(3, 27))
    #         for frame_label, frames in valid_frames.items():
    #             distributions = compute_feature_distributions(
    #                 train_feat=train_data[data_name].array[frames],
    #                 sim_feat=sim_data[data_name].array[frames],
    #                 use_log=use_log, n_bins=1000, feat_ids=feat_ids)
    #             kls = [val[-1] for val in distributions.values()]
    #             mean_kl = np.mean(kls)
    #             fig = plot_distributions(
    #                 distributions,
    #                 posenames=featnames,
    #                 title=f'{log_str}{data_name}, {frame_label} frames : mean KL = {mean_kl:.3f}',
    #                 pdf=None, symlog=False)
    #             save_name = f"distribution_{data_name}{frame_label}_{use_log}.png"
    #             save_and_show_fig(resultsdir, save_name, show_figures)
    #             all_frame_distributions[frame_label] = distributions
    #         all_distributions[data_name] = all_frame_distributions
    #
    #     save_pickle(resultsdir, 'valid_frames.pkl', valid_frames)
    #     save_pickle(resultsdir, 'all_distributions.pkl', all_distributions)


    # === Actions ===
    last_output_path = os.path.join(resultsdir, f"action_bouts_walking_sim_new.avi")
    if not os.path.exists(last_output_path):

        # --- Train actions on hidden states --- (this might take a while)
        train_layers = [5]
        use_layer_id = 5

        labels = train_data['labels']
        labels_val = val_data['labels']
        assert train_data['categories'] == val_data['categories']

        # Set labels of invalid data to nan for easy handling in training

        valid_transform = np.zeros((n_agents, n_frames), bool)
        for agent_id in tqdm(range(n_agents)):
            valid_transform[agent_id] = np.abs(transformed_data['train_new'][use_layer_id, agent_id]).sum(-1) > 0
        valid_val_transform = np.zeros((n_val_agents, n_val_frames), bool)
        for agent_id in tqdm(range(n_val_agents)):
            valid_val_transform[agent_id] = np.abs(transformed_data['val_new'][use_layer_id, agent_id]).sum(-1) > 0

        valid_data = np.logical_and(valid_transform.T, train_data['isdata'])
        valid_val_data = np.logical_and(valid_val_transform.T, val_data['isdata'])
        labels[:, ~valid_data] = np.nan
        labels_val[:, ~valid_val_data] = np.nan

        # Pick valid actions
        print('Extracting labels')

        categories = train_data['categories']
        min_labeled = 500
        actions = []
        action_counts = {}
        val_action_counts = {}
        for i, cat_name in enumerate(categories):
            if cat_name == 'BadTracking':
                continue
            n_pos = (labels[i] == 1).sum()  # ground truth True
            n_neg = (labels[i] == 0).sum()  # ground truth False
            if (min_labeled < n_pos) and (min_labeled < n_neg):
                actions.append(cat_name)
                print(f"{cat_name}: pos={n_pos} neg={n_neg}, sum={n_pos + n_neg}")
                action_counts[cat_name]: {'n_pos': n_pos, 'n_neg': n_neg, 'n_valid': valid_data.sum()}

                n_val_pos = (labels_val[i] == 1).sum()  # ground truth True
                n_val_neg = (labels_val[i] == 0).sum()  # ground truth False
                val_action_counts[cat_name] = {'n_pos': n_val_pos, 'n_neg': n_val_neg, 'n_valid': valid_val_data.sum()}
        action_count_summary = {'train': action_counts, 'val': val_action_counts}
        save_pickle(resultsdir, 'action_count_summary.pkl', action_count_summary)


        # Train classifiers
        print('Training classifiers')

        mlp_layers = ()  # Use a linear classifier

        action_stats = {}
        action_classifiers = {}
        for action in actions[::-1]:  # first do the perframe ones which have wayyy fewer labels

            layer_stats = {}
            layer_classifiers = {}

            for layer_id in train_layers:
                print(f'Classifier for {action}')
                classifier_path = f'from_hidden_classifier_{action}_layer{layer_id}.pth'
                classifier_path = os.path.join(resultsdir, classifier_path)
                action_idx = categories.index(action)
                classifier = None
                if os.path.exists(classifier_path) and not overwrite_classifiers:
                    print('- loading from existing')
                    classifier = MLP(input_size=model.d_model, layer_sizes=mlp_layers)
                    checkpoint = torch.load(classifier_path, weights_only=False)
                    classifier.load_state_dict(checkpoint['model_state_dict'])
                    stats = checkpoint['stats']
                    classifier.to(device)
                    classifier.eval()
                else:
                    print('- training')
                    t0 = time.time()
                    classifier, stats = train_binary_classifier(
                        transformed_data['train_new'][layer_id],
                        transformed_data['val_new'][layer_id],
                        labels[action_idx].T.astype(np.float32),
                        labels_val[action_idx].T.astype(np.float32),
                        device, mlp_layers=mlp_layers,
                        min_epochs=5, max_epochs=100,
                    )
                    t1 = time.time()
                    print(f"Classifier trained in {(t1 - t0) / 60} minutes")

                    classifier.eval()
                    checkpoint = {
                        'model_state_dict': classifier.state_dict(),
                        'stats': stats,
                    }
                    torch.save(checkpoint, classifier_path)

                layer_stats[layer_id] = stats
                layer_classifiers[layer_id] = classifier

            action_stats[action] = layer_stats
            action_classifiers[action] = layer_classifiers

        save_pickle(resultsdir, 'action_stats.pkl', action_stats)
        plot_classifier_accuracies(actions, action_stats, train_layers, resultsdir, show_figures)

        # --- MEASURE Label distribution between real and sim (including heuristic walk detector) ---

        # Apply classifiers to real and sim data
        frame_type = 'all'  # first  # by using all, and saving with sim_frames, we can choose only those that we care about for plotting
        classifications = {}
        datanames = ['train_new', 'sim_new']
        for dat_i, name in enumerate(datanames):
            print(f'Running classifiers for {name}')
            flies, frames = np.where(valid_frames[frame_type])

            # # Apply classifiers
            logits = np.ones((len(frames), len(actions) + 1)) * np.nan
            batch_size = 64

            for act_i, action in enumerate(actions):
                print(f" - {action}")
                classif = action_classifiers[action][use_layer_id]
                Xdata = transformed_data[name][use_layer_id]
                logits[:, act_i], prediction, _, _, _ = apply_binary_classifier(
                    classif, Xdata, None, frames, flies, batch_size=batch_size)

            if name == 'train_new':
                track_data = train_data['track']
                sessions = train_dataset.sessions
            else:
                track_data = sim_data['track']
                sessions = sim_dataset.sessions
            print(" - walking")
            walking = detect_walk(track_data, sessions, PXPERMM, ARENA_RADIUS_PX, kp=7, dur_thresh=20, log_vel_thresh=0)
            logits[:, -1] = walking[flies, frames]

            classifications[name] = {'logits': logits, 'frames': frames, 'flies': flies}

        # Extract bouts
        all_actions = actions + ['walking']
        min_dur = 20
        for max_dist_to_prompt in [64, 512]:
            total_actions = len(all_actions)
            bout_summary = np.zeros((2, total_actions), int)
            frame_summary = np.zeros((2, total_actions), int)
            for i, (name, classific) in enumerate(classifications.items()):
                predictions = classific['logits'] > 0.5
                frames = classific['frames']
                flies = classific['flies']

                if max_dist_to_prompt is None:
                    valid = np.ones(len(frames), bool)
                else:
                    valid = sim_frame[flies, frames] <= max_dist_to_prompt

                for action_id in range(total_actions):
                    bouts = get_bouts(predictions[valid], flies[valid], frames[valid], action_id, min_dur=min_dur)
                    bout_summary[i, action_id] = len(bouts[0])
                    frame_summary[i, action_id] = predictions[valid, action_id].sum()

            action_classification_results = {
                'frame_counts': frame_summary,
                'bout_counts': bout_summary,
                'datanames': datanames,
                'actions': all_actions,
                'classifications': classifications,
                'sim_frame': sim_frame,  # this lets us quantify things at different distances from the prompt
            }
            save_pickle(resultsdir, f"action_classification_results_max_dist_{max_dist_to_prompt}.pkl",
                        action_classification_results)

            # Get a visual representation
            plt.figure(figsize=(12, 3))
            plt.suptitle('Real vs. sim classification')
            plt.subplot(1, 2, 1)
            plt.plot(frame_summary[0], '.', label='real', markersize=5)
            plt.plot(frame_summary[1], '.', label='sim', markersize=3)
            for i in range(len(frame_summary[0])):
                plt.plot([i, i], [frame_summary[0][i], frame_summary[1][i]], '-k', linewidth=.1)
            plt.xticks(np.arange(len(all_actions)), all_actions, rotation=90)
            plt.title('frame count')
            plt.subplot(1, 2, 2)
            plt.plot(bout_summary[0], '.', label='real', markersize=5)
            plt.plot(bout_summary[1], '.', label='sim', markersize=3)
            for i in range(len(bout_summary[0])):
                plt.plot([i, i], [bout_summary[0][i], bout_summary[1][i]], '-k', linewidth=.1)
            plt.xticks(np.arange(len(all_actions)), all_actions, rotation=90)
            plt.title('bout count')
            plt.show()
            save_and_show_fig(resultsdir, f"label_stats_max_dist_{max_dist_to_prompt}.png", show_figures)


        # --- FIGURE Locomotion features for certain actions (walk, wing_ext), example bouts
        for action in ['walking', 'perframe_wingext']:
            for data_type in ['train_new', 'sim_new']:
                if data_type == 'train_new':
                    track_data = train_data['track']
                else:
                    track_data = sim_data['track']
                plot_with_features = action == 'walking'
                plot_locomotions(
                    track_data, classifications, all_actions, sim_frame, action, data_type,
                    plot_with_features=plot_with_features
                )
                save_and_show_fig(resultsdir, f"locomotion_{action}_{data_type}.png", show_figures)


        # --- VIDEOS of action bouts between real and sim ---
        video_actions = [name for name in actions if name.startswith('perframe')] + ['walking']
        is_female = labels[categories.index('female')].T
        for action in video_actions:
            for data_type in ['train_new', 'sim_new']:
                if data_type == 'train_new':
                    track_data = train_data['track']
                else:
                    track_data = sim_data['track']
                output_path = os.path.join(resultsdir, f"action_bouts_{action}_{data_type}.avi")
                video_of_action_bouts(track_data, classifications, all_actions, sim_frame,
                                      action, data_type, output_path, is_female, n_samples=10,
                                      min_dist=1, max_dist=None, bg_img=None, overwrite=False)

    # === Perframe discrimination ===
    if not os.path.exists(os.path.join(resultsdir, 'real_sim_discrimination.pkl')):
        print('Training true-vs-sim classifiers')
        results_per_input_type = {}
        layer_types = [[128], [512, 256, 128]]
        for input_type in ['pose', 'velocity']:
            class_results = np.zeros((len(layer_types), len(valid_frames), 2))
            n_epochs = 3
            for j in range(len(layer_types)):
                mlp_layers = layer_types[j]
                for i, (frame_label, frames) in enumerate(valid_frames.items()):
                    train_feat = train_dataset.inputs['velocity'].array[frames]
                    sim_feat = sim_dataset.inputs['velocity'].array[frames]

                    accuracy = classify_real_fake(train_feat, sim_feat, n_epochs=n_epochs, mlp_layers=mlp_layers)[0]
                    control_accuracy = \
                    classify_real_fake(train_feat, sim_feat, n_epochs=n_epochs * 2, control=True, mlp_layers=mlp_layers)[0]
                    class_results[j, i, 0] = accuracy
                    class_results[j, i, 1] = control_accuracy
            results_per_input_type[input_type] = class_results
        real_sim_discrimination = {
            'discr_result': results_per_input_type,
            'first_dim': layer_types,
            'second_dim': list(valid_frames.keys()),
            'third_dim': ['accuracy', 'control accuracy'],
        }
        save_pickle(resultsdir, 'real_sim_discrimination.pkl', real_sim_discrimination)

    # === Multitrack simulation ===
    # --- Simulate multiple trajectories ---

    # --- FIGURE Cluster the multitrajectories, visualize ---

    # -- VIDEOS of trajectories from different clusters


    # --- MEASURE smallest distance to ground truth, OR probability of being close to ground truth ---
    #     to use this metric, we need to simulate from several start points and agents and take the average
    #     and we need to use the same agent and start point
    #     its enough to simlate over 64 x 2 so hopefully it doesn't take too long
    #     I should/could cache the simulation if we want to visualize the different trajectories





import argparse

def __main__(__name__):
    p = argparse.ArgumentParser()
    p.add_argument('--nickname')
    args = p.parse_args()
    print(args.nickname)

    experiments = {}
    nickname = 'ref'
    configfile = "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/flyllm/configs/config_fly_llm_predvel_optimalbinning_20251113.json"
    modelfile = "/groups/branson/home/bransonk/behavioranalysis/code/AnimalPoseForecasting/notebooks/flyllm_models/flypredvel_20251007_20251114T194024_bestepoch200.pth"
    experiments[nickname] = [configfile, modelfile]

    nickname = 'short'
    configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_optimalbinning_short_20260430.json"
    modelfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/flyllm_models/flypredvel_optimal_binning_64_20260502T050431_epoch235.pth"
    experiments[nickname] = [configfile, modelfile]

    nickname = 'nobin'
    configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_nobinning_20260430.json"
    modelfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/flyllm_models/flypredvel_nobinning_20260501T050753_epoch300.pth"
    experiments[nickname] = [configfile, modelfile]

    nickname = 'bodycentric'
    configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_optimalbinning_bodycentrickp_20260430.json"
    modelfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/flyllm_models/flypredvel_body_centric_kp_20260501T060729_epoch295.pth"
    experiments[nickname] = [configfile, modelfile]

    nickname = 'predpose'
    configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_optimalbinning_globalvel_20260430.json"
    modelfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/flyllm_models/flypredvelandpose_20260501T070152_epoch295.pth"
    experiments[nickname] = [configfile, modelfile]

    configfile, modelfile = experiments[args.nickname]
    evaluate_model(configfile, modelfile)
