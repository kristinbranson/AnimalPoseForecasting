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


def evaluate_model(configfile, modelfile, frames_chunk_for_eval=64, show_figures=False, overwrite_classifiers=False, group=None):

    fly_type = [
        'control', # control 1?
        'BDP_sexseparated', # control sex separated
        'Control_RGB', # control 2
        '71G01', # R71G01
        'male71G01_femaleBDP',
        '65F12', # R65F12
        '91B01', # R91B01
        'BlindControl',
        'aIPgpublished3_newstim', # aIPg
        'pC1dpublished1_newstim', # pC1d
        'aIPgBlind_newstim', # blind aIPg
    ]
    stim_control = [
        'BlindControl_offvson',
        'BlindControl_offvsstrong',
        'BlindControl_offvsweak',
        'BlindControl_weakvsstrong',
        'BlindControl_firstvslast_strong',
        'Control_RGB_offvson',
        'Control_RGB_offvsstrong',
        'Control_RGB_offvsweak',
        'Control_RGB_weakvsstrong',
        'Control_RGB_firstvslast_strong',
    ]
    stim_aggression = [ # TODO: Not sure if these are the correct ones
        'aIPgBlind_newstim_offvson',
        'aIPgBlind_newstim_offvsstrong',
        'aIPgBlind_newstim_offvsweak',
        'aIPgBlind_newstim_weakvsstrong',
        'aIPgBlind_newstim_firstvslast_strong',
        'aIPgpublished3_newstim_offvson',
        'aIPgpublished3_newstim_offvsstrong',
        'aIPgpublished3_newstim_offvsweak',
        'aIPgpublished3_newstim_weakvsstrong',
        'aIPgpublished3_newstim_firstvslast_strong',
        'pC1dpublished1_newstim_offvson',
        'pC1dpublished1_newstim_offvsstrong',
        'pC1dpublished1_newstim_offvsweak',
        'pC1dpublished1_newstim_weakvsstrong',
        'pC1dpublished1_newstim_firstvslast_strong',
    ]
    line_cat = [
        'courtship', # any courtship
        'control_', # any control    # NOTE: Had to rename this in categories since it appear twice
        'blind', # any blind
        'aIPg', # any aIPg
        'aggression', # any agression
        'GMR71G01', # any R71G01
        'sexseparated', # any sexseparated
    ]
    female = ['female']
    manual = [
        'perframe_aggression',
        'perframe_chase',
        'perframe_courtship',
        'perframe_highfence2',
        'perframe_wingext',
        'perframe_wingflick'
    ]
    action_groups = {
        'fly_type': fly_type,
        'stim_control': stim_control,
        'stim_aggression': stim_aggression,
        'line_cat': line_cat,
        'female': female,
        'manual': manual,
    }

    if group is None:
        print("Group is None, training all probes")
    else:
        print(f"Group is: {group}")
        assert group in action_groups, f"Group name is invalid {group}"

    # Verify that simulation and caching has been done for this model
    configname = os.path.split(configfile)[-1].replace('.json', '')
    modelname = os.path.split(modelfile)[-1].replace('.pth', '')

    parentdir = "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/train_data/synthetic_test/"
    sim_savedir = os.path.join(parentdir, f"{configname}_{modelname}")
    assert os.path.exists(sim_savedir), f"data has not been simulated!"

    cache_foldername = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/hidden_cache"
    assert os.path.exists(cache_foldername)
    transformed_paths = {}

    for dataname in ['train', 'test']:
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

    # Load cached data
    d_model = model.d_model
    n_layers = len(model.transformer_encoder.layers)
    n_agents, n_frames = train_dataset.inputs['velocity'].array.shape[:2]
    n_val_agents, n_val_frames = val_dataset.inputs['velocity'].array.shape[:2]
    transformed_data = {}
    for dataname, transformed_path in transformed_paths.items():
        if dataname == 'train':
            num_agents, num_frames = n_agents, n_frames
        else:
            num_agents, num_frames = n_val_agents, n_val_frames
        transformed_data[dataname] = np.memmap(transformed_path, dtype=np.float32, mode='r')
        transformed_data[dataname] = transformed_data[dataname].reshape((n_layers, num_agents, num_frames, d_model))

    # ==== EVALUATING ====

    # Make a results directory
    resultsdir = f"/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/results/{configname}_{modelname}"
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    # first_chunk = frames_chunk_for_eval
    # valid_frames = {
    #     'first': np.logical_and(sim_frame > 0, sim_frame <= first_chunk),
    #     'last': sim_frame > sim_len - first_chunk,
    #     'all': sim_frame > 0,
    # }

    # === Actions ===

    # --- Train actions on hidden states --- (this might take a while)
    train_layers = [5]
    use_layer_id = 5

    labels = train_data['labels']
    labels_val = val_data['labels']
    assert train_data['categories'] == val_data['categories']

    # Set labels of invalid data to nan for easy handling in training

    valid_transform = np.zeros((n_agents, n_frames), bool)
    for agent_id in tqdm(range(n_agents)):
        valid_transform[agent_id] = np.abs(transformed_data['train'][use_layer_id, agent_id]).sum(-1) > 0
    valid_val_transform = np.zeros((n_val_agents, n_val_frames), bool)
    for agent_id in tqdm(range(n_val_agents)):
        valid_val_transform[agent_id] = np.abs(transformed_data['test'][use_layer_id, agent_id]).sum(-1) > 0

    valid_data = np.logical_and(valid_transform.T, train_data['isdata'])
    valid_val_data = np.logical_and(valid_val_transform.T, val_data['isdata'])
    labels[:, ~valid_data] = np.nan
    labels_val[:, ~valid_val_data] = np.nan

    # Pick valid actions
    print('Extracting labels')

    categories = []
    for cat in train_data['categories']:
        if cat in categories:
            print(f'adding underscore to {cat}')
            categories.append(str(cat) + "_")
        else:
            categories.append(str(cat))

    min_labeled = 50
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
            action_counts[cat_name] = {'n_pos': n_pos, 'n_neg': n_neg, 'n_valid': valid_data.sum()}

            n_val_pos = (labels_val[i] == 1).sum()  # ground truth True
            n_val_neg = (labels_val[i] == 0).sum()  # ground truth False
            val_action_counts[cat_name] = {'n_pos': n_val_pos, 'n_neg': n_val_neg, 'n_valid': valid_val_data.sum()}
    action_count_summary = {'train': action_counts, 'val': val_action_counts}
    save_pickle(resultsdir, 'action_count_summary.pkl', action_count_summary)


    if group is None:
        actions = []
        for val in action_groups.values():
            actions.extend(val)
    else:
        actions = action_groups[group]

    # Train classifiers
    print('Training classifiers')

    mlp_layers = ()  # Use a linear classifier

    action_stats = {}
    action_classifiers = {}
    for action in actions:

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
                    transformed_data['train'][layer_id],
                    transformed_data['test'][layer_id],
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
