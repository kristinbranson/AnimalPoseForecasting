import time
import os
from sklearn.metrics import precision_recall_curve

from apf.evaluation import *
from flyllm.config import ARENA_RADIUS_PX, PXPERMM
from apf.ml import apply_binary_classifier
from apf.models import MLP
from experiments.full_evaluation import save_pickle, save_and_show_fig


def evaluate_model(configfile, modelfile, frames_chunk_for_eval=64, show_figures=False, overwrite_classifiers=False):
    # Verify that simulation and caching has been done for this model
    configname = os.path.split(configfile)[-1].replace('.json', '')
    modelname = os.path.split(modelfile)[-1].replace('.pth', '')

    parentdir = "/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/train_data/synthetic_test"
    sim_savedir = os.path.join(parentdir, f"{configname}_{modelname}")
    assert os.path.exists(sim_savedir), f"data has not been simulated!"

    cache_foldername = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/hidden_cache"
    assert os.path.exists(cache_foldername)
    transformed_paths = {}

    for dataname in ['test', 'sim_test']:
        transformed_path = os.path.join(cache_foldername, f"{dataname}__{configname}__{modelname}.npy")
        assert os.path.exists(transformed_path), f"{dataname} has not been cached!"
        transformed_paths[dataname] = transformed_path

    # ==== LOADING ====

    # Load config
    print('Loading config')
    config = read_config(configfile)
    load_config_from_model_file(loadmodelfile=modelfile, config=config, weights_only=False)

    # Replace validation with test2
    config['invalfilestr'] = config['invalfilestr'].replace('testtrain', 'test2')
    config['invalfile'] = config['invalfile'].replace('testtrain', 'test2')

    # Load dataset
    t0 = time.time()
    print('Loading dataset')
    res = init_datasets(config, needvaldata=True, res={'config': config}, debug_uselessdata=False)
    train_dataset = res['train_dataset']
    train_data = res['train_data']
    contextl = train_dataset.context_length
    test_dataset = res['val_dataset']
    test_data = res['val_data']
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
    sim_track, sim_frame = load_sim_track(test_data['track'], sim_savedir, contextl)
    # For this analysis, only consider frames where all flies are valid
    intersect_sim_path = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/intersect_valid_test_frames.npy"
    if os.path.exists(intersect_sim_path):
        valid_frames = np.load(intersect_sim_path, allow_pickle=True).item()
        n_sim_flies = sim_frame.shape[0]
        n_valid_flies = valid_frames['all'].shape[0]
        if n_sim_flies != n_valid_flies:
            print(f"Mismatch between number of sim flies and valid flies: {n_sim_flies} vs {n_valid_flies}")
        sim_frame[valid_frames['all'][:n_sim_flies, :] == 0] = 0
    print('Making sim dataset')
    sim_dataset, sim_data = create_dataset_from_ref_data(test_dataset, test_data, sim_track, sim_frame)


    # Load cached data
    d_model = model.d_model
    n_layers = len(model.transformer_encoder.layers)
    # num_agents, num_frames = train_dataset.inputs['velocity'].array.shape[:2]
    num_test_agents, num_test_frames = test_dataset.inputs['velocity'].array.shape[:2]
    transformed_data = {}
    for dataname, transformed_path in transformed_paths.items():
        transformed = np.memmap(transformed_path, dtype=np.float32, mode='r').reshape((n_layers, -1, num_test_frames, d_model))
        n_transf_agents = transformed.shape[1]
        if n_transf_agents > num_test_agents:
            print("Removing extra agents from cached data")
            transformed = transformed[:, :num_test_agents]
        transformed_data[dataname] = transformed

    # ==== EVALUATING ====
    # Make a results directory
    resultsdir = f"/groups/branson/home/eyjolfsdottire/AnimalPoseForecastingData/results_test/{configname}_{modelname}"
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    first_chunk = frames_chunk_for_eval
    valid_frames = {
        'first': np.logical_and(sim_frame > 0, sim_frame <= first_chunk),
        'last': sim_frame > sim_len - first_chunk,
        'all': sim_frame > 0,
    }

    # --- Visualize simulation data ---
    video_name = "simulation_samples"
    if not os.path.exists(os.path.join(resultsdir, f"{video_name}.mp4")):
        make_dataset_video(sim_dataset, sim_data['track'], resultsdir, video_name)
        make_dataset_video(sim_dataset, test_data['track'], resultsdir, video_name)


    # --- Probe test ---
    train_layers = [5]
    use_layer_id = 5

    labels = train_data['labels']
    labels_test = test_data['labels']
    assert train_data['categories'] == test_data['categories']

    # Set labels of invalid data to nan for easy handling in training

    valid_test_transform = np.zeros((num_test_agents, num_test_frames), bool)
    for agent_id in tqdm(range(num_test_agents)):
        valid_test_transform[agent_id] = np.abs(transformed_data['test'][use_layer_id, agent_id]).sum(-1) > 0

    valid_test_data = np.logical_and(valid_test_transform.T, test_data['isdata'])
    labels_test[:, ~valid_test_data] = np.nan

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
            n_val_pos = (labels_test[i] == 1).sum()  # ground truth True
            n_val_neg = (labels_test[i] == 0).sum()  # ground truth False
            print(f"{cat_name}: pos={n_pos} neg={n_neg}, sum={n_pos + n_neg}, val_pos={n_val_pos}, val_neg={n_val_neg}")
            val_action_counts[cat_name] = {'n_pos': n_val_pos, 'n_neg': n_val_neg, 'n_valid': valid_test_data.sum()}
    action_count_summary = {'test': val_action_counts}
    save_pickle(resultsdir, 'action_count_summary.pkl', action_count_summary)


    # Train classifiers
    print('Applying probe classifiers')

    mlp_layers = ()  # Use a linear classifier

    action_stats = {}
    action_classifiers = {}
    for action in actions[::-1]:  # first do the perframe ones which have wayyy fewer labels

        layer_stats = {}
        layer_classifiers = {}

        for layer_id in train_layers:
            print(f'Classifier for {action}')
            classifier_path = f'from_hidden_classifier_{action}_layer{layer_id}.pth'
            classifier_path = os.path.join(resultsdir.replace('results_test', 'results'), classifier_path)
            action_idx = categories.index(action)
            if not os.path.exists(classifier_path):
                print(f'WARNING: Classifier {classifier_path} does not exists, skipping!')
                continue
            print('- loading from existing')
            classifier = MLP(input_size=model.d_model, layer_sizes=mlp_layers)
            checkpoint = torch.load(classifier_path, weights_only=False)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            stats = checkpoint['stats']
            classifier.to(device)
            classifier.eval()

            # Apply classifier to test data
            test_transf_data = transformed_data['test'][layer_id]
            test_label = labels_test[action_idx].T.astype(np.float32)
            if test_transf_data.shape[0] < test_label.shape[0]:
                test_label = test_label[:test_transf_data.shape[0]]
            valid = ~np.isnan(test_label)
            test_agents, test_frames = np.where(valid)
            testlogits, testpredictions, test_accuracy, test_f1, test_stats = apply_binary_classifier(
                classifier, test_transf_data, test_label, frames=test_frames, agents=test_agents, batch_size=128)

            test_PR_curve = precision_recall_curve(test_label[test_agents, test_frames], testlogits, drop_intermediate=True)
            stats = {
                'balanced_accuracy': test_accuracy,
                'f1': test_f1,
                'PR_curve': test_PR_curve,
                'test_stats': test_stats,
            }
            layer_stats[layer_id] = stats
            layer_classifiers[layer_id] = classifier

        action_stats[action] = layer_stats
        action_classifiers[action] = layer_classifiers

    save_pickle(resultsdir, 'test_action_stats.pkl', action_stats)
