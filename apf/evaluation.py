
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from sklearn.metrics import precision_recall_curve

from apf.io import load_config_from_model_file, load_model
from apf.simulation import simulate
from apf.dataset import Dataset, apply_opers_from_data
from apf.models import initialize_model
from experiments.flyllm import Sensory, Pose, Velocity
from flyllm.prepare import init_datasets, init_raw_data
from flyllm.config import read_config, featrelative, featangle, posenames


class MLP(nn.Module):
    def __init__(self, input_size, layer_size=(64, 32)):
        super(MLP, self).__init__()
        if len(layer_size) == 0:
            self.layers = []
            self.last_layer = nn.Linear(input_size, 1)
        else:
            layers = [nn.Linear(input_size, layer_size[0])]
            for i in range(len(layer_size) - 1):
                layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            self.layers = torch.nn.ModuleList(layers)
            self.last_layer = nn.Linear(layer_size[-1], 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return self.sigmoid(self.last_layer(x))


class ComboModel(torch.nn.Module):
    def __init__(self, pretrained_model, mlp_layers, device, finetune=True):
        super().__init__()
        self.mlp = MLP(input_size=pretrained_model.d_model, layer_size=mlp_layers).to(device)
        self.finetune = finetune
        if not finetune:
            pretrained_model.eval()
            for param in pretrained_model.parameters():
                param.requires_grad = False
        self.pretrained_model = pretrained_model

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune:
            self.pretrained_model.eval()

    def forward(self, x):
        if self.finetune:
            return self.mlp(self.pretrained_model.forward(x, return_hidden=True)['hidden'][:, -1])
        else:
            with torch.no_grad():
                hidden = self.pretrained_model.forward(x, return_hidden=True)['hidden'][:, -1]
            return self.mlp(hidden)


def eval_simulation(configfile, modelfile, savedir, context_multiple, stride=None):
    # Load config
    print('Loading config')
    config = read_config(configfile)
    load_config_from_model_file(loadmodelfile=modelfile, config=config)

    # Load dataset
    print('Loading dataset')
    res = init_datasets(config, needvaldata=True, res={'config': config})
    train_dataset = res['train_dataset']
    train_data = res['train_data']
    contextl = train_dataset.context_length
    val_dataset = res['val_dataset']
    val_data = res['val_data']

    # Load raw data, to get the action labels
    raw_res = init_raw_data(config, needtraindata=True, needvaldata=True)
    raw_data = raw_res['data']
    raw_val_data = raw_res['valdata']
    raw_data['y'][:, ~raw_data['isdata']] = np.nan
    raw_val_data['y'][:, ~raw_val_data['isdata']] = np.nan
    assert (raw_data['X'].shape[-2] == train_dataset.inputs['velocity'].array.shape[1])

    # Load model
    device = torch.device(config['device'])
    model, criterion = initialize_model(config, train_dataset, device)
    _ = load_model(modelfile, model, device)

    # Run/load the simulation and create a dataset from it
    print('Loading simulated data')
    track_len = contextl * context_multiple
    sim_track, sim_frame = simulate_tracks(model, train_dataset, train_data, savedir, track_len, stride)
    print('Creating simulated dataset')
    sim_dataset, sim_data = create_dataset_from_ref_data(train_dataset, train_data, sim_track, sim_frame)

    # Create and save hidden states for train, val, and sim datasets
    print('Computing and caching hidden states')
    foldername = 'hidden_cache'
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    configname = os.path.split(configfile)[-1].replace('.json', '')
    modelname = os.path.split(modelfile)[-1].replace('.pth', '')
    transformed_data = {}
    for dataname, dataset in zip(['train', 'val', 'sim'], [train_dataset, val_dataset, sim_dataset]):
        print(f'-- {dataname}')
        transformed_path = os.path.join(foldername, f"{dataname}__{configname}__{modelname}.npy")
        transformed_data[dataname] = transform_dataset(dataset, model, transformed_path, batch_size=128)

    # ====== QUANTITATIVE EVALUATION ========
    pdf = PdfPages(os.path.join(savedir, f"{modelname}_report.pdf"))

    # Get frame chunks of simulated frames
    frame_chunks = [np.logical_and(sim_frame > contextl * i, sim_frame <= contextl * (i + 1))
                    for i in range(1, context_multiple - 1)]
    valid_frames = {
        'first': frame_chunks[0],
        'last': frame_chunks[-1],
        'all': np.logical_and(sim_frame > train_dataset.context_length, sim_frame <= track_len)
    }

    # FEATURE DISTRIBUTIONS
    print('Plotting feature distributions')
    for frame_label, frames in valid_frames.items():
        plot_feature_distributions(train_data, sim_data, frames, frame_label, data_name='pose', use_log=False, pdf=pdf)
        plot_feature_distributions(train_data, sim_data, frames, frame_label, data_name='velocity', use_log=True, pdf=pdf)

    # ACTION DISTRIBUTIONS
    classifiers = {}
    try:
        print('Training per-frame action classifiers')
        classifications = {}

        actions = [name for name in raw_data['categories']]
        # actions = [name for name in raw_data['categories'] if name.startswith('perframe_')]
        frame_counts = np.zeros((len(actions), 2))
        for i in range(len(actions)):
            idx = raw_data['categories'].index(actions[i])
            frame_counts[i, 0] = (raw_data['y'][idx] == 1).sum()
            frame_counts[i, 1] = (raw_data['y'][idx] == 0).sum()
        _, n_frames, n_flies = raw_data['y'].shape
        total_frames = n_frames * n_flies
        actions = [actions[i] for i in range(len(actions)) if 500 < frame_counts[i, 0] < total_frames]
        # actions = [actions[i] for i in range(len(actions)) if frame_counts[i].sum() > 1000]

        print(actions)
        if len(actions) == 0:
            tmp = 0/1

        # Train classifiers
        overwrite = True
        for action in actions:
            print(f'Classifier for {action}')
            classifier_path = f'from_hidden_classifier_{action}.pth'
            action_idx = raw_data['categories'].index(action)
            val_action_idx = raw_val_data['categories'].index(action)
            if os.path.exists(classifier_path) and not overwrite:
                print('- loading from existing')
                classifier = MLP(input_size=model.d_model, layer_size=[128])
                # classifier.load_state_dict(torch.load(classifier_path, weights_only=True))
                checkpoint = torch.load(classifier_path, weights_only=False)
                classifier.load_state_dict(checkpoint['model_state_dict'])
                classifier.stats = checkpoint['stats']
                classifier.eval()
            else:
                print('- training')
                classifier = train_from_hidden(transformed_data['train'], transformed_data['val'],
                                               raw_data['y'][action_idx].T.astype(np.float32),
                                               raw_val_data['y'][val_action_idx].T.astype(np.float32),
                                               device, pdf, action)
                classifier.eval()
                # torch.save(classifier.state_dict(), classifier_path)
                checkpoint = {
                    'model_state_dict': classifier.state_dict(),
                    'stats': classifier.stats,
                }
                torch.save(checkpoint, classifier_path)
            classifiers[action] = classifier

        # Apply classifiers to real and sim data
        summarytable = np.zeros((2, len(actions)))
        for dat_i, (name, dataset) in enumerate({'train': train_dataset, 'sim': sim_dataset}.items()):
            print(f'Running classifiers for {name}')
            flies, frames = np.where(valid_frames['all'])

            # Apply classifiers
            logits = np.ones((len(frames), len(actions))) * np.nan
            batch_size = 64
            for b in tqdm(range(0, len(frames), batch_size)):
                batch_hidden = torch.from_numpy(transformed_data[name][flies[b:b+batch_size], frames[b:b+batch_size]]).to(device)
                for act_i, action in enumerate(actions):
                   logits[b:b+batch_size, act_i] = classifiers[action](batch_hidden).detach().cpu().numpy()[:, 0]

            summarytable[dat_i] = (logits > 0.5).sum(0)
            classifications[name] = {'logits': logits, 'frames': frames, 'flies': flies}
        plot_array_as_table(
            data=summarytable.T,
            row_labels=actions,
            col_labels=['real', 'sim'],
            title='Real vs sim label statistics', pdf=pdf
        )

        fig = plt.figure(figsize=(8, 4))
        fractions = summarytable / logits.shape[0]
        for i in range(len(actions)):
            plt.plot([i, i], fractions[:, i], '--k', linewidth=1)
        plt.plot(fractions[0, :], '.', markersize=10, label='real')
        plt.plot(fractions[1, :], '.', markersize=8, label='sim')
        plt.xticks(np.arange(len(actions)), actions, rotation=90)
        plt.ylabel('fraction of frames')
        plt.title('Perframe action classification')
        plt.legend()
        plt.ylim([0, 1])
        fig.tight_layout()
        if pdf is None:
            plt.show()
        else:
            pdf.savefig(fig)
            plt.close()

        tmp = 1/0

    except Exception as e:
        print(e)
        return train_data, val_data, sim_data, train_dataset, val_dataset, sim_dataset, transformed_data, valid_frames, sim_frame, raw_data, raw_val_data, classifiers, pdf, e

    # TRUE-vs-SIM CLASSIFICATION ACCURACY
    print('Training true-vs-sim classifiers')
    try:
        class_results = np.zeros((len(valid_frames), 2))
        n_epochs = 3
        for i, (frame_label, frames) in enumerate(valid_frames.items()):
            train_feat = train_dataset.inputs['velocity'].array[frames]
            sim_feat = sim_dataset.inputs['velocity'].array[frames]

            # print(f"Frames used: {frame_label}")
            accuracy = classify_real_fake(train_feat, sim_feat, n_epochs=n_epochs)
            control_accuracy = classify_real_fake(train_feat, sim_feat, n_epochs=n_epochs*2, control=True)
            class_results[i, 0] = accuracy
            class_results[i, 1] = control_accuracy
        plot_array_as_table(
            data=class_results,
            row_labels=list(valid_frames.keys()),
            col_labels=['accuracy', 'control accuracy'],
            title='Real vs fake classification accuracy', pdf=pdf
        )
        pdf.close()
    except Exception as e:
        pdf.close()

    # ====== QUALITATIVE ========
    # MULTI TRAJECTORIES (CLUSTERED BY WHERE IT ENDS UP)

    # PATTERNS IN AGENTS FRAME, DURING CERTAIN ACTIONS (LIKE WALKING)


    return train_data, sim_data, train_dataset, sim_dataset, valid_frames, sim_frame, raw_data, classifications


def apply_model_to_hidden(model, Xdata, ydata, frames, flies, batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    logits = np.zeros(len(frames))
    for i in range(0, len(frames), batch_size):
        chunk = torch.from_numpy(Xdata[flies[i:i+batch_size], frames[i:i+batch_size]]).to(device)
        logits[i:i + batch_size] = model(chunk).detach().cpu().numpy()[:, 0]
    chunk = torch.from_numpy(Xdata[flies[i + batch_size:], frames[i + batch_size:]]).to(device)
    logits[i + batch_size:] = model(chunk).detach().cpu().numpy()[:, 0]
    predictions = (logits > 0.5).astype(float)
    if ydata is None:
        accuracy = None
    else:
        accuracy = (predictions == ydata[flies, frames]).sum() / len(frames)
    return logits, predictions, accuracy


def apply_model(model, Xdata, ydata=None, batch_size=64):
    model.eval()
    logits = np.zeros(Xdata.shape[0])
    torchify = isinstance(Xdata, np.ndarray)
    for i in range(0, Xdata.shape[0], batch_size):
        if torchify:
            chunk = torch.from_numpy(Xdata[i:i + batch_size])
        else:
            chunk = Xdata[i:i + batch_size]
        logits[i:i + batch_size] = model(chunk).detach().cpu().numpy()[:, 0]
    if torchify:
        chunk = torch.from_numpy(Xdata[i + batch_size:])
    else:
        chunk = Xdata[i + batch_size:]
    logits[i + batch_size:] = model(chunk).detach().cpu().numpy()[:, 0]
    predictions = (logits > 0.5).astype(float)
    if ydata is None:
        accuracy = None
    else:
        accuracy = (predictions == ydata.detach().cpu().numpy()[:, 0]).sum() / ydata.size(0)
    return logits, predictions, accuracy


def transform_dataset(dataset, transformer, transformed_path, batch_size=64, overwrite=False):
    # It takes about 3 hours to process 10M data points.
    device = next(transformer.parameters()).device
    transformer.eval()

    # Initialize transformed matrix (load it if exists)
    d_model = transformer.d_model
    d_input = dataset.d_input
    n_agents, n_frames = dataset.inputs['velocity'].array.shape[:2]
    contextl = dataset.context_length
    mode = 'r+' if os.path.exists(transformed_path) else 'w+'
    transformed = np.memmap(transformed_path, dtype=np.float32, mode=mode, shape=(n_agents, n_frames, d_model))

    # Collect valid frames from dataset sessions
    frames = []
    agents = []
    for session in dataset.sessions:
        sframes = np.arange(session.start_frame + contextl - 1, session.start_frame + session.duration)
        sagents = np.ones(len(sframes), int) * session.agent_id
        frames.append(sframes)
        agents.append(sagents)
    frames = np.concatenate(frames)
    agents = np.concatenate(agents)

    # Process data in batches
    batch_inputs = np.zeros((batch_size, contextl, d_input), np.float32)
    batch_indices = np.zeros((2, batch_size), int)
    batch_idx = 0
    for i in tqdm(range(len(frames))):
        frame_id = frames[i]
        agent_id = agents[i]
        if overwrite or np.abs(transformed[agent_id, frame_id]).sum() > 0:
            # Skip frames that have already been processed
            continue
        batch_inputs[batch_idx] = dataset.get_chunk(
            start_frame=frame_id - contextl + 1, duration=contextl, agent_id=agent_id
        )['input']
        batch_indices[0, batch_idx] = agent_id
        batch_indices[1, batch_idx] = frame_id
        batch_idx += 1
        if batch_idx == batch_size:
            out = transformer.forward(torch.from_numpy(batch_inputs).to(device), return_hidden=True)['hidden']
            bagents, bframes = batch_indices
            transformed[bagents, bframes] = out[:, -1].detach().cpu().numpy()
            batch_idx = 0
    return transformed


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transformer, frames, flies, parent_dir, batch_size, labels=None):
        self.dataset = dataset
        self.transformer = transformer
        self.frames = frames
        self.flies = flies
        self.parent_dir = parent_dir
        self.batch_size = batch_size
        self.start_frames = np.arange(0, len(frames), batch_size)
        self.device = next(transformer.parameters()).device
        self.contextl = dataset.context_length
        self.labels = labels

    def __getitem__(self, i):
        start_frame = self.start_frames[i]
        path = os.path.join(self.parent_dir, f"transfomer_hidden_batch{self.batch_size}_{start_frame}.npy")
        if os.path.exists(path):
            batch_hidden = torch.from_numpy(np.load(path)).to(self.device)
        else:
            batch = np.zeros((self.batch_size, self.contextl, self.dataset.d_input), np.float32)
            # TODO: need to verify that a chunk always belongs to the same fly
            # I think it would be better just to use the chunks given in the dataset
            for i in range(self.batch_size):
                batch[i] = self.dataset.get_chunk(
                    start_frame=self.frames[start_frame + i] - self.contextl + 1,
                    duration=self.contextl,
                    agent_id=self.flies[start_frame + i]
                )['input']
            batch = torch.from_numpy(batch).to(self.device)
            batch_hidden = self.transformer.forward(batch, return_hidden=True)['hidden'][:, -1]
            np.save(path, batch_hidden.detach().cpu().numpy())
        if self.labels is not None:
            batch_labels = self.labels[start_frame:start_frame + self.batch_size]
            return batch_hidden, torch.from_numpy(batch_labels).to(self.device)
        return batch_hidden, None

    def __len__(self):
        return len(self.start_frames)


def train_from_hidden(train_data, val_data, train_label, val_label, device, pdf, action):
    if train_data.shape[0] < train_label.shape[0]:
        train_label = train_label[:train_data.shape[0]]
    if val_data.shape[0] < val_label.shape[0]:
         val_label = val_label[:val_data.shape[0]]

    valid = np.logical_and(~np.isnan(train_data.sum(-1)), ~np.isnan(train_label))
    flies, frames = np.where(valid)

    valid = np.logical_and(~np.isnan(val_data.sum(-1)), ~np.isnan(val_label))
    val_flies, val_frames = np.where(valid)
    
    pos_train = np.where(train_label[flies, frames] == 1)[0]
    neg_train = np.where(train_label[flies, frames] == 0)[0]
    n_pos = len(pos_train)
    n_neg = len(neg_train)
    n_min = min(n_pos, n_neg)
    n_frames = n_min * 2

    batch_size = 32
    if n_frames < 5000:
        batch_size = 8
    if n_frames > 100000:
        batch_size = 128
    n_batch_iter = 25000
    n_batches = n_frames // batch_size
    n_epochs = n_batch_iter // n_batches
    n_epochs = int(np.clip(n_epochs, 2, 50))
    half_batch_size = batch_size//2

    print(f"batch_size={batch_size}, n_epochs={n_epochs}, n_pos={n_pos}, n_neg={n_neg}, n_min={n_min}")

    # Initialize model, loss, optimizer
    lr = 0.0001
    mlp_layers = [128]
    classifier = MLP(train_data.shape[-1], mlp_layers).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # Training loop
    losses = np.zeros(n_epochs)
    accuracies = np.zeros((n_epochs, 2))
    for epoch in tqdm(range(n_epochs)):
        classifier.train()
        epoch_loss = 0
        # perm_inds = np.random.permutation(len(frames))
        perm_inds_pos = np.random.permutation(n_pos)
        perm_inds_neg = np.random.permutation(n_neg)
        for i in tqdm(range(0, n_min, half_batch_size), leave=False):
            optimizer.zero_grad()
            # inds = perm_inds[i:i+batch_size]
            pos_inds = pos_train[perm_inds_pos[i:i+half_batch_size]]
            neg_inds = neg_train[perm_inds_neg[i:i+half_batch_size]]
            inds = np.concatenate([pos_inds, neg_inds])

            batch_X = torch.from_numpy(train_data[flies[inds], frames[inds]]).to(device)
            batch_y = torch.from_numpy(train_label[flies[inds], frames[inds]][:, None]).to(device)
            outputs = classifier(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses[epoch] = avg_loss

        # Compute results
        logits, predictions, train_accuracy = apply_model_to_hidden(
            classifier, train_data, train_label, frames=frames, flies=flies, batch_size=128)
        testlogits, testpredictions, test_accuracy = apply_model_to_hidden(
            classifier, val_data, val_label, frames=val_frames, flies=val_flies, batch_size=128)

        accuracies[epoch, 0] = train_accuracy
        accuracies[epoch, 1] = test_accuracy

    precision, recall, _ = precision_recall_curve(train_label[flies, frames], logits, drop_intermediate=True)
    valprecision, valrecall, _ = precision_recall_curve(val_label[val_flies, val_frames], testlogits, drop_intermediate=True)
    stats = {
        'losses': losses,
        'accuracies': accuracies,
        'PR': np.vstack([precision, recall]),
        'valPR': np.vstack([valprecision, valrecall])
    }
    classifier.stats = stats

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'{action}', fontsize=16, fontweight='bold')
    plt.subplot(1,3, 1)
    plt.plot(losses)
    plt.title('Losses')
    plt.subplot(1, 3, 2)
    plt.plot(accuracies)
    plt.title('Accuracies')
    plt.subplot(1, 3, 3)
    plt.plot(precision, recall)
    plt.plot(valprecision, valrecall)
    fig.tight_layout()
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
        plt.close()

    return classifier


def train_action_classifier(action, raw_data, train_dataset, transformer, device, pdf=None):
    n_epochs = 25
    lr = 0.0001
    mlp_layers = [128]
    batch_size = 32
    finetune = False
    frac_train = 0.8
    normalize = False

    action_idx = raw_data['categories'].index(action)
    frames, flies = np.where(~np.isnan(raw_data['y'][action_idx]))
    label = raw_data['y'][action_idx][frames, flies]

    # Collect inputs per frame to be classified
    contextl = train_dataset.context_length
    inputs = np.zeros((len(frames), contextl, train_dataset.d_input))
    for i in range(len(frames)):
        inputs[i] = train_dataset.get_chunk(
            start_frame=frames[i] - contextl,
            duration=contextl,
            agent_id=flies[i]
        )['input']
    valid = [np.isnan(chunk).sum() == 0 for chunk in inputs]
    frames = frames[valid]
    flies = flies[valid]
    label = label[valid]
    inputs = inputs[valid]

    if len(frames) < 5000:
        batch_size = 8

    # Split labeled data into bouts
    framediff = np.ones(len(frames)) * 2
    flydiff = np.ones(len(flies))
    labeldiff = np.ones(len(label))
    framediff[1:] = np.diff(frames)
    flydiff[1:] = np.diff(flies)
    labeldiff[1:] = np.diff(label)
    first_frame = np.where((framediff > 1) + (np.abs(flydiff) > 0) + (np.abs(labeldiff) > 0))[0]
    bouts = [[first_frame[i], first_frame[i + 1] - 1] for i in range(len(first_frame) - 1)]
    bouts.append([first_frame[-1], len(frames) - 1])

    # Split bouts into train and test
    n_pos = (label == 1).sum()
    n_neg = (label == 0).sum()
    n_pos_train = int(np.round(n_pos * frac_train))
    n_neg_train = int(np.round(n_neg * frac_train))
    train_inds = []
    test_inds = []
    c_train_pos = 0
    c_train_neg = 0
    for (start_i, end_i) in bouts:
        if label[start_i] == 1:
            if c_train_pos < n_pos_train:
                train_inds.append(np.arange(start_i, end_i + 1))
                c_train_pos += (end_i - start_i + 1)
            else:
                test_inds.append(np.arange(start_i, end_i + 1))
        else:
            if c_train_neg < n_neg_train:
                train_inds.append(np.arange(start_i, end_i + 1))
                c_train_neg += (end_i - start_i + 1)
            else:
                test_inds.append(np.arange(start_i, end_i + 1))
    train_inds = np.concatenate(train_inds)
    test_inds = np.concatenate(test_inds)

    X_train = inputs[train_inds]
    X_test = inputs[test_inds]
    y_train = label[train_inds]
    y_test = label[test_inds]

    # Normalize features
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    # Initialize model, loss, optimizer
    pretrained = copy.deepcopy(transformer)
    classifier = ComboModel(pretrained, mlp_layers=mlp_layers, device=device, finetune=finetune)
    criterion = torch.nn.BCELoss()
    opt_params = [param for param in classifier.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(opt_params, lr=lr)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # Training loop
    losses = np.zeros(n_epochs)
    accuracies = np.zeros((n_epochs, 2))
    for epoch in tqdm(range(n_epochs)):
        classifier.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            outputs = classifier(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses[epoch] = avg_loss

        # Compute results
        logits, predictions, train_accuracy = apply_model(classifier, X_train, y_train, 64)
        testlogits, testpredictions, test_accuracy = apply_model(classifier, X_test, y_test, 64)

        accuracies[epoch, 0] = train_accuracy
        accuracies[epoch, 1] = test_accuracy

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'{action}', fontsize=16, fontweight='bold')
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Losses')
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Accuracies')
    fig.tight_layout()
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
        plt.close()

    return classifier


def simulate_tracks(model, train_dataset, train_data, savedir, track_len, stride=None, simulate_single=False):
    track = train_data['track']
    pose = train_data['pose']
    flyids = train_data['flyids']

    if stride is None:
        stride = track_len

    sessions = []
    if simulate_single:
        for session in train_dataset.sessions:
            if session.duration < track_len + 1:
                continue
            sessions.append((session.start_frame, session.duration, [session.agent_id]))
    else:
        n_flies, n_frames, _ = train_dataset.inputs['velocity'].array.shape
        session_frames = np.zeros((n_flies, n_frames), int)
        for i, session in enumerate(train_dataset.sessions):
            session_frames[session.agent_id, session.start_frame:session.start_frame + session.duration] = i

        # Find continuous chunks of frames where all valid session ids are same per fly
        diffs = np.zeros_like(session_frames)
        diffs[:, 1:] = np.abs(np.diff(session_frames, axis=1))
        start_frames = np.where(diffs.max(0) > 0)[0]

        for i in range(len(start_frames) - 1):
            start_frame = start_frames[i]
            end_frame = start_frames[i + 1]
            duration = end_frame - start_frame
            if duration < track_len + 1:
                continue
            agent_ids = np.where(session_frames[:, start_frame] > 0)[0]
            sessions.append((start_frame, duration, agent_ids))

    # Simulate
    sim_track = copy.deepcopy(train_data['track'])
    sim_frame = np.zeros(sim_track.array.shape[:2])
    for i, session in enumerate(sessions):
        print(f"Processing session {i} / {len(sessions)}")
        session_start, session_duration, agent_ids = session
        start_frames = np.arange(session_start, session_start + session_duration - track_len, stride)

        agent_str = str(agent_ids[0])
        for a in range(1, len(agent_ids)):
            agent_str = agent_str + f"_{agent_ids[a]}"

        for start_frame in start_frames:
            # save_str = f"session_{i}_startframe_{start_frame}_agent_id={agent_str}.npy"
            save_str = f"session_{i}_startframe_{start_frame}_agentid_{agent_str}_tracklen_{track_len}.npy"
            save_path = os.path.join(savedir, save_str)

            if os.path.exists(save_path):
                pred_track = np.load(save_path)
            else:
                # print("Can't find the data")
                # tmp = 1/0
                gt_track, pred_track = simulate(
                    dataset=train_dataset,
                    model=model,
                    track=track,
                    pose=pose,
                    identities=flyids,
                    track_len=track_len,
                    burn_in=train_dataset.context_length,
                    max_contextl=train_dataset.context_length,
                    agent_ids=agent_ids,
                    start_frame=start_frame,
                )
                pred_track = pred_track[agent_ids]
                np.save(save_path, pred_track)

            sim_track.array[agent_ids, start_frame:start_frame + track_len] = pred_track
            sim_frame[agent_ids, start_frame:start_frame + track_len] = (np.arange(pred_track.shape[1]) + 1)[None, :]

    return sim_track, sim_frame


def create_dataset_from_ref_data(ref_dataset, ref_data, sim_track, sim_frame):
    # Valid date only where flies are being simulated
    isdata = copy.deepcopy(ref_data['isdata'])
    isdata[sim_frame.T <= ref_dataset.context_length] = False
    # isdata[sim_frame.T < 1] = False

    # Start of each data is the first frame of simulation
    isstart = sim_frame.T == ref_dataset.context_length + 1
    # isstart = sim_frame.T == 1

    # Copy flyids and scale_perfly from original track
    flyids = copy.deepcopy(ref_data['flyids'])
    scale_perfly = copy.deepcopy(ref_data['pose'].operations[0].scale_perfly)

    # Use all outputs of valid data
    useoutputmask = np.ones(isdata.shape, dtype=bool)

    # Compute features for simulated data
    sensory = Sensory()(sim_track, isdata=isdata)
    pose = Pose()(sim_track, scale_perfly=scale_perfly, flyid=flyids, isdata=isdata)
    assert not np.any(np.isnan(sensory.array[isdata.T])) and np.all(
        np.isnan(sensory.array[~isdata.T])), "Sensory features should be nan iff isdata == False"
    assert not np.any(np.isnan(pose.array[isdata.T])) and np.all(
        np.isnan(pose.array[~isdata.T])), "Pose features should be nan iff isdata == False"
    velocity = Velocity(featrelative=featrelative, featangle=featangle)(pose, isstart=isstart)
    metadata = {'labels': {'velocity': {'pose': flyids.T, 'velocity': pose.array}},
                'inputs': {'pose': {'pose': flyids.T},
                           'velocity': {'pose': flyids.T}}}
    args = {
        'context_length': ref_dataset.context_length,
        'isstart': isstart,
        'metadata': metadata,
        'useoutputmask': useoutputmask
    }

    # Assemble the dataset
    sim_dataset = Dataset(
        inputs=apply_opers_from_data(ref_dataset.inputs, {'velocity': velocity, 'pose': pose, 'sensory': sensory}),
        labels=apply_opers_from_data(ref_dataset.labels, {'velocity': velocity}),
        **args
    )

    sim_data = {'track': sim_track,
                'pose': pose,
                'sensory': sensory,
                'velocity': velocity,
                'isstart': isstart,
                'isdata': isdata,
                'sim_frame': sim_frame}

    return sim_dataset, sim_data


def plot_feature_distributions(train_data, sim_data, valid_frames, frame_label, data_name='pose', use_log=False, pdf=None):

    sim_feat = sim_data[data_name].array[valid_frames]
    sim_feat = sim_feat[~np.isnan(sim_feat).max(1)]

    # train_feat = train_dataset.inputs['velocity'].array[valid_frames]
    train_feat = train_data[data_name].array[valid_frames]
    train_feat = train_feat[~np.isnan(train_feat).max(1)]

    fig = plt.figure(figsize=(15, 10))
    log_str = 'log ' if use_log else ''
    fig.suptitle(f'{log_str}{data_name}, {frame_label} frames', fontsize=16, fontweight='bold')
    feat_ids = [0, 1, 2, 27, 28] + list(np.arange(3, 27))
    for i, idx in enumerate(feat_ids):
        ax = plt.subplot(6, 5, i + 1)
        dat = train_feat[:, idx]
        dat = dat[~np.isnan(dat)]
        # if use_log:
        #     eps = 0.001
        #     dat = np.log(np.abs(dat) + eps)
        count, bin_edges = np.histogram(dat, 1000)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        plt.plot(bin_centers, count)
        dat = sim_feat[:, idx]
        dat = dat[~np.isnan(dat)]
        # if use_log:
        #     dat = np.log(np.abs(dat) + eps)
        count, _ = np.histogram(dat, bin_edges)
        plt.plot(bin_centers, count)
        plt.yticks([])
        plt.title(posenames[idx])
        if use_log:
            ax.set_xscale('symlog', linthresh=0.05)
    fig.tight_layout()
    if pdf is None:
        plt.show()
    else:
        pdf.savefig(fig)
        plt.close()


def classify_real_fake(train_feat, sim_feat, n_epochs, control=False, device='cuda:0', normalize=False, mlp_layers=(512, 256, 128)):
    if isinstance(device, str):
        device = torch.device(device)

    train_feat = train_feat[~np.isnan(train_feat).max(1)]
    sim_feat = sim_feat[~np.isnan(sim_feat).max(1)]

    if control:
        # Use half of real data as fake
        X = train_feat
        y = np.zeros(X.shape[0], int)
        y[y.shape[0] // 2:] = 1
    else:
        X = np.concatenate([train_feat, sim_feat], axis=0)
        y = np.zeros(X.shape[0], int)
        y[train_feat.shape[0]:] = 1

    # Split into train and test
    train_frac = 0.8
    count_true = (y == 0).sum()
    count_fake = (y == 1).sum()
    train_inds = np.concatenate([np.arange(int(count_true * train_frac)),
                                 np.arange(int(count_fake * train_frac)) + count_true])
    test_inds = np.setdiff1d(np.arange(y.shape[0]), train_inds)
    X_train = X[train_inds]
    y_train = y[train_inds]
    X_test = X[test_inds]
    y_test = y[test_inds]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    # Initialize model, loss, optimizer
    model = MLP(input_size=train_feat.shape[1], layer_size=mlp_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(optimizer)

    print(X_train.device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name} | Device: {param.device}")

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    for example in train_loader:
        for item in example:
            print(item.device)
        break

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # if (epoch + 1) % 20 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {avg_loss:.4f}')

    # Compute results
    logits = model(X_test)
    predictions = (logits > 0.5).float()
    accuracy = (predictions == y_test).sum().item() / y_test.size(0)
    return accuracy, logits, y_test, test_inds, model


def classify_action(train_dataset, data, action, device):
    # Extract relevant frames

    # Train the classifier

    # Compute validation numbers

    # Run classifier on real data

    # Run classifier on fake data

    # Compute number of frames and number of bouts per agent

    pass



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


def get_closest_fly(pred_trks, agent_id, kp):
    endpoints = [trk[agent_id, -1, :, kp] for trk in pred_trks]
    endpoints = np.array(endpoints)

    other_ids = np.concatenate([np.arange(0, agent_id), np.arange(agent_id + 1, 10)])
    other_flies = pred_trks[-1][other_ids, -1, :, kp]

    headings = np.array([trk[agent_id, -1, :, 7] - trk[agent_id, -1, :, 8] for trk in pred_trks])
    headings_norm = np.linalg.norm(headings, axis=1)
    headings = headings / headings_norm[:, None]
    # Note: this becomes a bit annoying because of the vectorization I'm doing

    dist_to_others = np.zeros((other_flies.shape[0], endpoints.shape[0]))
    angle_to_others = np.zeros_like(dist_to_others)
    for i, flyid in enumerate(other_flies):
        vec_to_others = other_flies[i] - endpoints
        dist_to_others[i, :] = np.linalg.norm(vec_to_others, axis=1)
        vec_to_others = vec_to_others / dist_to_others[i, :][:, None]
        dots = np.einsum("ij,ij->i", headings, vec_to_others)
        angles = np.arccos(np.clip(dots, -1, 1))
        # dist_to_others[i, angles > angle_thresh] = dist_to_others.max()
        angle_to_others[i, :] = angles

    distance = dist_to_others + 2*angle_to_others

    closest_id = np.argmin(distance, axis=0)
    dist_to_closest = np.min(distance, axis=0)

    closest_fly = np.zeros(endpoints.shape[0], int)
    for i, idx in enumerate(np.unique(closest_id)):
        inds = np.where(np.logical_and(closest_id == idx, dist_to_closest < 5))[0]
        closest_fly[inds] = other_ids[idx] + 1
    return closest_fly


# QUALITATIVE
import time
import sys
from scipy import ndimage

sys.path.append("/groups/branson/home/eyjolfsdottire/code/")
from flymovie.movies import Movie

from apf.simulation import simulate
from flyllm.plotting import plot_arena
from flyllm.config import ARENA_RADIUS_PX, PXPERMM, ARENA_RADIUS_MM

from flyllm.config import keypointnames
from flyllm.features import body_centric_kp
from apf.plotting import get_cropbox, get_rotmat, crop_image, crop_keypoints, rotate_image, rotate_keypoints


def plot_qualitative(train_dataset, train_data, raw_data, model, start_frame, track_len, agent_ids):
    track = train_data['track']
    pose = train_data['pose']
    flyids = train_data['flyids']

    contextl = train_dataset.context_length

    # simulate
    # start_frame = 45004
    # agent_ids = [0, 1, 4, 6]
    # agent_ids = [0]
    # track_len = contextl_repeats * contextl

    t0 = time.time()
    n_trials = 5
    pred_trks = []
    for i in range(n_trials):
        gt_track, pred_track = simulate(
            dataset=train_dataset,
            model=model,
            track=track,
            pose=pose,
            identities=flyids,
            track_len=track_len,
            burn_in=contextl,
            max_contextl=contextl,
            agent_ids=agent_ids,
            start_frame=start_frame,
        )
        pred_trks.append(pred_track)
    time.time() - t0

    first_frame = contextl
    last_frame = None

    video_idx = raw_data['videoidx'][start_frame + first_frame].astype(int)[0]
    video_frame = raw_data['frames'][start_frame + first_frame][0]
    moviepath = os.path.join(str(raw_data['expdirs'][video_idx]), 'movie.ufmf')
    movie = Movie(moviepath)
    frame, val = movie.get_frame(video_frame)

    first_img = movie.get_frame(raw_data['frames'][start_frame][0])[0]
    last_img = movie.get_frame(raw_data['frames'][start_frame + track_len][0])[0]

    plt.figure(figsize=(10, 10))

    frame = frame.astype(float) / 255
    last_img = last_img.astype(float) / 255

    plt.imshow(frame + 0.3 * last_img, cmap='gray')
    plt.imshow(frame, cmap='gray')

    kp = 7

    half_width = frame.shape[0] / 2
    clr_ct = 1
    def get_unique_colors(N, cmap_name='jet'):
        cmap = plt.get_cmap(cmap_name)
        indices = np.linspace(0, 1, N)
        colors = cmap(indices)
        return colors

    clrs = get_unique_colors(10)
    gray = 0.5 * np.ones(3)
    for agent_id in range(gt_track.shape[0]):
        if agent_id in agent_ids:
            closest_fly = get_closest_fly(pred_trks, agent_id, kp)
            for i, pred_track in enumerate(pred_trks):
                x, y = pred_track[agent_id, first_frame:last_frame, :, kp].T * PXPERMM + half_width
                clr_ct = closest_fly[i]
                if clr_ct == 0:
                    plt.plot(x, y, '-', color=gray, markersize=1, linewidth=1, alpha=0.5, zorder=1)
                    plt.plot(x[-1], y[-1], '.', color=gray, markersize=4, alpha=0.5, zorder=1)
                else:
                    plt.plot(x, y, '-', color=clrs[clr_ct - 1], markersize=1, linewidth=1, alpha=0.5, zorder=10)
                    plt.plot(x[-1], y[-1], '.', color=clrs[clr_ct - 1], markersize=4, alpha=0.5, zorder=10)
                clr_ct += 1

            x, y = gt_track[agent_id, first_frame:last_frame, :, kp].T * PXPERMM + half_width
            plt.plot(x, y, '-w', markersize=1, linewidth=3, zorder=11)
            plt.plot(x[-1], y[-1], '.w', markersize=4, zorder=11)
            plt.plot(x, y, '-', color=clrs[agent_id], markersize=1, linewidth=1, zorder=12)
            plt.plot(x[-1], y[-1], '.', color=clrs[agent_id], markersize=2, zorder=12)

    plt.axis('equal')
    plt.axis('off')
    plt.show()

    # PLOT LOCOMOTION OF LEGS
    pred = pred_trks[3]

    agent = 0
    x, y = track.array[agent, :, :, kp].T * PXPERMM + half_width

    vel = (np.diff(x) ** 2 + np.diff(y) ** 2) ** 0.5
    log_vel = np.log(np.abs(vel) + 0.00001).flatten()
    # log_vel[np.isinf(np.abs(log_vel))

    plt.figure()
    valid = np.where(~np.isnan(log_vel))[0]
    count, bins = np.histogram(log_vel[valid], bins=100)
    binc = (bins[1:] + bins[:-1]) / 2
    plt.plot(binc, count, '-')
    plt.plot([-1.25, -1.25], [0, count.max() * 1.01], '-k')
    plt.plot([0, 0], [0, count.max() * 1.01], '-k')
    plt.show()

    x, y = pred[agent, contextl:, :, kp].T * PXPERMM + half_width
    vel = (np.diff(x) ** 2 + np.diff(y) ** 2) ** 0.5
    log_vel = np.log(np.abs(vel) + 0.00001).flatten()

    dur_thresh = 20
    labeled_array, num_components = ndimage.label(log_vel > 0)
    component_durs = np.zeros(num_components)
    component_maxs = np.zeros(num_components)
    for i in range(num_components):
        component_durs[i] = (labeled_array == i + 1).sum()
        component_maxs[i] = log_vel[labeled_array == i + 1].max()
    valid_components = np.where(component_durs > dur_thresh)[0] + 1
    walking = np.zeros(len(vel))
    for i, c in enumerate(valid_components):
        walking[labeled_array == c] = i + 1
    walking[0] = 0

    plt.figure(figsize=(7, 2))
    plt.scatter(np.arange(len(vel)), log_vel, c=log_vel, s=1, cmap='jet')
    plt.plot([0, len(vel)], [-1.25, -1.25], '-k')
    plt.plot([0, len(vel)], [0, 0], '-k')
    # plt.show()

    # plt.figure(figsize=(7, 2))
    plt.plot((walking > 0) * 2, '-', linewidth=1)
    plt.show()

    plt.figure()
    plt.imshow(frame, cmap='gray')
    plt.scatter(x[1:], y[1:], c=walking, s=2, cmap='jet')
    plt.show()

    # Pick a window from one of the walking bouts

    agent_id = agent

    legids = np.array([idx for idx, name in enumerate(keypointnames) if 'leg_tip' in name])
    legnames = [keypointnames[idx] for idx in legids]

    body_centric = body_centric_kp(pred[:, contextl:].T)[0].T

    bout_id = 7
    frames = np.where(walking == bout_id)[0]
    # frames = frames[:15]

    # agent_id=0
    legpos = body_centric[agent_id, frames, :][..., legids]

    frame_id = contextl + 0  # frames[0]
    video_frame = raw_data['frames'][start_frame + frame_id][0]
    frame, val = movie.get_frame(video_frame)
    frame = frame.astype(float) / 255
    kpt = pred[agent_id, frame_id, :, :].T * PXPERMM + half_width
    rotmat = get_rotmat(kpt)
    rot_img = rotate_image(frame, rotmat)
    rot_kpt = rotate_keypoints(kpt, rotmat)
    buf = 70
    cropbox = get_cropbox(rot_kpt, buf)
    crop_img, out_of_bounds = crop_image(rot_img, cropbox, fill_value=0)
    # crop_kpts = crop_keypoints(rot_kpt, cropbox)


    plt.figure(figsize=(17, 5))
    plt.subplot(1, 3, 3)
    plt.imshow(crop_img, cmap='gray')
    # image of the fly at the first frame
    for i in range(legpos.shape[-1]):
        x, y = legpos[:, :, i].T * PXPERMM + crop_img.shape[0] / 2

        plt.subplot(1, 3, 1)
        plt.plot(x, np.arange(len(x)))
        plt.subplot(1, 3, 2)
        plt.plot(y)
        plt.subplot(1, 3, 3)
        plt.plot(x, crop_img.shape[0] - y, '.-')
        # plt.scatter(x, crop_img.shape[0]-y, c=np.arange(len(x)), s=5, cmap='jet')
    plt.subplot(1, 3, 1)
    plt.xlabel('fly left-to-right')
    plt.ylabel('time')
    plt.subplot(1, 3, 2)
    plt.ylabel('fly back-to-front')
    plt.xlabel('time')
    # plt.legend(legnames, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.subplot(1, 3, 3)
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def main():
    configfile = "/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/config_fly_llm_predvel_simpler_20251104.json"
    modelfile = os.path.join(
        '/groups/branson/home/eyjolfsdottire/code/AnimalPoseForecasting/notebooks/flyllm_models',
        'flypredvel_20251007_simple_20251119T080019_epoch200.pth')
    savedir = 'notebooks/train_data/synthetic'

    eval_simulation(configfile, modelfile, savedir)
