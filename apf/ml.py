import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import logging

LOG = logging.getLogger(__name__)

# from apf.models import MLP

class MLP(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size=1):
        super(MLP, self).__init__()

        if len(layer_sizes) == 0:
            layers = [torch.nn.Linear(input_size, output_size)]
        else:
            layers = [torch.nn.Linear(input_size, layer_sizes[0])]
            for i in range(len(layer_sizes) - 1):
                layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.Linear(layer_sizes[-1], output_size))

        self.layers = torch.nn.ModuleList(layers)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        return self.sigmoid(self.layers[-1](x))


def transform_dataset(dataset, transformer, transformed_path, batch_size=64, overwrite=False, save_layers=None):
    """

    :param dataset:
    :param transformer:
    :param transformed_path:
    :param batch_size:
    :param overwrite:
    :return:
    """
    # It takes about 3 hours to process 10M data points.
    device = next(transformer.parameters()).device
    transformer.eval()

    # Initialize transformed matrix (load it if exists)
    d_model = transformer.d_model
    n_layers = len(transformer.transformer_encoder.layers)
    d_input = dataset.d_input
    n_agents, n_frames = dataset.inputs['velocity'].array.shape[:2]
    contextl = dataset.context_length
    mode = 'r+' if os.path.exists(transformed_path) else 'w+'
    # TODO: I'm not sure if this is the optimal shape ordering for speed and caching...
    transformed = np.memmap(transformed_path, dtype=np.float32, mode=mode, shape=(n_layers, n_agents, n_frames, d_model))

    if save_layers is None:
        save_layers = np.arange(n_layers)
    elif save_layers == 'some':
        save_layers = [0, int(round(n_layers/2)), n_layers-1]

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
    n_frames = len(frames)
    n_skipped = 0
    print(f"frames to compute: {n_frames}")
    for i in tqdm(range(len(frames))):
        frame_id = frames[i]
        agent_id = agents[i]
        layer_sums = np.abs(transformed[save_layers, agent_id, frame_id]).sum(-1)
        any_missing = np.any(layer_sums == 0)
        if not any_missing and not overwrite: # and np.abs(transformed[:, agent_id, frame_id]).sum() > 0:
            # Skip frames that have already been processed
            n_skipped += 1
            continue
        batch_inputs[batch_idx] = dataset.get_chunk(
            start_frame=frame_id - contextl + 1, duration=contextl, agent_id=agent_id
        )['input']
        batch_indices[0, batch_idx] = agent_id
        batch_indices[1, batch_idx] = frame_id
        batch_idx += 1
        if batch_idx == batch_size:
            with torch.no_grad():
                layer_outs = transformer.forward(torch.from_numpy(batch_inputs).to(device), return_hidden=True)['hidden']
            bagents, bframes = batch_indices
            for layer_id, out in enumerate(layer_outs):
                if layer_id in save_layers:
                    transformed[layer_id, bagents, bframes, :] = out[:, -1]
            batch_idx = 0
    print(f"frames skipped: {n_skipped}")
    return transformed


def apply_binary_classifier(
        model: torch.nn.Module, Xdata: np.ndarray, ydata: np.ndarray | None,
        frames: np.ndarray, agents: np.ndarray, batch_size: int = 64):
    """Applies model to subset of data as specified by [frames, agents], in chunks of batch_size.

    Parameters
        model: Model to be applied to the input data
        Xdata: Data of shape (n_agents, n_frames, data_dim)
        ydata: Ground truth labels for the data, shape (n_agents, n_frames, 1). If None, no accuracy will be computed.
        frames: Which frames to use from the data, comes paired with agents. Shape (n_valid_entries, )
        agents: Which agents to use form teh data, comes paired with frames. Shape (n_valid_entries, )
        batch_size: Number of samples to process simultaneously by the model

    Returns
        logits: Raw outputs form the model
        predictions: Model outputs thresholded at 0.5
        accuracy: Accuracy of predictions compared with ydata, if provided
    """
    model.eval()
    device = next(model.parameters()).device
    logits = np.zeros(len(frames))
    for i in range(0, len(frames), batch_size):
        chunk = torch.from_numpy(Xdata[agents[i:i + batch_size], frames[i:i + batch_size]]).to(device)
        logits[i:i + batch_size] = model(chunk).detach().cpu().numpy()[:, 0]
    chunk = torch.from_numpy(Xdata[agents[i + batch_size:], frames[i + batch_size:]]).to(device)
    with torch.no_grad():
        logits[i + batch_size:] = model(chunk).detach().cpu().numpy()[:, 0]
    predictions = (logits > 0.5).astype(float)
    if ydata is None:
        accuracy = None
        balanced_accuracy = None
        F1 = None
        stats = None
    else:
        gt = ydata[agents, frames]
        # accuracy = (predictions == gt).sum() / len(frames)
        tp = np.logical_and(predictions == 1, gt == 1).sum()
        fp = np.logical_and(predictions == 1, gt == 0).sum()
        tn = np.logical_and(predictions == 0, gt == 0).sum()
        fn = np.logical_and(predictions == 0, gt == 1).sum()
        precision = tp / (tp + fp)
        tpr = recall = tp / (tp + fn)
        tnr = tn / (tn + fp)
        balanced_accuracy = (tpr + tnr) / 2
        F1 = precision * recall * 2 / (precision + recall)
        stats = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    return logits, predictions, balanced_accuracy, F1, stats


def train_binary_classifier(train_data, val_data, train_label, val_label, device, mlp_layers, lr=0.0001, min_epochs=2, max_epochs=50):
    """
    Parameters:
        train_data: shape (n_agents, n_frames, n_features)
        vald_data: shape (n_agents, n_frames, n_features)
        train_label: shape (n_agents, n_frames)
        val_label: shape (n_agents, n_frames)
        device: torch device to train the model on

    Returns:
        classifier:
        stats:
    """
    # TODO: can I remove this? Maybe add an assert?
    if train_data.shape[0] < train_label.shape[0]:
        train_label = train_label[:train_data.shape[0]]
    if val_data.shape[0] < val_label.shape[0]:
        val_label = val_label[:val_data.shape[0]]

    # valid = np.logical_and(~np.isnan(train_data.sum(-1)), ~np.isnan(train_label))
    valid = ~np.isnan(train_label)
    agents, frames = np.where(valid)

    # valid = np.logical_and(~np.isnan(val_data.sum(-1)), ~np.isnan(val_label))
    valid = ~np.isnan(val_label)
    val_agents, val_frames = np.where(valid)

    # TODO: need to skip the frames that have all zeros
    # maybe the best way to do that is in the code outside
    # find the invalid frames (within valid labels) and set the corrsponding label frames to be nan,
    # then this will work automatically and only needs to be done once

    pos_train = np.where(train_label[agents, frames] == 1)[0]
    neg_train = np.where(train_label[agents, frames] == 0)[0]
    n_pos = len(pos_train)
    n_neg = len(neg_train)
    n_min = min(n_pos, n_neg)
    n_frames = n_min * 2

    # TODO: make these configurable parameters
    batch_size = 32
    n_epochs = max_epochs
    if n_frames < 5000:
        batch_size = 8
    if n_frames > 100000:
        batch_size = 128
    half_batch_size = batch_size // 2
    # n_batches = n_frames // batch_size
    n_batches = n_min // half_batch_size
    if n_frames > 100000:
        n_batch_iter = 25000
        n_epochs = n_batch_iter // n_batches
        n_epochs = int(np.clip(n_epochs, min_epochs, max_epochs))

    print(f"batch_size={batch_size}, n_epochs={n_epochs}, n_pos={n_pos}, n_neg={n_neg}, n_min={n_min}")

    # Initialize model, loss, optimizer
    classifier = MLP(train_data.shape[-1], mlp_layers).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # TODO: Make this faster by using torch dataloader
    #

    # Training loop
    losses = np.zeros(n_epochs)
    accuracies = np.zeros((n_epochs, 2))
    F1s = np.zeros((n_epochs, 2))
    for epoch in tqdm(range(n_epochs)):
        classifier.train()
        epoch_loss = 0
        perm_inds_pos = np.random.permutation(n_pos)
        perm_inds_neg = np.random.permutation(n_neg)
        for i in tqdm(range(0, n_min, half_batch_size), leave=False):
            optimizer.zero_grad()
            pos_inds = pos_train[perm_inds_pos[i:i + half_batch_size]]
            neg_inds = neg_train[perm_inds_neg[i:i + half_batch_size]]
            inds = np.concatenate([pos_inds, neg_inds])

            if (np.isnan(train_data[agents[inds], frames[inds]])).any():
                LOG.warning("Batch contains nan, skipping!")
                continue
            if (train_data[agents[inds], frames[inds]].sum(-1) == 0).any():
                LOG.warning("Batch contains all zeros, skipping!")
                continue

            batch_X = torch.from_numpy(train_data[agents[inds], frames[inds]]).to(device)
            batch_y = torch.from_numpy(train_label[agents[inds], frames[inds]][:, None]).to(device)

            outputs = classifier(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses[epoch] = avg_loss

        # Compute results
        logits, predictions, train_accuracy, train_f1, train_stats = apply_binary_classifier(
            classifier, train_data, train_label, frames=frames, agents=agents, batch_size=128)
        testlogits, testpredictions, test_accuracy, test_f1, test_stats = apply_binary_classifier(
            classifier, val_data, val_label, frames=val_frames, agents=val_agents, batch_size=128)

        accuracies[epoch, 0] = train_accuracy
        accuracies[epoch, 1] = test_accuracy

        F1s[epoch, 0] = train_f1
        F1s[epoch, 1] = test_f1

    PR_curve = precision_recall_curve(train_label[agents, frames], logits, drop_intermediate=True)
    val_PR_curve = precision_recall_curve(val_label[val_agents, val_frames], testlogits, drop_intermediate=True)
    stats = {
        'losses': losses,
        'accuracies': accuracies,
        'F1s': F1s,
        'PR_curve': PR_curve,
        'val_PR_curve': val_PR_curve,
        'train_stats': train_stats,
        'test_stats': test_stats,
    }

    return classifier, stats


def train_binary_classifier_spedup(train_data, val_data, train_label, val_label, device, mlp_layers, lr=0.0001, min_epochs=2, max_epochs=50):
    """
    Parameters:
        train_data: shape (n_agents, n_frames, n_features)
        vald_data: shape (n_agents, n_frames, n_features)
        train_label: shape (n_agents, n_frames)
        val_label: shape (n_agents, n_frames)
        device: torch device to train the model on

    Returns:
        classifier:
        stats:
    """
    # valid = np.logical_and(~np.isnan(train_data.sum(-1)), ~np.isnan(train_label))
    valid = ~np.isnan(train_label)
    agents, frames = np.where(valid)

    # valid = np.logical_and(~np.isnan(val_data.sum(-1)), ~np.isnan(val_label))
    valid = ~np.isnan(val_label)
    val_agents, val_frames = np.where(valid)

    # TODO: need to skip the frames that have all zeros
    # maybe the best way to do that is in the code outside
    # find the invalid frames (within valid labels) and set the corrsponding label frames to be nan,
    # then this will work automatically and only needs to be done once

    pos_train = np.where(train_label[agents, frames] == 1)[0]
    neg_train = np.where(train_label[agents, frames] == 0)[0]
    n_pos = len(pos_train)
    n_neg = len(neg_train)
    n_min = min(n_pos, n_neg)
    n_frames = n_min * 2

    # TODO: make these configurable parameters
    batch_size = 32
    if n_frames < 5000:
        batch_size = 8
    if n_frames > 1000000:
        batch_size = 4096//4  #128
    n_batch_iter = 25000
    n_batches = n_frames // batch_size
    n_epochs = n_batch_iter // n_batches
    n_epochs = int(np.clip(n_epochs, min_epochs, max_epochs))
    half_batch_size = batch_size // 2

    print(f"batch_size={batch_size}, n_epochs={n_epochs}, n_pos={n_pos}, n_neg={n_neg}, n_min={n_min}")

    # Initialize model, loss, optimizer
    classifier = MLP(train_data.shape[-1], mlp_layers).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # TODO: Make this faster by using torch dataloader
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, X, y, frames, agents, half_batch_size):
            self.X = X
            self.y = y
            self.frames = frames
            self.agents = agents
            self.pos_start_inds = np.arange(0, n_pos - half_batch_size, half_batch_size)
            self.neg_start_inds = np.arange(0, n_neg - half_batch_size, half_batch_size)

            # These will change as dataset gets shuffled
            self.half_batch_ids = np.arange(half_batch_size)
            self.offset = np.random.randint(half_batch_size)
            self.half_batch_size = half_batch_size
            self.perm_pos_starts = np.random.permutation(self.pos_start_inds)
            self.perm_neg_starts = np.random.permutation(self.neg_start_inds)

        def shuffle(self):
            self.offset = np.random.randint(self.half_batch_size)
            self.perm_pos_starts = np.random.permutation(self.pos_start_inds)
            self.perm_neg_starts = np.random.permutation(self.neg_start_inds)

        def __getitem__(self, i):
            pos_inds = self.half_batch_ids + self.perm_pos_starts[i] + self.offset
            neg_inds = self.half_batch_ids + self.perm_neg_starts[i] + self.offset
            inds = np.concatenate([pos_inds, neg_inds])
            batch_X = self.X[self.agents[inds], self.frames[inds]]
            batch_y = self.y[self.agents[inds], self.frames[inds]][:, None]
            return batch_X, batch_y

        def __len__(self):
            return min(len(self.perm_pos_starts), len(self.perm_neg_starts))

    dataset = Dataset(train_data, train_label, frames, agents, half_batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) # , num_workers=4)

    # pos_start_inds = np.arange(0, n_pos - half_batch_size, half_batch_size)
    # neg_start_inds = np.arange(0, n_neg - half_batch_size, half_batch_size)
    # half_batch_ids = np.arange(half_batch_size)

    # Training loop
    losses = np.zeros(n_epochs)
    accuracies = np.zeros((n_epochs, 2))
    F1s = np.zeros((n_epochs, 2))
    for epoch in tqdm(range(n_epochs)):
        classifier.train()
        epoch_loss = 0

        dataloader.dataset.shuffle()

        # offset = np.random.randint(half_batch_size)
        # perm_pos_starts = np.random.permutation(pos_start_inds)
        # perm_neg_starts = np.random.permutation(neg_start_inds)

        # for i in tqdm(range(n_min//half_batch_size)):
        for i, (batch_X, batch_y) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()

            # pos_inds = half_batch_ids + perm_pos_starts[i] + offset
            # neg_inds = half_batch_ids + perm_neg_starts[i] + offset
            # inds = np.concatenate([pos_inds, neg_inds])
            #
            # if (np.isnan(train_data[agents[inds], frames[inds]])).any():
            #     LOG.warning("Batch contains nan, skipping!")
            #     continue
            # if (train_data[agents[inds], frames[inds]].sum(-1) == 0).any():
            #     LOG.warning("Batch contains all zeros, skipping!")
            #     continue
            #
            # batch_X = torch.from_numpy(train_data[agents[inds], frames[inds]]).to(device)
            # batch_y = torch.from_numpy(train_label[agents[inds], frames[inds]][:, None]).to(device)

            outputs = classifier(batch_X[0].to(device))
            loss = criterion(outputs, batch_y[0].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        losses[epoch] = avg_loss

        # Compute results
        logits, predictions, train_accuracy, train_f1, train_stats = apply_binary_classifier(
            classifier, train_data, train_label, frames=frames, agents=agents, batch_size=128)
        testlogits, testpredictions, test_accuracy, test_f1, test_stats = apply_binary_classifier(
            classifier, val_data, val_label, frames=val_frames, agents=val_agents, batch_size=128)

        accuracies[epoch, 0] = train_accuracy
        accuracies[epoch, 1] = test_accuracy

        F1s[epoch, 0] = train_f1
        F1s[epoch, 1] = test_f1

    PR_curve = precision_recall_curve(train_label[agents, frames], logits, drop_intermediate=True)
    val_PR_curve = precision_recall_curve(val_label[val_agents, val_frames], testlogits, drop_intermediate=True)
    stats = {
        'losses': losses,
        'accuracies': accuracies,
        'F1s': F1s,
        'PR_curve': PR_curve,
        'val_PR_curve': val_PR_curve,
        'train_stats': train_stats,
        'test_stats': test_stats,
    }

    return classifier, stats


def classify_real_fake(train_feat, sim_feat, n_epochs, control=False, device='cuda:0',
                       normalize=False, mlp_layers=(512, 256, 128), train_frac=0.8):
    # TODO: call train_binary_classifier internally?
    #   - maybe not, the training is quite simple here given that the data is balanced..
    #   - though, I'm not sure this handles big data well so maybe better to use the above

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
    count_true = (y == 0).sum()
    count_fake = (y == 1).sum()
    train_inds = np.concatenate([np.arange(int(count_true * train_frac)),
                                 np.arange(int(count_fake * train_frac)) + count_true])
    test_inds = np.setdiff1d(np.arange(y.shape[0]), train_inds)
    X_train = X[train_inds]
    y_train = y[train_inds]
    X_test = X[test_inds]
    y_test = y[test_inds]

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
    model = MLP(input_size=train_feat.shape[1], layer_sizes=mlp_layers).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
    model.eval()
    logits = model(X_test)
    predictions = (logits > 0.5).float()
    accuracy = (predictions == y_test).sum().item() / y_test.size(0)
    return accuracy, logits, y_test, test_inds, model
