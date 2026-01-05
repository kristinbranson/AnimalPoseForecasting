import torch
import numpy as np
from apf.dataset import invert_to_named, array_to_data

def labels_to_discrete_continuous(data_labels):
    
    # discrete and continuous (z-scored) predictions
        
    data_fusion = invert_to_named(data_labels,'fusion',return_data=True)
    fusion_op = data_fusion.operations[-1]
    velocity_unfused = fusion_op.unfuse(data_fusion)
    feature_names_unfused = fusion_op.unfuse_feature_names(data_fusion)
    velocity_discrete = velocity_unfused['discretize'] # n_agents x n_frames x (d_discrete * nbins)
    velocity_continuous = velocity_unfused['identity'] # n_agents x n_frames x d_continuous
    feature_names_discrete = feature_names_unfused['discretize']
    feature_names_continuous = feature_names_unfused['identity']
    idx = [op.name for op in fusion_op.operations].index('discretize')
    discretize_op = fusion_op.operations[idx]    
    velocity_discrete = discretize_op.unflatten(velocity_discrete) # n_agents x n_frames x d_discrete x nbins
    feature_names_discrete = discretize_op.invert_feature_names(feature_names_discrete)
    n_bins = velocity_discrete.shape[-1]
    d_discrete = velocity_discrete.shape[-2]
    d_continuous = velocity_continuous.shape[-1]
    
    return {
        'discrete': velocity_discrete,
        'continuous': velocity_continuous,
        'discrete_feature_names': feature_names_discrete,
        'continuous_feature_names': feature_names_continuous,
        'n_bins': n_bins,
        'd_discrete': d_discrete,
        'd_continuous': d_continuous
    }
    
def labels_to_velocity_samples(data_labels,nsamples=10):
    # invert to velocity -- discrete features will be inverted to continuous by sampling
    # do this nsamples times -- this is not the most efficient way to do this
    for samplei in range(nsamples):
        velocity_curr, feature_names = invert_to_named(data_labels, 'velocity', return_feature_names=True)
        if samplei==0:
            velocity_samples = np.zeros((nsamples,)+velocity_curr.shape,dtype=velocity_curr.dtype)
        velocity_samples[samplei,...] = velocity_curr
    return velocity_samples, feature_names

def compute_error(dataset,true_data,pred_data,nsamples=10):
    """
    compute_error(true_labels,pred_labels,nsamples=10,collapsetime=True)
    Computes the error between the true and predicted labels in various ways.
    Arguments:
    dataset: Dataset object containing the labels.
    true_data: Dictionary containing the true data. Must have key 'velocity' with Data object (using this because I don't have 
        a way to perfectly invert discretize)
    pred_data: Dictionary containing the predicted data. Must have key 'labels' with key 'velocity' containing Data object.
    nsamples: Number of samples to draw from the predicted distribution.
    Returns:
    Dictionary with the following keys:
        'isdata': Boolean array of shape (n_agents, n_frames) indicating which data points are valid (not NaN).
        'n': Number of valid data points.
        'velocity_absdiff_samplemean_datamean': Mean absolute difference in velocity over samples and data points, shape (dpose,).
        'velocity_absdiff_samplemin_datamean': Minimum absolute difference in velocity over samples, mean over data points, shape (dpose,).
        'discrete_velocity_ce_datamean': Cross-entropy error for discrete velocity predictions, mean over data points, shape (ddiscrete,).
        'continuous_zvelocity_absdiff_datamean': Mean absolute difference in continuous (z-scored) velocity predictions, mean over data points, shape (dcontinuous,).
        'velocity_absdiff_samplemean': Mean absolute difference in velocity over samples, shape (n, dpose).
        'velocity_absdiff_samplemin': Minimum absolute difference in velocity over samples, shape (n, dpose).
        'discrete_velocity_ce': Cross-entropy error for discrete velocity predictions, shape (n, ddiscrete).
        'continuous_zvelocity_absdiff': Absolute difference in continuous (z-scored) velocity predictions, shape (n, dcontinuous).
    """    

    # output labels
    pred_labels = pred_data['labels']['velocity']
    true_labels = dataset.labels['velocity']
    isdata = ~np.all(np.isnan(pred_labels.array),axis=-1)

    # discrete and continuous (z-scored) predictions
    pred_discrete_continuous = labels_to_discrete_continuous(pred_labels)
    true_discrete_continuous = labels_to_discrete_continuous(true_labels)

    # velocity samples
    pred_velocity, velocity_feature_names = labels_to_velocity_samples(pred_labels,nsamples=nsamples) # (nsamples, n_agents, n_frames, dpose)
        
    true_velocity = true_data['velocity']
        
    n = np.count_nonzero(isdata).item()

    # error in velocity predictions    
    dvelocity = true_velocity.array[isdata][None,...] - pred_velocity[:,isdata]
    assert ~np.any(np.isnan(dvelocity)), 'dsample has unexpected nans'
    velocity_absdiff_samplemean = np.mean(np.abs(dvelocity),axis=0) # mean over samples, (n,dpose)
    velocity_absdiff_samplemin = np.min(np.abs(dvelocity),axis=0) # min over samples, (n,dpose)
    velocity_absdiff_samplemin_datamean = np.mean(velocity_absdiff_samplemin,axis=0) # mean over data points (dpose,)
    velocity_absdiff_samplemean_datamean = np.mean(velocity_absdiff_samplemean,axis=0) # mean over data points, (dpose,)

    # error in discrete velocity predictions -- cross-entropy
    discrete_velocity_ce = torch.nn.functional.cross_entropy(
        torch.tensor(pred_discrete_continuous['discrete'][isdata].reshape(-1,pred_discrete_continuous['n_bins'])),
        torch.tensor(true_discrete_continuous['discrete'][isdata].reshape(-1,pred_discrete_continuous['n_bins'])),
        reduction='none').reshape(n,pred_discrete_continuous['d_discrete']).numpy() # (n, ddiscrete)
    discrete_velocity_ce_datamean = np.mean(discrete_velocity_ce,axis=0) # mean over data points, (ddiscrete,)

    # error in continuous velocity predictions
    continuous_zvelocity_absdiff = np.abs(true_discrete_continuous['continuous'][isdata] - pred_discrete_continuous['continuous'][isdata]) # (n,dcontinuous)
    continuous_zvelocity_absdiff_datamean = np.mean(continuous_zvelocity_absdiff,axis=0) # mean over data points, (dcontinuous,)

    return {
        'isdata': isdata,
        'n': n,
        'velocity_feature_names': velocity_feature_names,
        'discrete_velocity_feature_names': pred_discrete_continuous['discrete_feature_names'],
        'continuous_zvelocity_feature_names': pred_discrete_continuous['continuous_feature_names'],
        'velocity_absdiff_samplemean_datamean': velocity_absdiff_samplemean_datamean,
        'velocity_absdiff_samplemin_datamean': velocity_absdiff_samplemin_datamean,
        'discrete_velocity_ce_datamean': discrete_velocity_ce_datamean,
        'continuous_zvelocity_absdiff_datamean': continuous_zvelocity_absdiff_datamean,
        'velocity_absdiff_samplemean': velocity_absdiff_samplemean,
        'velocity_absdiff_samplemin': velocity_absdiff_samplemin,
        'discrete_velocity_ce': discrete_velocity_ce,
        'continuous_zvelocity_absdiff': continuous_zvelocity_absdiff
    }