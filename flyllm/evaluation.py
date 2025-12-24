import torch
import numpy as np
from apf.dataset import invert_to_named, array_to_data


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
    pred_fusion = invert_to_named(pred_labels,'fusion',return_data=True)
    fusion_op = pred_fusion.operations[-1]
    pred_unfused = fusion_op.unfuse(pred_fusion)
    pred_discrete = pred_unfused['discretize'] # n_agents x n_frames x (d_discrete * nbins)
    pred_continuous = pred_unfused['identity'] # n_agents x n_frames x d_continuous
    idx = [op.name for op in fusion_op.operations].index('discretize')
    discretize_op = fusion_op.operations[idx]    
    pred_discrete = discretize_op.unflatten(pred_discrete) # n_agents x n_frames x d_discrete x nbins
    n_bins = pred_discrete.shape[-1]
    d_discrete = pred_discrete.shape[-2]
    d_continuous = pred_continuous.shape[-1]

    # invert to velocity -- discrete features will be inverted to continuous by sampling
    # do this nsamples times -- this is not the most efficient way to do this
    for samplei in range(nsamples):
        pred_velocity_curr = invert_to_named(pred_fusion, 'velocity')
        if samplei==0:
            pred_velocity = np.zeros((nsamples,)+pred_velocity_curr.shape,dtype=pred_velocity_curr.dtype)
        pred_velocity[samplei,...] = pred_velocity_curr
        
    true_velocity = true_data['velocity']
    
    # discrete and continuous (z-scored) true
    true_fusion = invert_to_named(true_labels,'fusion',return_data=True)
    true_unfused = fusion_op.unfuse(true_fusion)
    true_discrete = discretize_op.unflatten(true_unfused['discretize']) # n_agents x n_frames x d_discrete x nbins
    true_continuous = true_unfused['identity'] # n_agents x n_frames x d_continuous
    
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
        torch.tensor(pred_discrete[isdata].reshape(-1,n_bins)),
        torch.tensor(true_discrete[isdata].reshape(-1,n_bins)),
        reduction='none').reshape(n,d_discrete).numpy() # (n, ddiscrete)
    discrete_velocity_ce_datamean = np.mean(discrete_velocity_ce,axis=0) # mean over data points, (ddiscrete,)

    # error in continuous velocity predictions
    continuous_zvelocity_absdiff = np.abs(true_continuous[isdata] - pred_continuous[isdata]) # (n,dcontinuous)
    continuous_zvelocity_absdiff_datamean = np.mean(continuous_zvelocity_absdiff,axis=0) # mean over data points, (dcontinuous,)

    return {
        'isdata': isdata,
        'n': n,
        'velocity_absdiff_samplemean_datamean': velocity_absdiff_samplemean_datamean,
        'velocity_absdiff_samplemin_datamean': velocity_absdiff_samplemin_datamean,
        'discrete_velocity_ce_datamean': discrete_velocity_ce_datamean,
        'continuous_zvelocity_absdiff_datamean': continuous_zvelocity_absdiff_datamean,
        'velocity_absdiff_samplemean': velocity_absdiff_samplemean,
        'velocity_absdiff_samplemin': velocity_absdiff_samplemin,
        'discrete_velocity_ce': discrete_velocity_ce,
        'continuous_zvelocity_absdiff': continuous_zvelocity_absdiff
    }