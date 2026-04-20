import numpy as np
import torch

import tqdm
from apf.utils import set_batch_concat, allocate_batch_concat, clip_batch_concat, get_model_device, ndarray_to_tensor, tensor_to_ndarray
from apf.models import ( 
    get_output_and_attention_weights,
    pred_apply_fun,
    get_causal_mask
)
import apf.dataset
from flyllm.pose import FlyExample
from contextlib import nullcontext
import datetime
import copy
from flyllm.features import regularize_pose

def predict(example,model,config,mask=None,device=None,is_causal=True,debugcheat=False):
    
    if device is None:
        device = get_model_device(model)
    if mask is None and is_causal:
        contextl = example['input'].shape[1]
        mask = get_causal_mask(device=device, contextl=contextl)        
    
    if debugcheat:
        pred = {'continuous': example['labels'].clone(), 'discrete': example['labels_discrete'].clone(), 
                'todiscretize': example['labels_todiscretize'].clone()}
    else:
        with torch.no_grad():
            if example['input'].device != device:
                example['input'] = example['input'].to(device=device)
            pred = model.output(example['input'], mask=mask, is_causal=is_causal)
            if config['modelstatetype'] == 'prob':
                pred = model.maxpred(pred)
            elif config['modelstatetype'] == 'best':
                pred = model.randpred(pred)
    return pred
    

def predict_all(dataset=None, model=None, config=None, keepall=True, earlystop=None, debugcheat=False,
                savepredfile=None,saveinterval=600,nkeep=None,batchsize=None,shuffle=False,stride=None):
    """
    predict_all(dataset=None, model=None, config=None, mask=None, keepall=True, earlystop=None, debugcheat=False,
                savepredfile=None,saveinterval=600,nkeep=None,batchsize=None,shuffle=False,stride=None)
    Generate predictions for all (or a subset of) frames in a dataset in a single pass, giving sufficient temporal context
    for each frame. It iteratively creates batches of length contextl = dataset.context_length, sampled every stride 
    frames. The predictions for the last nkeep frames of each batch are retained. 

    Parameters:
    dataset: Dataset object containing the data to predict on.
    model: The trained neural network. Required.
    config: Configuration dict. Required.
    mask: Attention mask for the transformer, if None, a causal mask will be created. Default None
    keepall: If True, keep all timestep predictions; if False, only keep last nkeep timesteps. Default True
    earlystop: If not None, only process this many batches. Default None
    debugcheat: If True, copy labels to predictions instead of running the model (for debugging). Default False
    savepredfile: If not None, periodically save intermediate predictions to this file. Default None
    saveinterval: Number of seconds between saving intermediate predictions to disk. Default 600 seconds
    nkeep: Number of last timesteps to keep from each batch if keepall is False. If None and stride is provided,
        nkeep is set to stride. Default None
    stride: Sample every contextl sequence every stride frames. If None, stride = nkeep. Default None
    batchsize: Batch size to use when creating the dataloader if dataloader is not provided. If None, uses 
        config['test_batch_size']. Default None
    shuffle: Whether to shuffle the dataset when creating the dataloader if dataloader is not provided. Default False
    Returns:
    all_pred: Dict or tensor of predictions, shape (N, nkeep, d) or (N, d) if nkeep=1
    metadata: Dict of metadata tensors with keys 'labelidx', 'frame', 'agent_id'

    """
    
    assert model is not None
    assert config is not None
    assert dataset is not None

    with torch.no_grad():
        w = next(iter(model.parameters()))
        device = w.device

    contextl = dataset.context_length

    mask = get_causal_mask(device=device, contextl=contextl)
    is_causal = True

    if batchsize is None:
        batchsize = config['test_batch_size']
        
    # if keepall, set nkeep to contextl
    if keepall:
        nkeep = contextl
    else:
        if nkeep is None:
            nkeep = stride
        assert nkeep is not None

    # if stride is not provided, set to nkeep to get predictions for all frames
    if stride is None:
        stride = nkeep        

    # set stride in dataset
    stride0 = dataset.stride
    if stride != stride0:
        dataset.set_stride(stride)

    try:

        if saveinterval is None:
            saveinterval = np.inf

        # create dataloader
        dataloader = apf.dataset.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, pin_memory=True)
        
        #example_params = dataset.get_flyexample_params()
        model.eval()

        # compute predictions and labels for all validation data using default masking
        all_pred = None
        metadata = {'labelidx': None,
                    'frame': None,
                    'agent_id': None}

        #all_mask = []
        # all_pred_discrete = []
        # all_labels_discrete = []
        off = 0
        n = len(dataloader)

        lastsavetime = datetime.datetime.now()
        
        if earlystop is not None:
            n = min(n,earlystop)

        for i,example in tqdm.tqdm(enumerate(dataloader),total=n):
            if earlystop is not None and i >= earlystop:
                break
            
            pred = predict(example,model,config,mask=mask,device=device,is_causal=is_causal,debugcheat=debugcheat)
            frame = example['metadata']['start_frame'][:, None] + np.arange(contextl)[None,:]
            if not keepall:
                # only keep the last nkeep predictions
                if isinstance(pred, dict):
                    pred = {k: v[:,-nkeep:] for k, v in pred.items()}
                else:
                    pred = pred[:,-nkeep:]
                frame = frame[:,-nkeep:]

            if isinstance(pred, dict):
                pred = {k: v.cpu() for k, v in pred.items()}
            else:
                pred = pred.cpu()

            if all_pred is None:
                # allocate
                all_pred = allocate_batch_concat(pred, n)
                sz = n*example['metadata']['idx'].shape[0]
                metadata['labelidx'] = torch.zeros(sz, dtype=torch.int64)
                metadata['frame'] = torch.zeros((sz,nkeep), dtype=torch.int64)
                metadata['agent_id'] = torch.zeros(sz, dtype=torch.int64)

            # assign
            all_pred,off1 = set_batch_concat(pred, all_pred, off)
            metadata['labelidx'][off:off1] = example['metadata']['idx']
            metadata['frame'][off:off1] = frame
            metadata['agent_id'][off:off1] = example['metadata']['agent_id']
            
            off = off1

            timenow = datetime.datetime.now()
            if savepredfile is not None and (timenow - lastsavetime).total_seconds() > saveinterval:
                print(f"Saving {off} predictions to {savepredfile}")
                metadata_save = {k: v[:off] for k, v in metadata.items()}
                savestuff = {'all_pred': clip_batch_concat(all_pred, off), 'metadata': metadata_save, 'i': i}
                np.savez(savepredfile, **savestuff)
                lastsavetime = timenow
                
            # pred1 = dataset.get_full_pred(pred)
            # labels1 = dataset.get_full_labels(example=example,use_todiscretize=True)
            # pred_obj = FlyExample(example_in=pred,**example_params)

            # example_obj = FlyExample(example_in=example, **example_params)
            # label_obj = example_obj.labels
            # pred_obj = label_obj.copy()
            # pred_obj.erase_labels()
            # pred_obj.set_prediction(pred)

            # for i in range(np.prod(label_obj.pre_sz)):
            #     all_pred.append(pred_obj.copy_subindex(idx_pre=i))
            #     all_labels.append(label_obj.copy_subindex(idx_pre=i))

            # if dataset.discretize:
            #   all_pred_discrete.append(pred['discrete'])
            #   all_labels_discrete.append(example['labels_discrete'])
            # if 'mask' in example:
            #   all_mask.append(example['mask'])
        all_pred = clip_batch_concat(all_pred, off)
        for k in metadata.keys():
            metadata[k] = metadata[k][:off]
        
        # all_pred[...] is (nbatches * batchsize) x nkeep x d
        
        # get rid of the batch dimension if nkeep == stride
        if not keepall and (nkeep == 1):
            if isinstance(all_pred, dict):
                all_pred = {k: v.squeeze(1) for k, v in all_pred.items()}
            else:
                all_pred = all_pred.squeeze(1)
        
        # create FlyLabels objects
    
    except Exception as e:
        # reset stride
        print(f'Error during prediction, resetting dataset stride to {stride0}')
        dataset.set_stride(stride0)
        raise e

    dataset.set_stride(stride0)

    return all_pred, metadata  # ,all_mask,all_pred_discrete,all_labels_discrete

# # this didn't seem to speed inference up
# def predict_all_twostream(dataloader, dataset, model, config, mask, keepall=True, earlystop=None, debugcheat=False, chunksize=5):
#     is_causal = dataset.ismasked() == False

#     with torch.no_grad():
#         w = next(iter(model.parameters()))
#         device = w.device

#     #example_params = dataset.get_flyexample_params()
#     model.eval()
#     dataset.set_eval_mode()

#     # compute predictions and labels for all validation data using default masking
#     all_pred = None
#     labelidx = None
#     #all_mask = []
#     # all_pred_discrete = []
#     # all_labels_discrete = []
#     off = 0
#     n = len(dataloader)
#     if earlystop is not None:
#         n = min(n,earlystop)        

#     # streams for parallelizing data transfer and computation
#     if device.type == 'cuda':
#         stream_compute = torch.cuda.Stream()
#         stream_transfer_input = torch.cuda.Stream()
#     else:
#         stream_compute = nullcontext()
#         stream_transfer_input = nullcontext()

#     # batch iterator
#     batch_iter = iter(dataloader)

#     # get the first example and put it on the device
#     example = next(batch_iter)
#     input = example['input'].to(device=device)
    
#     for i in tqdm.trange(n):


#         # just copying over the data
#         if debugcheat:
#             pred = {'continuous': example['labels'].clone(), 'discrete': example['labels_discrete'].clone(), 
#                     'todiscretize': example['labels_todiscretize'].clone()}
            
#             if not keepall:
#                 # only keep the last prediction
#                 if isinstance(pred, dict):
#                     pred = {k: v[:,[-1,]] for k, v in pred.items()}
#                 else:
#                     pred = pred[:,[-1,]]

            
#         else:
            
#             with torch.no_grad():
#                 if i + 1 < n:
#                     with torch.cuda.stream(stream_transfer_input):
#                         next_batch = next(batch_iter)
#                         next_input = next_batch['input'].to(device,non_blocking=True)
            
#                 with torch.cuda.stream(stream_compute):
#                     pred = model.output(input, mask=mask, is_causal=is_causal)
#                     if config['modelstatetype'] == 'prob':
#                         pred = model.maxpred(pred)
#                     elif config['modelstatetype'] == 'best':
#                         pred = model.randpred(pred)
                
#                     if not keepall:
#                         # only keep the last prediction
#                         if isinstance(pred, dict):
#                             pred = {k: v[:,[-1,]] for k, v in pred.items()}
#                         else:
#                             pred = pred[:,[-1,]]
            
#                     if isinstance(pred, dict):
#                         pred = {k: v.cpu() for k, v in pred.items()}
#                     else:
#                         pred = pred.cpu()

#         if all_pred is None:
#             # allocate
#             all_pred = allocate_batch_concat(pred, n)
#             labelidx = torch.zeros(n*example['idx'].shape[0], dtype=torch.int64)

#         # assign
#         all_pred,off1 = set_batch_concat(pred, all_pred, off)
#         labelidx[off:off1] = example['idx']
        
#         off = off1

#         torch.cuda.synchronize()
        
#         if i + 1 < n:
#             example = next_batch
#             input = next_input

#         # pred1 = dataset.get_full_pred(pred)
#         # labels1 = dataset.get_full_labels(example=example,use_todiscretize=True)
#         # pred_obj = FlyExample(example_in=pred,**example_params)

#         # example_obj = FlyExample(example_in=example, **example_params)
#         # label_obj = example_obj.labels
#         # pred_obj = label_obj.copy()
#         # pred_obj.erase_labels()
#         # pred_obj.set_prediction(pred)

#         # for i in range(np.prod(label_obj.pre_sz)):
#         #     all_pred.append(pred_obj.copy_subindex(idx_pre=i))
#         #     all_labels.append(label_obj.copy_subindex(idx_pre=i))

#         # if dataset.discretize:
#         #   all_pred_discrete.append(pred['discrete'])
#         #   all_labels_discrete.append(example['labels_discrete'])
#         # if 'mask' in example:
#         #   all_mask.append(example['mask'])
#     all_pred = clip_batch_concat(all_pred, off)
#     labelidx = labelidx[:off]
    
#     # get rid of the batch dimension if not keeping all
#     if not keepall:
#         if isinstance(all_pred, dict):
#             all_pred = {k: v.squeeze(1) for k, v in all_pred.items()}
#         else:
#             all_pred = all_pred.squeeze(1)
    
#     # create FlyLabels objects

#     return all_pred, labelidx  # ,all_mask,all_pred_discrete,all_labels_discrete

def pretile_datadict(datadict: dict, reps: int):
    """ Replicates all Data objects in a datadict by given number of times along a new first dimension.

    Args:
        datadict: Dictionary of Data objects to tile. Should have keys 'inputs' and 'labels', each containing a dict of Data objects.
        reps: Number of times to replicate
    """
    datadict = {k: v for k, v in datadict.items()}  # shallow copy

    for key1 in ['inputs','labels']:
        for key2 in datadict[key1].keys():
            datadict[key1][key2] = datadict[key1][key2][None].tile(reps)

    if 'useoutputmask' in datadict:
        datadict['useoutputmask'] = np.tile(datadict['useoutputmask'][None], (reps,)+ (1,)*datadict['useoutputmask'].ndim)

    return datadict

def copy_pred_to_example(example, pred, ts=None):
    """ Copy predictions to an example's labels at specified timepoints. Modifies example in place.

    Args:
        example: Example dict containing 'labels' and 'labels_discrete' ndarrays/tensors
        pred: Prediction dict containing 'continuous' and 'discrete' ndarrays/tensors
        ts: Timepoints to copy predictions to. 

    """
    if isinstance(ts, int):
        ts = [ts,]
    if 'continuous' in pred:
        if ts is None:
            example['labels'][...] = pred['continuous']
        else:
            example['labels'][...,ts,:] = pred['continuous']
    if 'discrete' in pred:
        if ts is None:
            example['labels_discrete'][...] = pred['discrete']
        else:
            newshape = list(example['labels_discrete'].shape)
            newshape[-2] = len(ts)
            example['labels_discrete'][...,ts,:] = pred['discrete'].reshape(newshape)
        
def predict_iterative(Xkp_true, burnin, agentspred, model, dataset, config, maxcontextl=np.inf, debugcheat=False,
                      labels_true=None, posestats=None, dampenconstant=0, prctilelim=None, dt=1, nsamples=1, extraargs={}):

    """
    predict_iterative(examples_pred,fliespred,scales,Xkp_fill,burnin,model,dataset,
                      maxcontextl=np.inf,debug=False,need_weights=False,nsamples=0,
                      labels_true=None)

    Args:
        examples_pred: list of FlyExample objects to be predicted in open loop. labels can be nan 
        for frames/flies to be predicted. Will be overwritten
        fliespred (ndarray, nfliespred): indices of flies to predict
        scales (ndarray, nscale x nfliespred): scale parameters for the flies to be predicted
        Xkp_fill (ndarray, nkpts x 2 x tpred x nflies): keypoints for all flies for all frames.
        Can be nan for frames/flies to be predicted. Will be overwritten.
        burnin (int): number of frames to use for initialization
        model (nn.Module): model to use for prediction.
        dataset (FlyDataset): dataset object used for prediction
        maxcontextl (int, optional): maximum number of frames to use for context. Default np.inf
        need_weights (bool, optional): whether to return attention weights. Default False
        nsamples (int, optional): number of samples to predict. If 0, only one sample is predicted
        and samples are collapsed. Default 0.
        labels_true (list of FlyPoseLabels objects, optional): true labels for debugging. Default None
    """


    # exampleobjs_pred = predict_iterative([data_example,], Xkp_fill, tpred, model, dataset, maxcontextl=contextl,
    #                                     debug=False, need_weights=False, nsamples=nsamples, **kwargs)
    
    model.eval()

    with torch.no_grad():
        w = next(iter(model.parameters()))
        dtype = w.cpu().numpy().dtype
        device = w.device

    # erase the fly we are predicting. Keep dt ground-truth frames past burnin
    # because label computation uses windows of dt frames into the past.
    Xkp_fill = Xkp_true.copy()
    Xkp_fill[agentspred,burnin+dt+1:] = np.nan
    
    # tile for multiple samples
    Xkp_fill = np.tile(Xkp_fill[None],[nsamples,]+[1,]*(Xkp_fill.ndim))
    extraargs = apf.utils.recursive_pretile(extraargs, nsamples)    

    nsamples = Xkp_fill.shape[0]
    ttotal = Xkp_fill.shape[2]
    tpred = ttotal - burnin - dt

    with torch.no_grad():
        w = next(iter(model.parameters()))
        device = w.device

    is_causal = True

    # global position of each fly in the previous frame, so that we don't have to integrate to compute position
    # TODO
    # pose_prev = []
    # for i in range(nagentspred):
    #     pose_curr = data_examples[i].labels.get_next_pose(ts=np.arange(burnin),use_todiscretize=True)[...,-1,:]
    #     pose_prev.append(pose_curr)
    
    # t is the last pose frame fed to the model; we predict frame t+1.
    # The Xkp window is dt frames longer than the model context because label
    # computation uses windows of dt frames into the past.
    for t in tqdm.trange(burnin+dt, burnin+tpred+dt-1):
        t0 = int(np.maximum(t - maxcontextl - dt, 0))
        duration = t - t0 + 1 - dt

        for i,agent in enumerate(agentspred):
            # copy frames up to t
            
            Xkp_curr = Xkp_fill[...,agent,t0:t+1,:,:]  # nsamples x duration x nkpts x 2
            extraargs_curr = apf.utils.recursive_subindex(extraargs, (slice(None), agent, slice(t0,t+1)))

            # start_frame=dt: the first dt frames of Xkp_curr supply the lookback
            # window for label computation, not the model context.
            example = dataset.rawdata_to_item(
                inputs={k: Xkp_curr for k in dataset.inputs.keys()},
                labels={k: Xkp_curr for k in dataset.labels.keys()},
                extraargs=extraargs_curr,
                start_frame=dt,
                duration=duration)
            example = ndarray_to_tensor(example)
            # dt+1 frames: computing the label at frame t+1 requires pose at t+1-dt.
            tsnext = slice(t+1-dt,t+2)
            extraargs_next = apf.utils.recursive_subindex(extraargs, (slice(None), agent, tsnext))

            if debugcheat:
                Xkp_next = np.tile(Xkp_true[None,agent,tsnext], [nsamples,1,1,1])

                pred = dataset.rawdata_to_item(labels={k: Xkp_next for k in dataset.labels.keys()},
                                               extraargs=extraargs_next)
                
            else:
                pred = predict(example,model,config,device=device,is_causal=is_causal)
            
            pred = pred_apply_fun(pred, lambda x: x[:, [-1,], ...].cpu() if type(x) is torch.Tensor else x)

            Xkp_next = Xkp_fill[...,agent,tsnext,:,:]
            example_next = dataset.rawdata_to_item(labels={k: Xkp_next for k in dataset.labels.keys()},
                                                    extraargs=extraargs_next)

            if (posestats is not None) and ((dampenconstant > 0) or (prctilelim is not None)):
                raise NotImplementedError("Regularization not implemented yet")
                pose = data_examples[i].labels.get_next_pose(ts=[t,],init_pose=pose_prev[i], use_todiscretize=True)
                # pose for frames [t-1,t]
                pose[...,-1,:] = regularize_pose(pose[...,-1,:],posestats,dampenconstant,prctilelim=prctilelim)
                data_examples[i].labels.set_next_pose(pose,ts=[t,])

            # set the label for frame t
            copy_pred_to_example(example_next, pred, -1)
            example_next_datadict = dataset.item_to_data(example_next)
            Xkp_next = apf.dataset.invert_to_named(example_next_datadict['labels']['velocity'],'original')
            Xkp_fill[...,agent,t+1,:,:] = Xkp_next[...,-1,:,:]

        # end loop over flies

    # end loop over time

    return Xkp_fill

def clear_predictions(example: dict, burnin: int):
    """Clear predictions in example dict after burnin frame by setting to nan."""

    exampleout = {k: v for k, v in example.items()}
    exampleout['input'][...,burnin+1:,:] = np.nan
    for key in ['labels','labels_discrete','continuous','discrete']:
        if not key in example:
            continue
        exampleout[key][...,burnin:,:] = np.nan
    return exampleout

def predict_iterative_all(dataset, model, config, tpred, track=None, extraargs=None, N=None, keepall=True, debugcheat=False, nsamples=0, labelidx=None, stride=None, **kwargs):

    if apf.dataset.isData(track):
        track = track.array
        
    # total number of frames we will crop out for each example
    contextl = dataset.context_length
    burnin = contextl - 1

    if stride is None:
        stride = tpred

    dataset.set_context_length(contextl+tpred-1)
    dataset.set_stride(stride)

    #example_params = dataset.get_flyexample_params()
    model.eval()
    
    if labelidx is not None:
        N = len(labelidx)
    elif N is not None and N < len(dataset):
        labelidx = np.random.choice(len(dataset),N,replace=False)
    else:
        labelidx = np.array(range(len(dataset)))
        N = len(dataset)

    Xkp_pred_all = []
    
    dt = 1
    
    for i,examplei in tqdm.tqdm(enumerate(labelidx),total=N):

        t0, agentnum = dataset.chunk_indices[examplei]
        # predict_iterative writes predicted frames [burnin+dt+1, burnin+tpred+dt-1]
        # into Xkp_fill, so we slice t0 through t0+burnin+tpred+dt from the track.
        tslice = slice(t0,t0+burnin+tpred+dt)

        if extraargs is not None:
            extraargscurr = apf.utils.recursive_subindex(extraargs, (slice(None),tslice))
        else:
            extraargscurr = None
            
        if track is None:
            example = dataset.get_example_datadict(examplei)
            track = apf.dataset.invert_to_named(example['labels']['velocity'], 'original')
            invertdata = track.get_invertdata_dict()
            if extraargs is None:
                extraargscurr = invertdata
            else:
                # user-supplied extraargs override auto-derived invertdata per op-name key.
                extraargscurr = {k: invertdata.get(k,{}) | extraargscurr.get(k,{}) for k in set(invertdata.keys()).union(set(extraargscurr.keys()))}
            track = track.array
        else:
            # get positions of all agents    
            Xkp_true = track[:,tslice].copy() 
            
        # flyidscurr = flyids[:,t0:t0+contextl+tpred+dt]
        # assert np.all(flyidscurr[agentnum,0] == flyidscurr[agentnum,:])

        Xkp_pred = predict_iterative(Xkp_true, burnin, [agentnum,], model, dataset, config, maxcontextl=contextl,
                                    debugcheat=False, dt=dt, nsamples=nsamples, extraargs=extraargscurr, **kwargs)

        Xkp_pred_all.append(Xkp_pred)
        
    dataset.set_context_length(contextl)
    dataset.set_stride()

    return Xkp_pred_all, labelidx  # ,all_mask,all_pred_discrete,all_labels_discrete

# def get_global_predictions(all_pred,labelidx,dataset):

#     for i,idx in tqdm.tqdm(enumerate(labelidx),total=len(labelidx)):
#         labelobj = dataset.get_example(idx).labels
#         unz_global_label = labelobj.get_future_global(zscored=False, use_todiscretize=True)
#         if i == 0:
#             unz_glabelsv = np.zeros((len(labelidx),) + unz_global_label.shape, dtype=unz_global_label.dtype)
#             unz_gpredv = np.zeros((len(labelidx),) + unz_global_label.shape, dtype=unz_global_label.dtype)
#         unz_glabelsv[i] = unz_global_label
#         predobj = labelobj.copy()
#         predobj.erase_labels()
#         if type(all_pred) is dict:
#             predcurr = {k: v[i] for k,v in all_pred.items()}
#         else:
#             predcurr = all_pred[i]
#         predobj.set_prediction(predcurr)
#         unz_global_pred = predobj.get_future_global(zscored=False, use_todiscretize=False)
#         unz_gpredv[i] = unz_global_pred
        
#     return unz_gpredv,unz_glabelsv

def compute_prediction_errors(all_pred,labelidx,dataset):
    # compare predictions to labels

    # pred_data is a list of FlyExample objects
    pred_data,true_data = dataset.create_data_from_pred(all_pred, labelidx)

    # compute error in various ways
    err_example = []
    for pred_example,true_example in zip(pred_data,true_data):
        errcurr = pred_example.compute_error(true_example=true_example,pred_example=pred_example)
        err_example.append(errcurr)
        
    keysmean = ['l1_multi','mse_multi','l1_multi_samplemean','l1_multi_samplemin',
                'mse_multi_samplemean','mse_multi_samplemin','ce_discrete_mean',
                'l2_err_kp_mean']
    # compute mean
    meanerr = {}
    n = np.sum([errcurr['n'] for errcurr in err_example]).item()
    for k in keysmean:
        meanerr[k] = 0.
        for errcurr in err_example:
            meanerr[k] += errcurr[k]*errcurr['n']/n
            
    true_labels = pred_example[0].labels
    idx_multiglobal_to_multi = true_labels._idx_multiglobal_to_multi
    errcurr['l1_multi'][idx_multiglobal_to_multi]
        
    return meanerr,err_example


def hist_predictions(all_pred,labelidx,dataset,binedges=None,nbins=50):

    example = dataset.get_example(0)
    labelobj = example.labels
    d_output = dataset.d_output
    if binedges is None:
        if dataset.discretize:
            nbins = dataset.discretize_nbins
        binedges = np.zeros((d_output,nbins+1),dtype=dataset.dtype)
        binedges[:] = np.nan
        if dataset.discretize:
            binedges_discrete = dataset.get_bin_edges()
            binedges[labelobj._idx_multidiscrete_to_multi] = binedges_discrete
        if dataset.continuous:
            minv = np.zeros(labelobj.d_multicontinuous,dtype=dataset.dtype)
            minv[:] = np.inf
            maxv = np.zeros(labelobj.d_multicontinuous,dtype=dataset.dtype)
            maxv[:] = -np.inf
            
            for i in range(len(dataset)):
                example = dataset.get_example(i)
                labels_continuous = example.labels.get_multi_continuous(makecopy=False,zscored=False)
                minv = np.minimum(minv,np.nanmin(labels_continuous,axis=0))
                maxv = np.maximum(maxv,np.nanmax(labels_continuous,axis=0))
            binedges_continuous = np.linspace(minv,maxv,nbins+1).T
            binedges[labelobj._idx_multicontinuous_to_multi,:] = binedges_continuous
            
    labelcounts = 0
    predcounts = 0
    labeln = 0
    predn = 0
    for i in range(len(dataset)):
        idx = labelidx[i]
        labelobj = dataset.get_example(i).labels
        labelcurr = labelobj.get_multi(zscored=False,use_todiscretize=True)
        for j in range(d_output):
            countscurr = np.histogram(labelcurr[:,j],bins=binedges[j])
        predobj = labelobj.copy()
        predobj.erase_labels()
        predobj.set_prediction(all_pred[i])


#     example = dataset.get_example(0)
#     labelobj = example.labels
#     d_output = dataset.d_output
#     if binedges is None:
#         if dataset.discretize:
#             nbins = dataset.discretize_nbins
#         binedges = np.zeros((d_output,nbins+1),dtype=dataset.dtype)
#         binedges_discrete = dataset.get_bin_edges()
    
#     return
