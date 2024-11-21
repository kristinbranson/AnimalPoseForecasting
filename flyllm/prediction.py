import numpy as np
import torch

import tqdm
from apf.utils import set_batch_concat, allocate_batch_concat, clip_batch_concat
from apf.models import ( 
    get_output_and_attention_weights,
    pred_apply_fun
)
from flyllm.pose import FlyExample
from contextlib import nullcontext
import datetime

def predict_all(dataloader=None, dataset=None, model=None, config=None, mask=None, keepall=True, earlystop=None, debugcheat=False,
                savepredfile=None,saveinterval=600,nkeep=None,batchsize=None,shuffle=False,skipinterval=None):

    # get dataset from dataloader
    if dataset is None:
        assert dataloader is not None
        dataset = dataloader.dataset        

    assert model is not None
    assert config is not None
    assert mask is not None
        
    if batchsize is None:
        batchsize = config['test_batch_size']
        
    if dataloader is None:
        
        if skipinterval is not None:
            if not keepall and (nkeep is None):
                nkeep = skipinterval
            sampler = np.arange(0,len(dataset),skipinterval)
        else:
            sampler = None

        # create a dataloader
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=batchsize, 
                                                 shuffle=shuffle,
                                                 pin_memory=dataset.cudaoptimize==False,
                                                 sampler=sampler)

    is_causal = dataset.ismasked() == False

    if not keepall:
        assert nkeep is not None

    with torch.no_grad():
        w = next(iter(model.parameters()))
        device = w.device

    #example_params = dataset.get_flyexample_params()
    model.eval()
    dataset.set_eval_mode()

    # compute predictions and labels for all validation data using default masking
    all_pred = None
    labelidx = None
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
            
        if not keepall:
            # only keep the last nkeep predictions
            if isinstance(pred, dict):
                pred = {k: v[:,-nkeep:] for k, v in pred.items()}
            else:
                pred = pred[:,-nkeep:]

        if isinstance(pred, dict):
            pred = {k: v.cpu() for k, v in pred.items()}
        else:
            pred = pred.cpu()

        if all_pred is None:
            # allocate
            all_pred = allocate_batch_concat(pred, n)
            labelidx = torch.zeros(n*example['idx'].shape[0], dtype=torch.int64)

        # assign
        all_pred,off1 = set_batch_concat(pred, all_pred, off)
        labelidx[off:off1] = example['idx']
        
        off = off1

        timenow = datetime.datetime.now()
        if savepredfile is not None and (timenow - lastsavetime).total_seconds() > saveinterval:
            print(f"Saving {off} predictions to {savepredfile}")
            savestuff = {'all_pred': clip_batch_concat(all_pred, off), 'labelidx': labelidx[:off], 'i': i}
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
    labelidx = labelidx[:off]
    
    # all_pred[...] is (nbatches * batchsize) x nkeep x d
    
    # get rid of the batch dimension if nkeep == skipinterval
    if not keepall and (nkeep == 1):
        if isinstance(all_pred, dict):
            all_pred = {k: v.squeeze(1) for k, v in all_pred.items()}
        else:
            all_pred = all_pred.squeeze(1)
    
    # create FlyLabels objects

    return all_pred, labelidx  # ,all_mask,all_pred_discrete,all_labels_discrete

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


def predict_iterative(examples_pred, fliespred, scales, Xkp_fill, burnin, model, dataset, maxcontextl=np.inf, debug=False,
                               need_weights=False, nsamples=0, labels_true=None):

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
    
    model.eval()

    with torch.no_grad():
        w = next(iter(model.parameters()))
        dtype = w.cpu().numpy().dtype
        device = w.device

    if nsamples > 0:
        for example_pred in examples_pred:
            example_pred.pre_tile(nsamples)

    tpred = examples_pred[0].ntimepoints
    nfliespred = len(examples_pred)
    if need_weights:
        attn_weights = [None, ] * tpred

    if dataset.ismasked():
        # to do: figure this out for flattened models
        masktype = 'last'
        dummy = np.zeros((1, dataset.d_output))
        dummy[:] = np.nan
    else:
        masktype = None

    # start predicting motion from frame burnin-1 to burnin = t
    masksizeprev = None
    
    # global position of each fly in the previous frame, so that we don't have to integrate to compute position
    pose_prev = []
    for i in range(nfliespred):
        pose_curr = examples_pred[i].labels.get_next_pose(ts=np.arange(burnin),use_todiscretize=True)[...,-1,:]
        pose_prev.append(pose_curr)
    
    for t in tqdm.trange(burnin, tpred): 
        t0 = int(np.maximum(t - maxcontextl, 0))

        for i,fly in enumerate(fliespred):
            # copy frames up to t
            # don't use the init_pose
            # get_next_pose[:,-1] will be nan
            example_pred = examples_pred[i].copy_subindex(ts=np.arange(t0, t+1),needinit=False)
            # inputs will go from t0 through t
            # labels (unused) will go from t0+example_pred.starttoff through t+example_pred.starttoff
            test_example = example_pred.get_train_example()
            xcurr = test_example['input']
            assert not torch.any(torch.isnan(xcurr))
            xcurr, _, _ = dataset.mask_input(xcurr, masktype)
            if nsamples == 0:
                xcurr = xcurr[None, ...]

            if debug:
                label_pred = labels_true[i].copy_subindex(ts=np.arange(t0, t+1))
                pred = label_pred.get_train_labels()
                pred = {k: v.reshape((1,) + v.shape) if type(v) is torch.Tensor else v for k, v in pred.items()}
                #zmovementout = np.tile(self.zscore_labels(movement_true[t - 1, :, i]).astype(dtype)[None],
                #                       (nsamples1, 1))
            else:

                if dataset.flatten:
                    raise NotImplementedError("Flattening not yet implemented")
                    # not implemented yet
                    # to do: not sure if multiple samples here works

                    zmovementout = np.zeros((nsamples1, dataset.d_output), dtype=dtype)
                    zmovementout_flattened = np.zeros((dataset.noutput_tokens_per_timepoint, dataset.flatten_max_doutput),
                                                        dtype=dtype)

                    for token in range(dataset.noutput_tokens_per_timepoint):

                        lastidx = xcurr.shape[0] - dataset.noutput_tokens_per_timepoint
                        masksize = lastidx + token
                        net_mask, is_causal = dataset.get_predict_mask(masksize=masksize, device=device)

                        with torch.no_grad():
                            predtoken = model(xcurr[None, :lastidx + token, ...].to(device), mask=net_mask,
                                                is_causal=is_causal)
                        # to-do: integrate with labels object
                        if token < len(dataset.discreteidx):
                            # sample
                            sampleprob = torch.softmax(predtoken[0, -1, :dataset.discretize_nbins], dim=-1)
                            binnum = int(weighted_sample(sampleprob, nsamples=nsamples1))

                            # store in input
                            xcurr[lastidx + token, binnum[0]] = 1.
                            zmovementout_flattened[token, binnum[0]] = 1.

                            # convert to continuous
                            nsamples_per_bin = dataset.discretize_bin_samples.shape[0]
                            sample = int(torch.randint(low=0, high=nsamples_per_bin, size=(nsamples,)))
                            zmovementcurr = dataset.discretize_bin_samples[sample, token, binnum]

                            # store in output
                            zmovementout[:, dataset.discreteidx[token]] = zmovementcurr
                        else:  # else token < len(self.discreteidx)
                            # continuous
                            zmovementout[:, dataset.continuous_idx] = predtoken[0, -1, :len(dataset.continuous_idx)].cpu()
                            zmovementout_flattened[token, :len(dataset.continuous_idx)] = zmovementout[
                                dataset.continuous_idx, 0]

                else:  # else flatten

                    masksize = t - t0
                    if masksize != masksizeprev:
                        net_mask, is_causal = dataset.get_predict_mask(masksize=masksize, device=device)
                        masksizeprev = masksize

                    if need_weights:
                        with torch.no_grad():
                            pred, attn_weights_curr = get_output_and_attention_weights(model,
                                                                                        xcurr.to(device),
                                                                                        net_mask)
                        # dimensions correspond to layer, output frame, input frame
                        attn_weights_curr = torch.cat(attn_weights_curr, dim=0).cpu().numpy()
                        if i == 0:
                            attn_weights[t] = np.tile(attn_weights_curr[..., None], (1, 1, 1, nfliespred))
                            attn_weights[t][..., 1:] = np.nan
                        else:
                            attn_weights[t][..., i] = attn_weights_curr
                    else:
                        with torch.no_grad():
                            # predict for all frames
                            # masked: movement from 0->1, ..., t->t+1
                            # causal: movement from 1->2, ..., t->t+1
                            # last prediction: t->t+1
                            pred = model.output(xcurr.to(device), mask=net_mask, is_causal=is_causal)
                    # to-do: this is not incorportated into sampling, probably should be
                    if model.model_type == 'TransformerBestState' or model.model_type == 'TransformerState':
                        pred = model.randpred(pred)
                    # z-scored movement from t to t+1

                # end else flatten
            # end else debug
            
            if nsamples == 0:
                pred = pred_apply_fun(pred, lambda x: x[0, [-1,], ...].cpu().numpy() if type(x) is torch.Tensor else x)
            else:
                pred = pred_apply_fun(pred, lambda x: x[:, [-1,], ...].cpu().numpy() if type(x) is torch.Tensor else x)
                
            # set the label for frame t, but not the inputs yet
            examples_pred[i].labels.set_prediction(pred,ts=t)                
            
            # store keypoints predicted for this frame
            Xkpcurr = examples_pred[i].labels.get_next_keypoints(ts=[t,],init_pose=pose_prev[i])
            Xkp_fill[...,:,:,t+1,fly] = Xkpcurr[...,-1,:,:]


            #globapos_curr = examples_pred[i].labels.get_next_pose_global(ts=[t,],globalpos0=globalpos_prev[i])
            pose_curr = examples_pred[i].labels.get_next_pose(ts=[t,],init_pose=pose_prev[i])
            pose_prev[i] = pose_curr[...,-1,:]
            #globalpos_prev[i] = globapos_curr
                
        # end loop over flies
        
        if t < tpred-1:
            # update observations for the next frame
            for i,fly in enumerate(fliespred):
                # this is just one frame of inputs, so don't crop the end
                examples_pred[i].inputs.set_inputs_from_keypoints(Xkp_fill[...,:,:,[t+1,],:],fly,scale=scales[i],ts=[t+1,],npad=0)

    if need_weights:
        return examples_pred, attn_weights
    else:
        return examples_pred


def predict_iterative_all(rawdata, dataset, model, tpred, N=None, keepall=True, debugcheat=False, nsamples=0):

    # total number of frames we will crop out for each example
    ttotal = dataset.contextl
    burnin = ttotal - tpred

    #example_params = dataset.get_flyexample_params()
    model.eval()
    dataset.set_eval_mode()
    
    if N is not None and N < len(dataset):
        labelidx = np.random.choice(len(dataset),N,replace=False)
    else:
        labelidx = np.array(range(len(dataset)))
        N = len(dataset)

    all_pred = []
            
    for i,examplei in tqdm.tqdm(enumerate(labelidx),total=N):

        example = dataset[examplei]
        exampleobj = FlyExample(example_in=example,dataset=dataset)

        if debugcheat:
            exampleobj_pred = exampleobj
        else:
            exampleobj.labels.erase_labels(ts=slice(burnin+1,None))

            metadata = exampleobj.get_metadata()
            scale = exampleobj.labels.get_scale()
            agentnum = metadata['flynum']
            t0 = metadata['t0']

            # get positions of all agents
            Xkp_true = rawdata['X'][...,t0:t0+ttotal+1,:].copy()
            # nan out the fly we are predicting
            Xkp_fill = Xkp_true.copy()
            Xkp_fill[...,burnin+1:,agentnum] = np.nan
            if nsamples > 0:
                Xkp_fill = np.tile(Xkp_fill[None],(nsamples,1,1,1,1))

            exampleobjs_pred = predict_iterative([exampleobj,], [agentnum,], [scale,], Xkp_fill, burnin, model, dataset, maxcontextl=burnin,
                                                        debug=False, need_weights=False, nsamples=nsamples)

            exampleobj_pred = exampleobjs_pred[0]
        all_pred.append(exampleobj_pred)

    return all_pred, labelidx  # ,all_mask,all_pred_discrete,all_labels_discrete

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