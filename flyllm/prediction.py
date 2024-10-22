import numpy as np
import torch

from flyllm.pose import FlyExample
import tqdm
from apf.utils import set_batch_concat, allocate_batch_concat, clip_batch_concat


def predict_all(dataloader, dataset, model, config, mask, keepall=True, earlystop=None, debugcheat=False):
    is_causal = dataset.ismasked() == False

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
                pred = model.output(example['input'].to(device=device), mask=mask, is_causal=is_causal)
                if config['modelstatetype'] == 'prob':
                    pred = model.maxpred(pred)
                elif config['modelstatetype'] == 'best':
                    pred = model.randpred(pred)
            
        if not keepall:
            # only keep the last prediction
            if isinstance(pred, dict):
                pred = {k: v[:,[-1,]] for k, v in pred.items()}
            else:
                pred = pred[:,[-1,]]

        if isinstance(pred, dict):
            pred = {k: v.cpu() for k, v in pred.items()}
        else:
            pred = pred.cpu()

        if all_pred is None:
            # allocate
            all_pred = allocate_batch_concat(pred, len(dataloader))
            labelidx = torch.zeros(len(dataloader)*example['idx'].shape[0], dtype=torch.int64)

        # assign
        all_pred,off1 = set_batch_concat(pred, all_pred, off)
        labelidx[off:off1] = example['idx']
        
        off = off1

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
    
    # get rid of the batch dimension if not keeping all
    if not keepall:
        if isinstance(all_pred, dict):
            all_pred = {k: v.squeeze(1) for k, v in all_pred.items()}
        else:
            all_pred = all_pred.squeeze(1)
    
    # create FlyLabels objects

    return all_pred, labelidx  # ,all_mask,all_pred_discrete,all_labels_discrete

def get_global_predictions(all_pred,labelidx,dataset):

    for i,idx in tqdm.tqdm(enumerate(labelidx),total=len(labelidx)):
        labelobj = dataset.get_example(idx).labels
        unz_global_label = labelobj.get_future_global(zscored=False, use_todiscretize=True)
        if i == 0:
            unz_glabelsv = np.zeros((len(labelidx),) + unz_global_label.shape, dtype=unz_global_label.dtype)
            unz_gpredv = np.zeros((len(labelidx),) + unz_global_label.shape, dtype=unz_global_label.dtype)
        unz_glabelsv[i] = unz_global_label
        predobj = labelobj.copy()
        predobj.erase_labels()
        if type(all_pred) is dict:
            predcurr = {k: v[i] for k,v in all_pred.items()}
        else:
            predcurr = all_pred[i]
        predobj.set_prediction(predcurr)
        unz_global_pred = predobj.get_future_global(zscored=False, use_todiscretize=False)
        unz_gpredv[i] = unz_global_pred
        
    return unz_gpredv,unz_glabelsv

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