# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Imports

# +
import numpy as np
import torch
import matplotlib.pyplot as plt
import transformers
import tqdm
import datetime
import os
from matplotlib import animation

from flyllm.io import read_config, get_modeltype_str
from flyllm.utils import get_dct_matrix, compute_npad
from flyllm.config import scalenames, nfeatures, featglobal
from flyllm.features import compute_features, compute_scale_perfly
from flyllm.data import load_and_filter_data, sanity_check_tspred, chunk_data, interval_all, debug_less_data
from flyllm.dataset import FlyMLMDataset
from flyllm.plotting import (
    initialize_debug_plots, 
    initialize_loss_plots, 
    update_debug_plots,
    debug_plot_global_histograms, 
    debug_plot_dct_relative_error, 
    debug_plot_global_error, 
    debug_plot_predictions_vs_labels,
    select_featidx_plot,
)
from flyllm.models import (
    initialize_model, 
    initialize_loss, 
    generate_square_full_mask, 
    sanity_check_temporal_dep,
    predict_all,
    criterion_wrapper,
    stack_batch_list,
)
from flyllm.simulation import animate_predict_open_loop
# -

torch.cuda.is_available()

# ## Load data

# +
# configuration parameters for this model
loadmodelfile = None
restartmodelfile = None
config = read_config("/groups/branson/home/eyjolfsdottire/code/MABe2022/config_fly_llm_multitimeglob_discrete_20230907.json")

print(f"batch size = {config['batch_size']}")

# seed the random number generators
np.random.seed(config['numpy_seed'])
torch.manual_seed(config['torch_seed'])

# set device (cuda/cpu)
device = torch.device(config['device'])
# -

# Skip augmentation for debugging purposes
config['augment_flip'] = False 

config['intrainfile']

# load raw data
data, scale_perfly = load_and_filter_data(config['intrainfile'], config)
valdata, val_scale_perfly = load_and_filter_data(config['invalfile'], config)

# for debugging, use only a subset of the data
max_frames = config['contextl'] * 100
for data in [data, valdata]:
    debug_less_data(data)

# +
print(config['contextl'])
print(config['tspred_global'])  # times to look ahead

print(data.keys())
print(data['X'].shape)  # n_kpts x n_dim x n_frames x n_flies
print(data['y'].shape)  # n_categories x n_frames x n_flies
print(data['ids'].max())  # 907: why is this so high? Is it because it adds a + i*12 for each new video? Yep that's it

print(scale_perfly.shape)  # len(scalenames) x max_n_flies

scalenames
# # compute_scale_perfly??
# # compute_npad??
# # get_dct_matrix??
# # chunk_data??
# -

# ## Compute features

# +
# # compute_features??

# +
# if using discrete cosine transform, create dct matrix
# this didn't seem to work well, so probably won't use in the future
if config['dct_tau'] is not None and config['dct_tau'] > 0:
    dct_m, idct_m = get_dct_matrix(config['dct_tau'])
    # this gives the maximum of 
    #   a) max number of frames to lookahead or 
    #   b) dct_tau (number of timepoints for cosine transform)
else:
    dct_m = None
    idct_m = None

# how much to pad outputs by -- depends on how many frames into the future we will predict
npad = compute_npad(config['tspred_global'], dct_m)
chunk_data_params = {'npad': npad}

compute_feature_params = {
 "simplify_out": config['simplify_out'],
 "simplify_in": config['simplify_in'],
 "dct_m": dct_m,
 "tspred_global": config['tspred_global'],
 "compute_pose_vel": config['compute_pose_vel'],
 "discreteidx": config['discreteidx'],
}

# function for computing features
reparamfun = lambda x, id, flynum, **kwargs: compute_features(
    x, id, flynum, scale_perfly, outtype=np.float32, **compute_feature_params, **kwargs)

val_reparamfun = lambda x, id, flynum, **kwargs: compute_features(
    x, id, flynum, val_scale_perfly, outtype=np.float32, **compute_feature_params, **kwargs)

# sanity check on computing features when predicting many frames into the future
sanity_check_tspred(
    data, 
    compute_feature_params,
    npad,
    scale_perfly,
    contextl=config['contextl'],
    t0=510,
    flynum=0
)

# chunk the data if we didn't load the pre-chunked cache file
print('Chunking training data...')
X = chunk_data(data, config['contextl'], reparamfun, **chunk_data_params)
print('Chunking val data...')
valX = chunk_data(valdata, config['contextl'], val_reparamfun, **chunk_data_params)
print('Done.')
# -

print(len(X))  # batches
print(X[0].keys())
print(X[0]['input'].shape)  # contextl x n_features ?
print(X[0]['labels'].shape)  # contextl x n_pred_features ?
print(X[0]['scale'].shape)  # len(scalenames)
X[0]['metadata']

# ## Create dataloader

# +
dataset_params = {
    'max_mask_length': config['max_mask_length'],
    'pmask': config['pmask'],
    'masktype': config['masktype'],
    'simplify_out': config['simplify_out'],
    'pdropout_past': config['pdropout_past'],
    'input_labels': config['input_labels'],
    'dozscore': True,
    'discreteidx': config['discreteidx'],
    'discretize_nbins': config['discretize_nbins'],
    'discretize_epsilon': config['discretize_epsilon'],
    'flatten_labels': config['flatten_labels'],
    'flatten_obs_idx': config['flatten_obs_idx'],
    'flatten_do_separate_inputs': config['flatten_do_separate_inputs'],
    'p_add_input_noise': config['p_add_input_noise'],
    'dct_ms': (dct_m,idct_m),
    'tspred_global': config['tspred_global'],
    'discrete_tspred': config['discrete_tspred'],
    'compute_pose_vel': config['compute_pose_vel'],
}
train_dataset_params = {
    'input_noise_sigma': config['input_noise_sigma'],
}

print('Creating training data set...')
train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)
print('Done.')

# zscore and discretize parameters for validation data set based on train data
# we will continue to use these each time we rechunk the data
dataset_params['zscore_params'] = train_dataset.get_zscore_params()
dataset_params['discretize_params'] = train_dataset.get_discretize_params()

print('Creating validation data set...')
val_dataset = FlyMLMDataset(valX,**dataset_params)
print('Done.')

# get properties of examples from the first training example
example = train_dataset[0]
d_input = example['input'].shape[-1]
d_output = train_dataset.d_output
outnames = train_dataset.get_outnames()

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=True,
                                                pin_memory=True,
                                                )
ntrain = len(train_dataloader)

val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=config['batch_size'],
                                              shuffle=False,
                                              pin_memory=True,
                                              )
nval = len(val_dataloader)

example = next(iter(train_dataloader))
sz = example['input'].shape
print(f'batch input shape = {sz}')

# +
# X has dim 211, but train data has dim 241, where does the extra 30 come from?
print(len(train_dataset))  # same as x
print(ntrain)
print(len(X) / config['batch_size'])
print(type(train_dataset))  # torch.utils.data.Dataset
print(d_input, d_output)
# train_dataset.data[0].shape
# outnames

print(train_dataset.dfeat)
print(train_dataset.nextframeidx)
# -

# set up debug plots
plt.ion()
debug_params = {}
# if contextl is long, still just look at samples from the first 64 frames
if config['contextl'] > 64:
    debug_params['tsplot'] = np.round(np.linspace(0,64,5)).astype(int)
    debug_params['traj_nsamplesplot'] = 1
hdebug = {}
hdebug['train'] = initialize_debug_plots(train_dataset, train_dataloader, data,name='Train', **debug_params)
hdebug['val'] = initialize_debug_plots(val_dataset, val_dataloader, valdata, name='Val', **debug_params)

# ## Set up model and training

# Smaller model for debuggin purposes
config['nlayers'] = 2
config['niterplot'] = 2

# +
# create the model
model, criterion = initialize_model(config, train_dataset, device)

# optimizer
num_training_steps = config['num_train_epochs'] * ntrain
optimizer = transformers.optimization.AdamW(model.parameters(), **config['optimizer_args'])
lr_scheduler = transformers.get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


# initialize structure to keep track of loss
loss_epoch = initialize_loss(train_dataset, config)
last_val_loss = None

hloss = initialize_loss_plots(loss_epoch)

# +
print(type(model))  # torch.nn.Module
# criterion??

# lossfcn_discrete = torch.nn.CrossEntropyLoss()
# lossfcn_continuous = torch.nn.L1Loss()
# -

# ## Create attention mask 
# (e.g. mask out t+1, .. t+n)

# +
# create attention mask
contextl = example['input'].shape[1]
if config['modeltype'] == 'mlm':
    train_src_mask = generate_square_full_mask(contextl).to(device)
    is_causal = False
elif config['modeltype'] == 'clm':
    train_src_mask = torch.nn.Transformer.generate_square_subsequent_mask(contextl, device=device)
    is_causal = True
    #train_src_mask = generate_square_subsequent_mask(contextl).to(device)
else:
    raise

# sanity check on temporal dependences
sanity_check_temporal_dep(train_dataloader, device, train_src_mask, is_causal, model, tmess=300)

modeltype_str = get_modeltype_str(config, train_dataset)
if ('model_nickname' in config) and (config['model_nickname'] is not None):
    modeltype_str = config['model_nickname']
# -

m = train_src_mask.cpu()
plt.figure()
plt.imshow(m)
plt.show()
# -Inf at future timesteps, otherwise 0

# # Train the model

# +
# train
epoch = 0
progress_bar = tqdm.tqdm(range(num_training_steps))

savetime = datetime.datetime.now()
savetime = savetime.strftime('%Y%m%dT%H%M%S')
ntimepoints_per_batch = train_dataset.ntimepoints
valexample = next(iter(val_dataloader))

for epoch in range(epoch, config['num_train_epochs']):
      
    model.train()
    tr_loss = torch.tensor(0.0).to(device)
    if train_dataset.discretize:
        tr_loss_discrete = torch.tensor(0.0).to(device)
        tr_loss_continuous = torch.tensor(0.0).to(device)
    
    nmask_train = 0
    for step, example in enumerate(train_dataloader):
    
        pred = model(example['input'].to(device=device), mask=train_src_mask, is_causal=is_causal)
        loss, loss_discrete, loss_continuous = criterion_wrapper(example, pred, criterion, train_dataset, config)
          
        loss.backward()
        
        # how many timepoints are in this batch for normalization
        if config['modeltype'] == 'mlm':
            nmask_train += torch.count_nonzero(example['mask'])
        else:
            nmask_train += example['input'].shape[0]*ntimepoints_per_batch 
    
        if step % config['niterplot'] == 0:
          
            with torch.no_grad():
                trainpred = model.output(example['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
                valpred = model.output(valexample['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
            update_debug_plots(hdebug['train'],config,model,train_dataset,example,trainpred,name='Train',criterion=criterion,**debug_params)
            update_debug_plots(hdebug['val'],config,model,val_dataset,valexample,valpred,name='Val',criterion=criterion,**debug_params)
            plt.show()
            plt.pause(.1)
    
        tr_loss_step = loss.detach()
        tr_loss += tr_loss_step
        if train_dataset.discretize:
            tr_loss_discrete += loss_discrete.detach()
            tr_loss_continuous += loss_continuous.detach()
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
        optimizer.step()
        lr_scheduler.step()
        model.zero_grad()
        
        # update progress bar
        stat = {'train loss': tr_loss.item()/nmask_train,'last val loss': last_val_loss,'epoch': epoch}
        if train_dataset.discretize:
            stat['train loss discrete'] = tr_loss_discrete.item()/nmask_train
            stat['train loss continuous'] = tr_loss_continuous.item()/nmask_train
        progress_bar.set_postfix(stat)
        progress_bar.update(1)
        
        # end of iteration loop

        break ## DEBUG

    break ## DEBUG
    
    # training epoch complete
    loss_epoch['train'][epoch] = tr_loss.item() / nmask_train
    if train_dataset.discretize:
        loss_epoch['train_discrete'][epoch] = tr_loss_discrete.item() / nmask_train
        loss_epoch['train_continuous'][epoch] = tr_loss_continuous.item() / nmask_train
    
    # compute validation loss after this epoch
    if val_dataset.discretize:
         loss_epoch['val'][epoch],loss_epoch['val_discrete'][epoch],loss_epoch['val_continuous'][epoch] = \
           compute_loss(model,val_dataloader,val_dataset,device,train_src_mask,criterion,config)
    else:
        loss_epoch['val'][epoch] = \
          compute_loss(model,val_dataloader,val_dataset,device,train_src_mask,criterion,config)
    last_val_loss = loss_epoch['val'][epoch].item()
    
    update_loss_plots(hloss, loss_epoch)
    plt.show()
    plt.pause(.1)
    
    # rechunk the training data
    if np.mod(epoch+1,config['epochs_rechunk']) == 0:
        print(f'Rechunking data after epoch {epoch}')
        X = chunk_data(data,config['contextl'],reparamfun,**chunk_data_params)
      
        train_dataset = FlyMLMDataset(X,**train_dataset_params,**dataset_params)
        print('New training data set created')
    
    if (epoch+1)%config['save_epoch'] == 0:
        savefile = os.path.join(config['savedir'],f"fly{modeltype_str}_epoch{epoch+1}_{savetime}.pth")
        print(f'Saving to file {savefile}')
        save_model(savefile,model,lr_optimizer=optimizer,scheduler=lr_scheduler,loss=loss_epoch,config=config)

print('Done training')
# -

# # Evaluate

# +
model.eval()

# compute predictions and labels for all validation data using default masking
all_pred, all_labels = predict_all(
    val_dataloader, val_dataset, model, config, train_src_mask
)

# +
# # plot comparison between predictions and labels on validation data
# predv = stack_batch_list(all_pred)
# labelsv = stack_batch_list(all_labels)
# maskv = stack_batch_list(all_mask)
# pred_discretev = stack_batch_list(all_pred_discrete)
# labels_discretev = stack_batch_list(all_labels_discrete)

# +
fig, ax = debug_plot_global_histograms(all_pred, all_labels, train_dataset, nbins=25, subsample=1, compare='pred')

if train_dataset.dct_m is not None:
    debug_plot_dct_relative_error(all_pred, all_labels, train_dataset)
if train_dataset.ntspred_global > 1:
    debug_plot_global_error(all_pred, all_labels, train_dataset)

# crop to nplot for plotting
nplot = min(len(all_labels),8000//config['batch_size']//config['contextl']+1)
# -

ntspred_plot = np.minimum(4, train_dataset.ntspred_global)
featidxplot, ftplot = all_labels[0].select_featidx_plot(ntspred_plot)
naxc = np.maximum(1, int(np.round(len(featidxplot) / nfeatures)))
fig, ax = debug_plot_predictions_vs_labels(
    all_pred[:nplot], all_labels[:nplot], naxc=naxc, featidxplot=featidxplot
)
if train_dataset.ntspred_global > 1:
    featidxplot, ftplot = all_labels[0].select_featidx_plot(
        ntsplot=train_dataset.ntspred_global, ntsplot_relative=0
    )
    naxc = np.maximum(1, int(np.round(len(featidxplot) / nfeatures)))
    fig, ax = debug_plot_predictions_vs_labels(
        all_pred[:nplot], all_labels[:nplot], naxc=naxc, featidxplot=featidxplot
    )
    featidxplot, _ = all_labels[0].select_featidx_plot(ntsplot=1, ntsplot_relative=1)
    fig, ax = debug_plot_predictions_vs_labels(
        all_pred[:nplot], all_labels[:nplot], naxc=naxc, featidxplot=featidxplot
    )

# # Simulate

# +
# generate an animation of open loop prediction
tpred = np.minimum(2000 + config['contextl'], valdata['isdata'].shape[0] // 2)
print(tpred)

# all frames must have real data

burnin = config['contextl'] - 1
contextlpad = burnin + train_dataset.ntspred_max
allisdata = interval_all(valdata['isdata'], contextlpad)
isnotsplit = interval_all(valdata['isstart'] == False, tpred)[1:, ...]
canstart = np.logical_and(allisdata[:isnotsplit.shape[0], :], isnotsplit)
flynum = 3  # 2
t0 = np.nonzero(canstart[:, flynum])[0]
idxstart = np.minimum(40000, len(t0) - 1)
if len(t0) > idxstart:
    t0 = t0[idxstart]
else:
    t0 = t0[0]
fliespred = np.array([flynum, ])

randstate_np = np.random.get_state()
randstate_torch = torch.random.get_rng_state()

nsamplesfuture = 32

# reseed numpy random number generator with randstate_np
np.random.set_state(randstate_np)
# reseed torch random number generator with randstate_torch
torch.random.set_rng_state(randstate_torch)
ani = animate_predict_open_loop(
    model, 
    val_dataset, 
    valdata, 
    val_scale_perfly, 
    config, 
    fliespred, 
    t0, 
    tpred,
    debug=False,
    plotattnweights=False, 
    plotfuture=train_dataset.ntspred_global > 1,
    nsamplesfuture=nsamplesfuture
)
# -

vidtime = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
# savevidfile = os.path.join(config['savedir'], f"samplevideo_{modeltype_str}_{savetime}_{vidtime}.gif")
savevidfile = os.path.join("/groups/branson/home/eyjolfsdottire/data", f"samplevideo_{modeltype_str}_{savetime}_{vidtime}.gif")
print('Saving animation to file %s...'%savevidfile)
writer = animation.PillowWriter(fps=30)
ani.save(savevidfile, writer=writer)
print('Finished writing.')


