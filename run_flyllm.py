import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import animation
import tqdm
import torch
import transformers
import datetime
import argparse
import pickle

from config import posenames, featglobal, featrelative, nfeatures
from features import (
    compute_features,
)
from plotting import (
    debug_plot_dct_relative_error,
    debug_plot_global_error,
    debug_plot_global_histograms,
    debug_plot_predictions_vs_labels,
    initialize_debug_plots, update_debug_plots,
    initialize_loss_plots, update_loss_plots,
    select_featidx_plot,
)
from data import load_and_filter_data, interval_all, chunk_data, sanity_check_tspred
from dataset import FlyMLMDataset
from simulation import animate_predict_open_loop
from utils import get_dct_matrix, compute_npad
from models import (
    initialize_model, initialize_loss,
    generate_square_full_mask,
    criterion_wrapper,
    compute_loss,
    predict_all,
    update_loss_nepochs,
    sanity_check_temporal_dep,
    stack_batch_list,
)
from io import (
    read_config, load_config_from_model_file, get_modeltype_str,
    load_model, save_model, parse_modelfile,
    clean_intermediate_results
)


def main(configfile, loadmodelfile=None, restartmodelfile=None):
    tmpsavefile = ''
    # to save time, i saved the chunked data to a pkl file
    # tmpsavefile = 'chunkeddata20230905.pkl'
    # tmpsavefile = 'chunkeddata20230828.pkl'
    doloadtmpsavefile = os.path.exists(tmpsavefile)
    # tmpsavefile = None

    # configuration parameters for this model
    config = read_config(configfile)

    # set loadmodelfile and restartmodelfile from config if not specified
    if loadmodelfile is None and 'loadmodelfile' in config:
        loadmodelfile = config['loadmodelfile']
    if restartmodelfile is None and 'restartmodelfile' in config:
        loadmodelfile = config['restartmodelfile']

    # if loadmodelfile or restartmodelfile specified, use its config
    if loadmodelfile is not None:
        load_config_from_model_file(loadmodelfile, config)
    elif restartmodelfile is not None:
        no_overwrite = ['num_train_epochs', ]
        load_config_from_model_file(restartmodelfile, config, no_overwrite=no_overwrite)

    print(f"batch size = {config['batch_size']}")

    # seed the random number generators
    np.random.seed(config['numpy_seed'])
    torch.manual_seed(config['torch_seed'])

    # set device (cuda/cpu)
    device = torch.device(config['device'])

    plt.ion()

    if doloadtmpsavefile:
        # load cached, pre-chunked data
        print(f'Loading tmp save file {tmpsavefile}')
        with open(tmpsavefile, 'rb') as f:
            tmp = pickle.load(f)
        data = tmp['data']
        scale_perfly = tmp['scale_perfly']
        valdata = tmp['valdata']
        val_scale_perfly = tmp['val_scale_perfly']
        X = tmp['X']
        valX = tmp['valX']
    else:
        # load raw data
        data, scale_perfly = load_and_filter_data(config['intrainfile'], config)
        valdata, val_scale_perfly = load_and_filter_data(config['invalfile'], config)

    # if using discrete cosine transform, create dct matrix
    # this didn't seem to work well, so probably won't use in the future
    if config['dct_tau'] is not None and config['dct_tau'] > 0:
        dct_m, idct_m = get_dct_matrix(config['dct_tau'])
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
    reparamfun = lambda x, id, flynum, **kwargs: compute_features(x, id, flynum, scale_perfly, outtype=np.float32,
                                                                  **compute_feature_params, **kwargs)

    val_reparamfun = lambda x, id, flynum, **kwargs: compute_features(x, id, flynum, val_scale_perfly,
                                                                      outtype=np.float32,
                                                                      **compute_feature_params, **kwargs)

    # sanity check on computing features when predicting many frames into the future
    sanity_check_tspred(data, compute_feature_params, npad, scale_perfly, contextl=config['contextl'], t0=510, flynum=0)

    if not doloadtmpsavefile:
        # chunk the data if we didn't load the pre-chunked cache file
        print('Chunking training data...')
        X = chunk_data(data, config['contextl'], reparamfun, **chunk_data_params)
        print('Chunking val data...')
        valX = chunk_data(valdata, config['contextl'], val_reparamfun, **chunk_data_params)
        print('Done.')

        if len(tmpsavefile) > 0:
            print('Saving chunked data to file')
            with open(tmpsavefile, 'wb') as f:
                pickle.dump({'X': X, 'valX': valX, 'data': data, 'valdata': valdata, 'scale_perfly': scale_perfly,
                             'val_scale_perfly': val_scale_perfly}, f)
            print('Done.')

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
        'dct_ms': (dct_m, idct_m),
        'tspred_global': config['tspred_global'],
        'discrete_tspred': config['discrete_tspred'],
        'compute_pose_vel': config['compute_pose_vel'],
    }
    train_dataset_params = {
        'input_noise_sigma': config['input_noise_sigma'],
    }

    print('Creating training data set...')
    train_dataset = FlyMLMDataset(X, **train_dataset_params, **dataset_params)
    print('Done.')

    # zscore and discretize parameters for validation data set based on train data
    # we will continue to use these each time we rechunk the data
    dataset_params['zscore_params'] = train_dataset.get_zscore_params()
    dataset_params['discretize_params'] = train_dataset.get_discretize_params()

    print('Creating validation data set...')
    val_dataset = FlyMLMDataset(valX, **dataset_params)
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

    # set up debug plots
    debug_params = {}
    # if contextl is long, still just look at samples from the first 64 frames
    if config['contextl'] > 64:
        debug_params['tsplot'] = np.round(np.linspace(0, 64, 5)).astype(int)
        debug_params['traj_nsamplesplot'] = 1
    hdebug = {}
    hdebug['train'] = initialize_debug_plots(train_dataset, train_dataloader, data, name='Train', **debug_params)
    hdebug['val'] = initialize_debug_plots(val_dataset, val_dataloader, valdata, name='Val', **debug_params)

    # create the model
    model, criterion = initialize_model(d_input, d_output, config, train_dataset, device)

    # optimizer
    num_training_steps = config['num_train_epochs'] * ntrain
    optimizer = transformers.optimization.AdamW(model.parameters(), **config['optimizer_args'])
    lr_scheduler = transformers.get_scheduler('linear', optimizer, num_warmup_steps=0,
                                              num_training_steps=num_training_steps)

    # initialize structure to keep track of loss
    loss_epoch = initialize_loss(train_dataset, config)
    last_val_loss = None

    progress_bar = tqdm.tqdm(range(num_training_steps))

    # create attention mask
    contextl = example['input'].shape[1]
    if config['modeltype'] == 'mlm':
        train_src_mask = generate_square_full_mask(contextl).to(device)
        is_causal = False
    elif config['modeltype'] == 'clm':
        train_src_mask = torch.nn.Transformer.generate_square_subsequent_mask(contextl, device=device)
        is_causal = True
        # train_src_mask = generate_square_subsequent_mask(contextl).to(device)
    else:
        raise

    # sanity check on temporal dependences
    sanity_check_temporal_dep(train_dataloader, device, train_src_mask, is_causal, model, tmess=300)

    modeltype_str = get_modeltype_str(config, train_dataset)
    if ('model_nickname' in config) and (config['model_nickname'] is not None):
        modeltype_str = config['model_nickname']

    hloss = initialize_loss_plots(loss_epoch)

    # epoch = 40
    # restartmodelfile = f'llmnets/flyclm_flattened_mixed_71G01_male_epoch{epoch}_20230517T153613.pth'
    # loss_epoch = load_model(restartmodelfile,model,device,lr_optimizer=optimizer,scheduler=lr_scheduler)
    # with torch.no_grad():
    #   pred = model(example['input'].to(device=device),mask=train_src_mask,is_causal=is_causal)
    # update_debug_plots(hdebug['train'],config,model,train_dataset,example,pred,name='Train',criterion=criterion)

    # train
    if loadmodelfile is None:

        # restart training
        if restartmodelfile is not None:
            loss_epoch = load_model(restartmodelfile, model, device, lr_optimizer=optimizer, scheduler=lr_scheduler)
            update_loss_nepochs(loss_epoch, config['num_train_epochs'])
            update_loss_plots(hloss, loss_epoch)
            # loss_epoch = {k: v.cpu() for k,v in loss_epoch.items()}
            epoch = np.nonzero(np.isnan(loss_epoch['train'].cpu().numpy()))[0][0]
            progress_bar.update(epoch * ntrain)
        else:
            epoch = 0

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
                loss, loss_discrete, loss_continuous = criterion_wrapper(example, pred, criterion, train_dataset,
                                                                         config)

                loss.backward()

                # how many timepoints are in this batch for normalization
                if config['modeltype'] == 'mlm':
                    nmask_train += torch.count_nonzero(example['mask'])
                else:
                    nmask_train += example['input'].shape[0] * ntimepoints_per_batch

                if step % config['niterplot'] == 0:
                    with torch.no_grad():
                        trainpred = model.output(example['input'].to(device=device), mask=train_src_mask,
                                                 is_causal=is_causal)
                        valpred = model.output(valexample['input'].to(device=device), mask=train_src_mask,
                                               is_causal=is_causal)
                    update_debug_plots(hdebug['train'], config, model, train_dataset, example, trainpred, name='Train',
                                       criterion=criterion, **debug_params)
                    update_debug_plots(hdebug['val'], config, model, val_dataset, valexample, valpred, name='Val',
                                       criterion=criterion, **debug_params)
                    plt.show()
                    plt.pause(.1)

                tr_loss_step = loss.detach()
                tr_loss += tr_loss_step
                if train_dataset.discretize:
                    tr_loss_discrete += loss_discrete.detach()
                    tr_loss_continuous += loss_continuous.detach()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()

                # update progress bar
                stat = {'train loss': tr_loss.item() / nmask_train, 'last val loss': last_val_loss, 'epoch': epoch}
                if train_dataset.discretize:
                    stat['train loss discrete'] = tr_loss_discrete.item() / nmask_train
                    stat['train loss continuous'] = tr_loss_continuous.item() / nmask_train
                progress_bar.set_postfix(stat)
                progress_bar.update(1)

                # end of iteration loop

            # training epoch complete
            loss_epoch['train'][epoch] = tr_loss.item() / nmask_train
            if train_dataset.discretize:
                loss_epoch['train_discrete'][epoch] = tr_loss_discrete.item() / nmask_train
                loss_epoch['train_continuous'][epoch] = tr_loss_continuous.item() / nmask_train

            # compute validation loss after this epoch
            if val_dataset.discretize:
                loss_epoch['val'][epoch], loss_epoch['val_discrete'][epoch], loss_epoch['val_continuous'][epoch] = \
                    compute_loss(model, val_dataloader, val_dataset, device, train_src_mask, criterion, config)
            else:
                loss_epoch['val'][epoch] = \
                    compute_loss(model, val_dataloader, val_dataset, device, train_src_mask, criterion, config)
            last_val_loss = loss_epoch['val'][epoch].item()

            update_loss_plots(hloss, loss_epoch)
            plt.show()
            plt.pause(.1)

            # rechunk the training data
            if np.mod(epoch + 1, config['epochs_rechunk']) == 0:
                print(f'Rechunking data after epoch {epoch}')
                X = chunk_data(data, config['contextl'], reparamfun, **chunk_data_params)

                train_dataset = FlyMLMDataset(X, **train_dataset_params, **dataset_params)
                print('New training data set created')

            if (epoch + 1) % config['save_epoch'] == 0:
                savefile = os.path.join(config['savedir'], f"fly{modeltype_str}_epoch{epoch + 1}_{savetime}.pth")
                print(f'Saving to file {savefile}')
                save_model(savefile, model, lr_optimizer=optimizer, scheduler=lr_scheduler, loss=loss_epoch,
                           config=config)

        savefile = os.path.join(config['savedir'], f'fly{modeltype_str}_epoch{epoch + 1}_{savetime}.pth')
        save_model(savefile, model, lr_optimizer=optimizer, scheduler=lr_scheduler, loss=loss_epoch, config=config)

        print('Done training')
    else:
        modeltype_str, savetime = parse_modelfile(loadmodelfile)
        loss_epoch = load_model(loadmodelfile, model, device, lr_optimizer=optimizer, scheduler=lr_scheduler)
        update_loss_plots(hloss, loss_epoch)

    model.eval()

    # compute predictions and labels for all validation data using default masking
    all_pred, all_labels, all_mask, all_pred_discrete, all_labels_discrete = predict_all(val_dataloader, val_dataset,
                                                                                         model, config, train_src_mask)

    # plot comparison between predictions and labels on validation data
    predv = stack_batch_list(all_pred)
    labelsv = stack_batch_list(all_labels)
    maskv = stack_batch_list(all_mask)
    pred_discretev = stack_batch_list(all_pred_discrete)
    labels_discretev = stack_batch_list(all_labels_discrete)

    fig, ax = debug_plot_global_histograms(predv, labelsv, train_dataset, nbins=25, subsample=1, compare='pred')
    # glabelsv = train_dataset.get_global_movement(labelsv)
    # gpredprev = torch.zeros(glabelsv.shape)
    # gpredprev[:] = np.nan
    # for i,dt in enumerate(train_dataset.tspred_global):
    #   gpredprev[dt:,i,:] = glabelsv[:-dt,i,:]
    # predprev = torch.zeros(labelsv.shape)
    # predprev[:] = np.nan
    # train_dataset.set_global_movement(gpredprev,predprev)
    # fig,ax = debug_plot_global_histograms(predprev,labelsv,train_dataset,nbins=25,subsample=100)

    if train_dataset.dct_m is not None:
        debug_plot_dct_relative_error(predv, labelsv, train_dataset)
    if train_dataset.ntspred_global > 1:
        debug_plot_global_error(predv, labelsv, pred_discretev, labels_discretev, train_dataset)

    # crop to nplot for plotting
    nplot = 8000  # min(len(all_labels),8000//config['batch_size']//config['contextl']+1)
    predv = predv[:nplot, :]
    labelsv = labelsv[:nplot, :]
    if len(maskv) > 0:
        maskv = maskv[:nplot, :]
    pred_discretev = pred_discretev[:nplot, :]
    labels_discretev = labels_discretev[:nplot, :]

    if maskv is not None and len(maskv) > 0:
        maskidx = torch.nonzero(maskv)[:, 0]
    else:
        maskidx = None

    ntspred_plot = np.minimum(4, train_dataset.ntspred_global)
    featidxplot = select_featidx_plot(train_dataset, ntspred_plot)
    naxc = np.maximum(1, int(np.round(len(featidxplot) / nfeatures)))
    fig, ax = debug_plot_predictions_vs_labels(predv, labelsv, pred_discretev, labels_discretev, outnames=outnames,
                                               maskidx=maskidx, naxc=naxc, featidxplot=featidxplot, dataset=val_dataset)
    if train_dataset.ntspred_global > 1:
        featidxplot = select_featidx_plot(train_dataset, ntspred_plot=train_dataset.ntspred_global, ntsplot_relative=0)
        naxc = np.maximum(1, int(np.round(len(featidxplot) / nfeatures)))
        fig, ax = debug_plot_predictions_vs_labels(predv, labelsv, pred_discretev, labels_discretev, outnames=outnames,
                                                   maskidx=maskidx, naxc=naxc, featidxplot=featidxplot,
                                                   dataset=val_dataset)

    if train_dataset.ntspred_global > 1:
        featidxplot = train_dataset.ravel_label_index([(featglobal[0], t) for t in train_dataset.tspred_global])
        fig, ax = debug_plot_predictions_vs_labels(predv, labelsv, pred_discretev, labels_discretev, outnames=outnames,
                                                   maskidx=maskidx, featidxplot=featidxplot, dataset=val_dataset)

    if train_dataset.dct_tau > 0:
        fstrs = ['left_middle_leg_tip_angle', 'left_front_leg_tip_angle', 'left_wing_angle']
        fs = [posenames.index(x) for x in fstrs]
        featidxplot = train_dataset.ravel_label_index(
            [(f, i + 1) for i in range(train_dataset.dct_tau + 1) for f in fs])
        fig, ax = debug_plot_predictions_vs_labels(predv, labelsv, pred_discretev, labels_discretev, outnames=outnames,
                                                   maskidx=maskidx, featidxplot=featidxplot, dataset=val_dataset,
                                                   naxc=len(fs))

        predrelative_dct = train_dataset.get_relative_movement_dct(predv.numpy())
        labelsrelative_dct = train_dataset.get_relative_movement_dct(labelsv.numpy())
        fsdct = [np.array(posenames)[featrelative].tolist().index(x) for x in fstrs]
        predrelative_dct = predrelative_dct[:, :, fsdct].astype(train_dataset.dtype)
        labelsrelative_dct = labelsrelative_dct[:, :, fsdct].astype(train_dataset.dtype)
        outnamescurr = [f'{f}_dt{i + 1}' for i in range(train_dataset.dct_tau) for f in fstrs]
        fig, ax = debug_plot_predictions_vs_labels(
            torch.as_tensor(predrelative_dct.reshape((-1, train_dataset.dct_tau * len(fsdct)))),
            torch.as_tensor(labelsrelative_dct.reshape((-1, train_dataset.dct_tau * len(fsdct)))),
            outnames=outnamescurr, maskidx=maskidx, naxc=len(fstrs))

    # generate an animation of open loop prediction
    tpred = 2000 + config['contextl']

    # all frames must have real data

    burnin = config['contextl'] - 1
    contextlpad = burnin + 1
    allisdata = interval_all(valdata['isdata'], contextlpad)
    isnotsplit = interval_all(valdata['isstart'] == False, tpred)[1:, ...]
    canstart = np.logical_and(allisdata[:isnotsplit.shape[0], :], isnotsplit)
    flynum = 2
    t0 = np.nonzero(canstart[:, flynum])[0][40000]
    # flynum = 2
    # t0 = np.nonzero(canstart[:,flynum])[0][0]
    fliespred = np.array([flynum, ])

    randstate_np = np.random.get_state()
    randstate_torch = torch.random.get_rng_state()

    nsamplesfuture = 32

    # reseed numpy random number generator with randstate_np
    np.random.set_state(randstate_np)
    # reseed torch random number generator with randstate_torch
    torch.random.set_rng_state(randstate_torch)
    ani = animate_predict_open_loop(model, val_dataset, valdata, val_scale_perfly, config, fliespred, t0, tpred,
                                    debug=False,
                                    plotattnweights=False, plotfuture=train_dataset.ntspred_global > 1,
                                    nsamplesfuture=nsamplesfuture)

    vidtime = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    savevidfile = os.path.join(config['savedir'], f"samplevideo_{modeltype_str}_{savetime}_{vidtime}.gif")

    print('Saving animation to file %s...' % savevidfile)
    writer = animation.PillowWriter(fps=30)
    ani.save(savevidfile, writer=writer)
    print('Finished writing.')


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, required=False, help='Path to config file', metavar='configfile',
                        dest='configfile')
    parser.add_argument('-l', type=str, required=False, help='Path to model file to load', metavar='loadmodelfile',
                        dest='loadmodelfile')
    parser.add_argument('-r', type=str, required=False, help='Path to model file to restart training from',
                        metavar='restartmodelfile', dest='restartmodelfile')
    parser.add_argument('--clean', type=str, required=False,
                        help='Delete intermediate networks saved in input directory.', metavar='cleandir',
                        dest='cleandir')
    args = parser.parse_args()

    if args.cleandir is not None:
        assert os.path.isdir(args.cleandir)
        removedfiles = clean_intermediate_results(args.cleandir)
    else:
        main(args.configfile, loadmodelfile=args.loadmodelfile, restartmodelfile=args.restartmodelfile)
    # explore_representation(args.configfile)
