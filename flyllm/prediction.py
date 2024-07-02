import numpy as np
import torch

from flyllm.pose import FlyExample


def predict_all(dataloader, dataset, model, config, mask):
    is_causal = dataset.ismasked() == False

    with torch.no_grad():
        w = next(iter(model.parameters()))
        device = w.device

    example_params = dataset.get_flyexample_params()

    # compute predictions and labels for all validation data using default masking
    all_pred = []
    all_mask = []
    all_labels = []
    # all_pred_discrete = []
    # all_labels_discrete = []
    with torch.no_grad():
        for example in dataloader:
            pred = model.output(example['input'].to(device=device), mask=mask, is_causal=is_causal)
            if config['modelstatetype'] == 'prob':
                pred = model.maxpred(pred)
            elif config['modelstatetype'] == 'best':
                pred = model.randpred(pred)
            if isinstance(pred, dict):
                pred = {k: v.cpu() for k, v in pred.items()}
            else:
                pred = pred.cpu()
            # pred1 = dataset.get_full_pred(pred)
            # labels1 = dataset.get_full_labels(example=example,use_todiscretize=True)
            example_obj = FlyExample(example_in=example, **example_params)
            label_obj = example_obj.labels
            pred_obj = label_obj.copy()
            pred_obj.erase_labels()
            pred_obj.set_prediction(pred)

            for i in range(np.prod(label_obj.pre_sz)):
                all_pred.append(pred_obj.copy_subindex(idx_pre=i))
                all_labels.append(label_obj.copy_subindex(idx_pre=i))

            # if dataset.discretize:
            #   all_pred_discrete.append(pred['discrete'])
            #   all_labels_discrete.append(example['labels_discrete'])
            # if 'mask' in example:
            #   all_mask.append(example['mask'])

    return all_pred, all_labels  # ,all_mask,all_pred_discrete,all_labels_discrete
