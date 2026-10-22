import torch
from Utils.io_utils import instantiate_from_config


def build_dataloader(config, args=None):
    batch_size = config['dataloader']['batch_size']
    jud = config['dataloader']['shuffle']
    config['dataloader']['train_dataset']['params']['output_dir'] = args.save_dir
    dataset = instantiate_from_config(config['dataloader']['train_dataset'])

    sampler = None
    state_weights = config['dataloader']['train_dataset'].get('state_sampling_weights')
    if state_weights is not None:
        if getattr(dataset, 'sample_states', None) is None:
            raise ValueError('state_sampling_weights requires train_dataset.params.state_root')
        state_weights = torch.as_tensor(state_weights, dtype=torch.float32)
        if state_weights.numel() != dataset.sample_states.shape[1]:
            raise ValueError('state_sampling_weights must have one weight per state dimension')
        if not torch.isfinite(state_weights).all() or torch.any(state_weights < 0) or state_weights.sum() <= 0:
            raise ValueError('state_sampling_weights must be finite, non-negative, and not all zero')
        weights = torch.as_tensor(dataset.sample_states, dtype=torch.float32) @ state_weights
        if torch.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('state_sampling_weights produce invalid sample weights')
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=jud if sampler is None else False,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=sampler,
                                             drop_last=jud)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': dataset
    }

    return dataload_info

def build_dataloader_cond(config, args=None):
    batch_size = config['dataloader']['sample_size']
    config['dataloader']['test_dataset']['params']['output_dir'] = args.save_dir
    if args.mode == 'infill':
        config['dataloader']['test_dataset']['params']['missing_ratio'] = args.missing_ratio
    elif args.mode == 'predict':
        config['dataloader']['test_dataset']['params']['predict_length'] = args.pred_len
    test_dataset = instantiate_from_config(config['dataloader']['test_dataset'])

    dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=False)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': test_dataset
    }

    return dataload_info


if __name__ == '__main__':
    pass

