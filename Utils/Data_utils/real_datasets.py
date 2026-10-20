import os
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=64, 
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3,
        state_root=None,
        strict_path_split=False
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert not (predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.state_root = state_root
        self.market_states = self._read_market_states(state_root) if state_root else None
        self.strict_path_split = bool(strict_path_split or self.market_states is not None)
        if self.strict_path_split:
            # Fit all value transforms on the training era only. State rows must be
            # causal filtered probabilities/features aligned one-for-one to raw rows.
            split = int(len(self.rawdata) * proportion)
            self.scaler = MinMaxScaler().fit(self.rawdata[:split])
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        if not self.strict_path_split:
            train, inference = self.__getsamples(self.data, proportion, seed)
            self.samples = train if period == 'train' else inference
            self.sample_states = None
        else:
            train, train_states, inference, inference_states = self._get_path_samples(self.data, proportion)
            self.samples = train if period == 'train' else inference
            self.sample_states = None if self.market_states is None else (
                train_states if period == 'train' else inference_states)
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    @staticmethod
    def _read_market_states(path):
        """Load an as-of-time state series: integer labels or an N x K probability matrix."""
        if path.endswith('.npy'):
            states = np.load(path)
        elif path.endswith('.csv'):
            states = pd.read_csv(path).to_numpy()
        else:
            raise ValueError('state_root must point to an aligned .npy or numeric .csv file')
        states = np.asarray(states)
        if states.ndim == 1:
            numeric = states.astype(np.float32)
            if np.allclose(numeric, np.round(numeric)) and np.all(numeric >= 0):
                labels = numeric.astype(int)
                states = np.eye(int(labels.max()) + 1, dtype=np.float32)[labels]
            else:
                states = numeric[:, None]
        elif states.ndim == 2:
            states = states.astype(np.float32)
        else:
            raise ValueError('states must be a label vector or an N x K matrix')
        if not np.isfinite(states).all():
            raise ValueError('state series contains missing or non-finite values')
        if states.shape[1] > 1 and np.all(states >= 0) and np.allclose(states.sum(axis=1), 1., atol=1e-4):
            # Preserve filtered HMM posteriors as supplied; normalize small drift only.
            states = states / np.maximum(states.sum(axis=1, keepdims=True), 1e-12)
        return states.astype(np.float32)

    def _get_path_samples(self, data, proportion):
        if self.market_states is not None and len(self.market_states) != len(data):
            raise ValueError('state_root must have exactly one row per raw data row')
        n = len(data)
        cut = int(n * proportion)
        # A path beginning at s is conditioned on information available at s-1.
        # Train targets end no later than the split; test targets start at the split.
        train_starts = np.arange(1, max(1, cut - self.window + 1))
        test_starts = np.arange(max(1, cut), max(max(1, cut), n - self.window + 1))

        def make(starts):
            if len(starts) == 0:
                empty_paths = np.empty((0, self.window, self.var_num), dtype=np.float32)
                empty_states = None if self.market_states is None else np.empty(
                    (0, self.market_states.shape[1]), dtype=np.float32)
                return empty_paths, empty_states
            paths = np.stack([data[s:s + self.window] for s in starts]).astype(np.float32)
            states = None if self.market_states is None else self.market_states[starts - 1]
            return paths, states

        return (*make(train_starts), *make(test_starts))

    def __getsamples(self, data, proportion, seed):
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i in range(self.sample_num_total):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        train_data, test_data = self.divide(x, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)
    
    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        # id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            if self.sample_states is not None:
                return torch.from_numpy(x).float(), torch.from_numpy(m), torch.from_numpy(self.sample_states[ind]).float()
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        if self.sample_states is not None:
            return torch.from_numpy(x).float(), torch.from_numpy(self.sample_states[ind]).float()
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
