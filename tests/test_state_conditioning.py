"""CPU smoke tests for ShiftDiF's minimal state-conditioned path interface."""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.Data_utils.real_datasets import CustomDataset
from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS


class StateAlignmentTests(unittest.TestCase):
    def test_state_is_as_of_day_before_ten_day_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = np.arange(80, dtype=np.float32).reshape(40, 2)
            pd.DataFrame(values, columns=['a', 'b']).to_csv(os.path.join(tmp, 'returns.csv'), index=False)
            state_ids = np.arange(40) % 3
            state_path = os.path.join(tmp, 'state.npy')
            np.save(state_path, state_ids)

            ds = CustomDataset(name='toy', data_root=os.path.join(tmp, 'returns.csv'), window=10,
                               proportion=.7, save2npy=False, neg_one_to_one=False,
                               period='train', output_dir=tmp, state_root=state_path)
            # First eligible target is rows 1..10; its information cutoff is row 0.
            self.assertEqual(ds.samples.shape, (18, 10, 2))
            self.assertEqual(ds.sample_states.shape, (18, 3))
            np.testing.assert_array_equal(ds.sample_states[0], np.eye(3, dtype=np.float32)[state_ids[0]])
            np.testing.assert_allclose(ds.samples[0, 0, 0], ds.data[1, 0])

            test_ds = CustomDataset(name='toy', data_root=os.path.join(tmp, 'returns.csv'), window=10,
                                    proportion=.7, save2npy=False, neg_one_to_one=False,
                                    period='test', output_dir=tmp, predict_length=10,
                                    state_root=state_path)
            # The first test target begins at row 28 and uses the state at row 27.
            np.testing.assert_allclose(test_ds.samples[0, 0, 0], test_ds.data[28, 0])
            np.testing.assert_array_equal(test_ds.sample_states[0],
                                          np.eye(3, dtype=np.float32)[state_ids[27]])


class ConditionalDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = Diffusion_TS(
            seq_length=10, feature_size=3, n_layer_enc=1, n_layer_dec=1,
            d_model=8, timesteps=8, sampling_timesteps=4, loss_type='l2',
            beta_schedule='cosine', n_heads=2, mlp_hidden_times=2,
            eta=0., attn_pd=0., resid_pd=0., kernel_size=1, padding_size=0,
            use_ff=False, state_dim=2,
        ).cpu().eval()
        self.x = torch.randn(2, 10, 3)
        self.state = torch.tensor([[1., 0.], [0., 1.]])

    def test_training_loss_and_conditioned_generation_shapes(self):
        loss = self.model(self.x, target=self.x, market_state=self.state)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        state_grad = self.model.model.state_embed.weight.grad
        self.assertIsNotNone(state_grad)
        self.assertGreater(state_grad.abs().sum().item(), 0.)
        paths = self.model.generate_mts(batch_size=2, market_state=self.state)
        self.assertEqual(tuple(paths.shape), (2, 10, 3))

    def test_state_changes_output_and_mask_remains_separate(self):
        t = torch.tensor([3, 3])
        out_a = self.model.output(self.x, t, market_state=self.state)
        out_b = self.model.output(self.x, t, market_state=self.state.flip(0))
        self.assertFalse(torch.allclose(out_a, out_b))

        # An all-false historical mask leaves every target value unknown; state is
        # still passed independently and the model returns a full future path.
        torch.manual_seed(11)
        path_a = self.model.fast_sample_infill(
            shape=self.x.shape, target=torch.zeros_like(self.x),
            sampling_timesteps=4, partial_mask=torch.zeros_like(self.x, dtype=torch.bool),
            model_kwargs={'coef': 0., 'learning_rate': 0.}, market_state=self.state)
        self.assertEqual(tuple(path_a.shape), (2, 10, 3))

        observed_mask = torch.zeros_like(self.x, dtype=torch.bool)
        observed_mask[:, 0, :] = True
        path_b = self.model.fast_sample_infill(
            shape=self.x.shape, target=self.x, sampling_timesteps=4,
            partial_mask=observed_mask, model_kwargs={'coef': 0., 'learning_rate': 0.},
            market_state=self.state)
        torch.testing.assert_close(path_b[observed_mask], self.x[observed_mask])

    def test_missing_or_wrong_state_shape_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, 'market_state is required'):
            self.model.output(self.x, torch.tensor([1, 1]))
        with self.assertRaisesRegex(ValueError, 'market_state must have shape'):
            self.model.output(self.x, torch.tensor([1, 1]), market_state=torch.ones(2, 3))


if __name__ == '__main__':
    unittest.main()
