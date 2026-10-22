"""Synthetic CPU tests; no financial performance claim."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Utils.Data_utils.causal_hmm import FrozenHMM, generate
from Utils.Data_utils.real_datasets import CustomDataset


class CausalHMMTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dates = pd.bdate_range('2020-01-01', periods=160)
        rng = np.random.RandomState(9)
        self.features = pd.DataFrame({'date': self.dates, 'r': rng.randn(160),
                                      'vol': rng.randn(160)})
        self.features.loc[[0, 5, 110], 'vol'] = np.nan
        self.returns = pd.DataFrame({'date': self.dates, 'a': rng.randn(160), 'b': rng.randn(160)})

    def tearDown(self):
        self.tmp.cleanup()

    def run_pipeline(self, name='out'):
        f, r = self.root / 'f.csv', self.root / 'r.csv'
        self.features.to_csv(f, index=False)
        self.returns.to_csv(r, index=False)
        return generate(f, r, self.root / name, ['r', 'vol'], 40, .75,
                        states=3, iterations=12)

    def dataset(self, period='train', **overrides):
        args = dict(name='toy', data_root=str(self.root / 'out/asset_log_returns.csv'),
                    state_root=str(self.root / 'out/market_state_filtered.npy'),
                    state_metadata_required=True, window=10, proportion=.75,
                    save2npy=False, neg_one_to_one=False, period=period,
                    output_dir=str(self.root), predict_length=10 if period == 'test' else None)
        args.update(overrides)
        return CustomDataset(**args)

    def test_forward_filter_matches_enumerated_joint_probabilities(self):
        from itertools import product
        model = FrozenHMM(np.array([.6, .4]), np.array([[.8, .2], [.3, .7]]),
                          np.array([[-1.], [1.]]), np.ones((2, 1)))
        x = np.array([[-.3], [.7], [-1.2]])
        emissions = np.exp(model.emissions(x))
        for t, row in enumerate(x):
            expected = np.zeros(2)
            for path in product(range(2), repeat=t+1):
                prob = model.start[path[0]] * emissions[0, path[0]]
                for j in range(1, t+1):
                    prob *= model.transition[path[j-1], path[j]] * emissions[j, path[j]]
                expected[path[-1]] += prob
            np.testing.assert_allclose(model.update(row), expected / expected.sum(), atol=1e-12)

    def test_future_mutation_and_reproducibility(self):
        p, meta = self.run_pipeline()
        repeated, _ = self.run_pipeline('repeat')
        np.testing.assert_array_equal(p, repeated)
        # Includes mutations in diffusion TRAINING era after calibration, not just test.
        for cutoff in (55, 119, 135):
            original = self.features.copy()
            self.features.loc[cutoff+1:, ['r', 'vol']] = 10000.
            changed, _ = self.run_pipeline('changed')
            np.testing.assert_array_equal(p[:cutoff+1], changed[:cutoff+1])
            self.assertFalse(np.allclose(p[cutoff+1:], changed[cutoff+1:]))
            self.features = original
        self.assertEqual(p.shape, (160, 3))
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
        # Prefix replay with the same calibration data must match full-run history.
        self.features, self.returns = self.features.iloc[:140], self.returns.iloc[:140]
        prefix, _ = self.run_pipeline('prefix')
        np.testing.assert_array_equal(p[:140], prefix)

    def test_calibration_exclusion_and_one_day_alignment(self):
        p, _ = self.run_pipeline()
        train, test = self.dataset(), self.dataset('test')
        self.assertEqual(len(train), 71)  # starts 40..110, targets end <=119
        self.assertEqual(len(test), 31)  # starts 120..150
        for dataset, starts in ((train, range(40, 111)), (test, range(120, 151))):
            for i, start in enumerate(starts):
                np.testing.assert_array_equal(dataset.sample_states[i], p[start-1])
                np.testing.assert_allclose(dataset.samples[i], dataset.data[start:start+10], rtol=1e-6, atol=1e-7)
        with self.assertRaisesRegex(ValueError, 'metadata split'):
            self.dataset(proportion=.8)
        path = self.root / 'out/asset_log_returns.csv'
        frame = pd.read_csv(path).iloc[::-1]
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.dataset()

    def test_invalid_dates_targets_and_dimensions(self):
        original = self.features.copy()
        for frame in (original.iloc[::-1], pd.concat([original.iloc[:1], original.iloc[:-1]])):
            self.features = frame
            with self.assertRaises(ValueError):
                self.run_pipeline()
        self.features = original
        self.returns.loc[3, 'date'] = pd.Timestamp('2019-01-01')
        with self.assertRaises(ValueError):
            self.run_pipeline()
        self.returns['date'] = self.dates
        self.returns.loc[10, 'a'] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite numeric'):
            self.run_pipeline()
        self.returns.loc[10, 'a'] = 0
        self.run_pipeline()
        manifest = self.root / 'out/market_state_filtered.json'
        meta = json.loads(manifest.read_text())
        meta['state_dim'] = 7
        manifest.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.dataset()
        manifest.unlink()
        with self.assertRaisesRegex(ValueError, 'metadata is required'):
            self.dataset()


if __name__ == '__main__':
    unittest.main()
