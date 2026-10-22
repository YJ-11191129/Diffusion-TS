"""Frozen diagonal Gaussian HMM. EM is confined to an initial calibration prefix."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.special import logsumexp


class OnlineStateFilter(Protocol):
    """Extension point for a future CPD backend; update sees one observed row only."""
    def update(self, observation): ...


class FrozenHMM:
    def __init__(self, start, transition, means, variances):
        self.start, self.transition = start, transition
        self.means, self.variances = means, variances
        self.posterior = None

    def emissions(self, x):
        return -.5 * (np.log(2 * np.pi * self.variances)[None] +
                      (x[:, None] - self.means[None]) ** 2 / self.variances[None]).sum(2)

    def update(self, observation):
        prior = self.start if self.posterior is None else self.posterior @ self.transition
        logp = np.log(prior) + self.emissions(np.asarray(observation)[None])[0]
        self.posterior = np.exp(logp - logsumexp(logp))
        return self.posterior.copy()

    @classmethod
    def fit(cls, x, states=3, seed=123, iterations=50):
        n, d = x.shape
        if states < 2 or n < max(10, states * 3) or iterations < 1:
            raise ValueError('need >=2 states, >=max(10,3*K) calibration rows and positive iterations')
        rng = np.random.RandomState(seed)
        model = cls(np.full(states, 1 / states),
                    .9 * np.eye(states) + .1 / states,
                    x[rng.choice(n, states, replace=False)].copy(),
                    np.tile(np.maximum(x.var(0), 1e-3), (states, 1)))
        # Forward/backward is used ONLY for parameter estimation on calibration data.
        for _ in range(iterations):
            emission = model.emissions(x)
            a, b = np.empty((n, states)), np.zeros((n, states))
            a[0] = np.log(model.start) + emission[0]
            logtrans = np.log(model.transition)
            for t in range(1, n):
                a[t] = emission[t] + logsumexp(a[t-1, :, None] + logtrans, axis=0)
            for t in range(n-2, -1, -1):
                b[t] = logsumexp(logtrans + emission[t+1] + b[t+1], axis=1)
            likelihood = logsumexp(a[-1])
            gamma = np.exp(a + b - likelihood)
            counts = np.full((states, states), 1e-6)
            for t in range(n-1):
                counts += np.exp(a[t, :, None] + logtrans + emission[t+1] + b[t+1] - likelihood)
            model.start = (gamma[0] + 1e-6) / (gamma[0].sum() + states * 1e-6)
            model.transition = counts / counts.sum(1, keepdims=True)
            weights = gamma.sum(0) + 1e-12
            model.means = gamma.T @ x / weights[:, None]
            model.variances = np.maximum(
                np.einsum('nk,nkd->kd', gamma, (x[:, None] - model.means) ** 2) / weights[:, None], 1e-3)
        # Stable component order derived from calibration features only.
        order = np.argsort(model.means[:, 0])
        return cls(model.start[order], model.transition[order][:, order],
                   model.means[order], model.variances[order])


def dated_csv(path, date_column):
    frame = pd.read_csv(path)
    dates = pd.to_datetime(frame.pop(date_column), errors='raise')
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('dates must be unique, nonmissing and strictly increasing')
    if not (dates == dates.dt.normalize()).all():
        raise ValueError('use trading dates without intraday timestamps')
    return dates, frame


def generate(features_csv, returns_csv, output_dir, feature_columns, fit_rows,
             proportion=.9, states=3, seed=123, iterations=50, date_column='date'):
    dates, features = dated_csv(features_csv, date_column)
    return_dates, returns = dated_csv(returns_csv, date_column)
    if not dates.equals(return_dates):
        raise ValueError('feature and return trading dates must match exactly; no positional join')
    n = len(dates)
    cut = int(n * proportion)
    if not 0 < fit_rows < cut < n or cut - fit_rows < 10 or n - cut < 10:
        raise ValueError('require calibration prefix, >=10 training target rows and >=10 test rows')
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError('feature columns must be nonempty and unique')
    x = features[feature_columns].apply(pd.to_numeric, errors='raise').to_numpy(float)
    if np.isinf(x).any():
        raise ValueError('infinite features are invalid')
    # Forward fill only; initial unavailable features use a fixed zero, never backfill.
    x = pd.DataFrame(x).ffill().fillna(0.).to_numpy()
    mean, scale = x[:fit_rows].mean(0), x[:fit_rows].std(0)
    scale = np.maximum(scale, 1e-8)
    x = (x - mean) / scale
    values = returns.apply(pd.to_numeric, errors='raise').to_numpy(float)
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError('returns must be nonempty finite numeric data; do not impute targets')
    model = FrozenHMM.fit(x[:fit_rows], states, seed, iterations)
    probabilities = np.stack([model.update(row) for row in x]).astype(np.float32)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return_path = out / 'asset_log_returns.csv'
    pd.DataFrame(values, columns=returns.columns).to_csv(return_path, index=False)
    state_path = out / 'market_state_filtered.npy'
    np.save(state_path, probabilities)
    np.savez(out / 'hmm_parameters.npz', start=model.start, transition=model.transition,
             means=model.means, variances=model.variances, feature_mean=mean, feature_scale=scale)
    pd.DataFrame({'date': dates.dt.strftime('%Y-%m-%d'),
                  'eligible_condition': np.arange(n) >= fit_rows - 1}).to_csv(out / 'state_dates.csv', index=False)
    metadata = dict(schema_version=1, backend='frozen_diagonal_gaussian_hmm',
                    fit_rows=fit_rows, fit_end=str(dates.iloc[fit_rows-1].date()),
                    first_target_row=fit_rows, split_row=cut, proportion=proportion,
                    rows=n, state_dim=states, feature_columns=feature_columns,
                    seed=seed, iterations=iterations, missing='forward_fill_then_fixed_zero',
                    features_source_sha256=hashlib.sha256(Path(features_csv).read_bytes()).hexdigest(),
                    returns_source_sha256=hashlib.sha256(Path(returns_csv).read_bytes()).hexdigest(),
                    returns_sha256=hashlib.sha256(return_path.read_bytes()).hexdigest(),
                    states_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest())
    state_path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return probabilities, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features-csv', 'returns-csv', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--feature-columns', nargs='+', required=True)
    parser.add_argument('--fit-rows', type=int, required=True)
    parser.add_argument('--proportion', type=float, default=.9)
    parser.add_argument('--states', type=int, default=3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--date-column', default='date')
    args = vars(parser.parse_args())
    generate(**args)


if __name__ == '__main__':
    main()
