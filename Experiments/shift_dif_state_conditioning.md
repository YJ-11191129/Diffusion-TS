# ShiftDiF: minimal filtered-state conditioning

This patch adds an optional global, as-of-time market-state condition to Diffusion-TS. It generates a complete 10-day multivariate return path from the filtered state at the forecast origin. The causal HMM pipeline below prepares those states; online CPD remains an extension point.

## State and target alignment

Let `r_t` be the vector of asset log returns on trading day `t`. A training example beginning at row `s` is

```text
target path:  (r_s, r_{s+1}, ..., r_{s+9})
condition:    filtered state S_{s-1}, using observations available through day s-1
```

`state_root` points to an `.npy` or numeric `.csv` file with exactly one row per row in `data_root`. A one-dimensional integer array is interpreted as discrete state labels and expanded to one-hot vectors. A two-dimensional `N x K` array is passed as supplied (for example, HMM filtered posterior probabilities); an `N x D` matrix can also carry continuous CPD features. `state_dim` must match the number of columns after label expansion.

The state file must contain **filtered** states. Fit state-model parameters on the training era only, then recursively filter each new observation. Do not use smoothed probabilities, full-sample state assignments, or a detector that inspects any observation after the forecast origin. Legacy external files receive positional checks only. Generated bundles carry hashes, calibration boundaries and split metadata; the example configuration requires that metadata. The pipeline validates dates before stripping them from the numeric return CSV. These checks cannot certify the causal construction of user-supplied input features.

When `state_root` is set, the dataset fits the return scaler on the training era, removes the origin with no prior state, and splits by raw time: each training path ends before the cutoff and each test path starts at or after it. Thus the tested prediction target contains only the ten future returns. The `predict` sampling route ignores target values and the historical mask, and samples a full path from the market state alone. Results are inverse-transformed to raw log-return units.

## Run

The template `Config/shift_dif_state.yaml` uses ten-step windows, a three-state HMM posterior, and placeholder input paths. Set `feature_size` and `state_dim` to match the actual return and state files. Then run:

```bash
python main.py --name shift_dif_state --config_file Config/shift_dif_state.yaml --train
python main.py --name shift_dif_state --config_file Config/shift_dif_state.yaml \
  --sample 1 --mode predict --pred_len 10 --milestone <checkpoint>
```

Each generated row in `ddpm_predict_shift_dif_state.npy` is one ten-day, multi-asset path. The matching test target windows are available from the same test dataset for scoring.

## Required controls

1. **Unconditional diffusion:** use the same split, windows, scaler, architecture, training budget, and random seeds; omit `state_root`, set `state_dim: 0`, and set `strict_path_split: true` in both dataset parameter blocks. The strict split keeps complete target paths on their side of the time cutoff and fits the scaler only on training rows.
2. **State-weighted resampling:** keep `state_root` on the training dataset, set `state_dim: 0`, and add `state_sampling_weights: [w_0, ..., w_{K-1}]` under `dataloader.train_dataset`. The loader uses posterior-weighted `WeightedRandomSampler` weights. Derive `w_k` from training-era frequencies only. The model receives no state vector in this control.
3. **State-conditioned diffusion:** use the filtered states in both datasets and `state_dim: K`; train and sample with identical target windows and budget.

Compare all methods on identical held-out origins and generated path counts. Report a proper multivariate distribution score (for example, energy score plus a variogram score), marginal CRPS, and any independent downstream forecast metric. No training or forecast performance increment is asserted by this code change; market data and leakage-safe state files were not available here.

## Meaning of the two conditions

`market_state` has shape `[batch, state_dim]` and summarizes the observed market at the forecast origin. It is projected to a global embedding and passed into every encoder and decoder adaptive normalization layer at every denoising step.

`partial_mask` has shape `[batch, time, asset]` and marks observed target values to preserve during infilling. It does not encode a market state and does not summarize the history. For the ten-day ShiftDiF path task, no historical values are supplied to the generator; an all-false mask is equivalent to state-conditioned generation, while a nonempty mask remains a separate inpainting constraint.

## Smoke test

Run `python tests/test_state_conditioning.py` in the repository's PyTorch environment. The test checks state-to-target alignment, scalar training loss, condition-dependent outputs, sampling dimensions, all-unknown-mask behavior, and fail-fast shape validation on CPU.

## Reproducible causal HMM pipeline

Run from the repository root with NumPy, pandas and SciPy (already covered by
`requirements.txt`). No HMM package or GPU is needed for state generation:

```bash
python scripts/shift_dif_hmm_toy.py --output-dir Data/datasets
python -m Utils.Data_utils.causal_hmm --features-csv Data/datasets/market_features.csv \
  --returns-csv Data/datasets/dated_asset_returns.csv --output-dir Data/datasets \
  --feature-columns market_return volatility --fit-rows 100 --states 3 \
  --proportion 0.9 --seed 123 --iterations 50
python tests/test_causal_hmm.py
python tests/test_state_conditioning.py
```

The first command produces 500 synthetic trading rows and six assets, matching
`Config/shift_dif_state.yaml`. The second shows the equivalent CSV conversion.
For real inputs, both CSVs must have an identical, unique, strictly increasing
`date` column and one row per observed trading day. No calendar-day reindexing,
sorting, row dropping or silent joins occur. Holidays are not missing rows.
The input producer must ensure every feature on date t was available by that
close; avoid centered rolling statistics, revised future data and forward labels.
The code cannot infer release lags from values alone.

**Three eras:** rows `[0,100)` calibrate the HMM; rows `[100,450)` supply
eligible diffusion targets; rows `[450,500)` are held out. Calibration is wholly
inside the training era. Standardization and seeded diagonal-Gaussian HMM EM
use only the calibration prefix. EM uses forward/backward responsibilities for
parameter estimation, then discards them. Exported probabilities are recomputed
by forward recursion with frozen parameters, preserving the last posterior at
the train/test boundary; no refit or reset occurs there.

A fixed fitted model uses later calibration observations when filtering earlier
calibration rows. Those rows are **diagnostic only**: the manifest forces the
first target to start at row 100, conditioned on row 99, when all calibration
observations are available. No eligible origin uses parameters fitted after that
origin. This differs from fitting on all training rows and using those parameters
for retrospective training conditions. The latter would leak within training.
Test origins also use frozen parameters and only observed features through s-1.
The state NPY is unshifted: row t means p(z_t | x_0,...,x_t; frozen parameters).
The existing dataset performs the single lag when pairing it with rows s..s+9.
Do not shift the NPY again.

Missing feature values are forward-filled then filled with fixed zero if no past
value exists. Infinities and nonnumeric features are rejected. Returns must all
be finite; missing targets are rejected. No test statistic is used for filling,
scaling, initialization or component ordering. K is fixed by `--states` and must
match `model.params.state_dim`; asset count must match `feature_size`. State IDs
are ordered by the first standardized feature's fitted mean, with no assumed
bull/bear interpretation. K, features and calibration length must be fixed from
training information, not chosen from test scores.

Outputs include `market_state_filtered.npy` (N x K float32),
`market_state_filtered.json` (split, seeds, preprocessing and SHA256 hashes),
`asset_log_returns.csv` (numeric columns only, preserving the verified dates),
`state_dates.csv` (audit dates and eligible condition flags), and
`hmm_parameters.npz` (frozen parameters and scaler). Keep the manifest adjacent
to the NPY. Regenerate the bundle if rows, dates or values change; changing only
`proportion` in the diffusion config is rejected when its split changes.
The manifest protects accidental mismatches, not maliciously forged provenance.

For an unconditional control set `path_start: 100` in both datasets, remove
`state_root`, set `state_metadata_required: false`, retain `strict_path_split: true`
and set `state_dim: 0`. This preserves the calibrated model's eligible origins.
The state-weighted control retains the bundle and its automatic prefix exclusion.

`OnlineStateFilter.update(observation)` is the reserved causal backend contract
for a future CPD implementation; this change supplies only a frozen HMM.
Tests mutate future observations (including post-calibration training rows),
replay truncated histories, verify deterministic output, check all ten-day window
alignments and reject date, split, dimension and file-integrity mismatches.
They validate temporal mechanics only; no forecast-performance comparison has
been run on either synthetic or real market data.
