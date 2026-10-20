# ShiftDiF: minimal filtered-state conditioning

This patch adds an optional global, as-of-time market-state condition to Diffusion-TS. It generates a complete 10-day multivariate return path from the filtered state at the forecast origin. It does not estimate an HMM or online change-point detector; those states must be prepared upstream.

## State and target alignment

Let `r_t` be the vector of asset log returns on trading day `t`. A training example beginning at row `s` is

```text
target path:  (r_s, r_{s+1}, ..., r_{s+9})
condition:    filtered state S_{s-1}, using observations available through day s-1
```

`state_root` points to an `.npy` or numeric `.csv` file with exactly one row per row in `data_root`. A one-dimensional integer array is interpreted as discrete state labels and expanded to one-hot vectors. A two-dimensional `N x K` array is passed as supplied (for example, HMM filtered posterior probabilities); an `N x D` matrix can also carry continuous CPD features. `state_dim` must match the number of columns after label expansion.

The state file must contain **filtered** states. Fit state-model parameters on the training era only, then recursively filter each new observation. Do not use smoothed probabilities, full-sample state assignments, or a detector that inspects any observation after the forecast origin. The loader enforces row count and date order by positional alignment; it cannot determine whether an upstream file was produced without look-ahead.

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
