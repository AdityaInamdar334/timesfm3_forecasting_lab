# Reproducibility and Experimental Limitations

## Experiment provenance

**Independent, synthetic PyTorch benchmark.** This experiment never loads or calls Google's TimesFM-3 weights. Its intended lesson is about the evaluation logic behind multivariate forecasting and distribution shifts.

All five seeds, 0 through 4, were run with 160 supervised optimization steps using PyTorch 2.10.0+cpu and NumPy 2.3.5 in the author's execution environment. Individual results were saved as `results/seed0.json` through `results/seed4.json` and combined into `results/benchmark.json`. `benchmark.py --seeds 5 --steps 160` regenerates the complete benchmark in one invocation on an unrestricted machine.

## Data construction

- 1488 hourly steps, with hourly and weekly components.
- Targets are three coupled synthetic demand series.
- Additional channels: past-only temperature, known-ahead calendar cosine, known-ahead scheduled pricing indicator.
- The autoregressive process has a daily lag, cross-channel lag, temperature influence, scheduled pricing effect, and additive noise.
- Training stops at hour index 864. In-domain forecast origins are 864 through 1128, every 24 hours.
- Regime shift starts at hour index 1152. Shift forecast origins are 1152 through 1464, every 24 hours.
- Every forecast is 24 hours ahead, conditioned on 96 observed hours. This is a fixed origin holdout without incremental retraining.

## Competing methods

1. Seasonal naive: repeat the previous 24 values for each target.
2. Regularized direct multivariate ridge: compact lags, past observed weather, and legally known future calendar and schedule features.
3. Ridge without future-known features: same ridge family, retrained without future schedule/calendar features.
4. Miniature alternating-attention patch model: 8-step patches, 24-dimensional embeddings, two alternating blocks, four attention heads, 160 training steps, jointly forecasts three target signals.
5. Temporal-only model: a separately trained small network with cross-variate attention disabled.
6. No-future-covariate model: a separately trained small network with horizon calendar and schedule features masked.

All models except the seasonal baseline see the same training period. They do not receive the same architecture or optimization budget, and the toy neural network is not tuned for best test performance. All models use 96 observed time steps and 24-step forecast horizon. Seasonal naive deliberately does not use covariates.

## Metrics

- MAE and RMSE average per absolute error across origins, targets, and horizons.
- MASE uses the training-period seasonal naive absolute-difference scale for each target, computed with 24-hour lag. Each target's scaled errors are averaged equally.
- `paired_origin_bootstrap_delta` draws origins with replacement and forms a paired difference in raw MAE. While horizons do not overlap, origin-level errors may still be autocorrelated, so the percentile bounds should be considered exploratory, not rigorous confidence intervals under arbitrary serial correlation.
- The saved results report across-seed mean and sample standard deviation of MASE, not a formal 95% confidence interval.
- The synthetic test is designed with a sharp structural break. It is not suitable for claiming population-level forecasting model superiority.

## Observed five-seed results

In-domain mean MASE and across-seed sample standard deviation:

- Seasonal naive: **0.98507 ± 0.01876**
- Ridge multivariate: **1.38615 ± 0.07628**
- Ridge without future covariates: **1.97920 ± 0.13110**
- Mini alternating-attention: **0.89882 ± 0.02406**
- Mini temporal-only: **3.60984 ± 1.25124**
- Mini trained without known future covariates: **4.10601 ± 1.69223**

Shifted mean MASE and across-seed sample standard deviation:

- Seasonal naive: **1.20975 ± 0.03145**
- Ridge multivariate: **4.39003 ± 0.31406**
- Ridge without future covariates: **4.97102 ± 0.41741**
- Mini alternating-attention: **4.58516 ± 0.33173**
- Mini temporal-only: **6.26107 ± 0.93425**
- Mini trained without known future covariates: **6.18302 ± 1.12469**

Key negative result: seasonal naive is much stronger after the structural break. A trained model that exploits historic cross-series relationships may fail when those relationships change. The ablations should not be interpreted as architecture rankings because brief training and different feature access create strong confounders.

## What was not measured

Actual TimesFM-3 accuracy, actual TimesFM-3 load time or latency, GPU energy consumption, production memory requirements, commercial costs, multiple independent real-world data sources, cross-model calibration, theoretical expressivity, and source paper benchmark ranks. The optional `timesfm3_eval.py` script is supplied for users to measure official model inference independently, under the model's license.
