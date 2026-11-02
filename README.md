# Multivariate Forecasting Lab: TimesFM-3 Architecture and Honest Evaluation

A reproducible teaching project for understanding multivariate time-series forecasting, temporal and variate attention, leakage-safe rolling-origin evaluation, and distribution shift.

**Independent educational implementation.** This project is **not** Google's TimesFM-3 model, does not contain its weights, and does not claim to reproduce the official GIFT-Eval, fev-bench, or TIME results. An explicitly optional script invokes the official model through Google's API when the user installs it and downloads the checkpoint under the applicable license.

## What's inside

- `forecast_lab/data.py`: synthetic multichannel hourly energy system, temporal origin splits, future-variable masking, training-only standardization, and the optional real ETTh1 loader.
- `forecast_lab/models.py`: seasonal naive, direct multi-output ridge, and a compact PyTorch patch model with **causal temporal attention followed by cross-variate attention**.
- `forecast_lab/train.py`: deterministic, leakage-safe supervised training of a small scratch model.
- `forecast_lab/metrics.py`: MAE, RMSE, MASE with training-only seasonal denominator, and paired origin-level bootstrap.
- `forecast_lab/benchmark.py`: five-seed rolling-origin evaluation, six model configurations, before and after a known synthetic regime shift.
- `forecast_lab/timesfm3_eval.py`: **optional actual TimesFM-3 inference** through Google's official PyTorch API, comparing joint multivariate mode against three individual univariate calls.
- `forecast_lab/download_etth1.py`: optional ETTh1 dataset downloader. Original: [ETDataset](https://github.com/zhouhaoyi/ETDataset).
- `tests/`: unit and causality tests.
- `results/benchmark.json`: five independently run and combined seed results, **synthetic** only.
- `figures/`: two figures for a Medium article.
- `ARTICLE.md`: complete tutorial.

## Prerequisites

Python 3.10 or later. A GPU is **not required** for the educational benchmark. PyTorch and NumPy are required.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the default benchmark

```bash
python -m unittest discover -s tests -v
python -m forecast_lab.benchmark --seeds 5 --steps 160 --output results/benchmark.json
python plot_results.py
```

The saved five-seed benchmark was produced with the same arguments. Training is done from scratch per seed. Metrics are synthetic MASE, computed over three targets and non-overlapping 24-hour horizons.

To run one seed at a time, useful on restricted CPU environments:

```bash
python -m forecast_lab.benchmark --seeds 1 --start-seed 0 --steps 160 --output results/seed0.json
python -m forecast_lab.benchmark --seeds 1 --start-seed 1 --steps 160 --output results/seed1.json
```

## Reproduce on real transformer-operations data

ETTh1 is a public real-world **electricity-transformer operations dataset, not a national electricity demand dataset**. The project loads HUFL, MUFL, and OT as target channels, HULL as a past-only observed covariate, and hour-of-day sine and cosine as legally known future covariates.

```bash
python -m forecast_lab.download_etth1 --output data/ETTh1.csv
python -m forecast_lab.benchmark --dataset etth1 --data-path data/ETTh1.csv \
  --seeds 3 --steps 160 --output results/etth1_benchmark.json
```

The real-data path requires an internet connection once to download the CSV. It is **not** included as a claimed completed experiment in the article. Its later-period evaluation split is chronological, and is not proof of a measured regime shift.

## Optional: Evaluate actual TimesFM-3

Read the [official repository](https://github.com/google-research/timesfm) and the [TimesFM-3 model license](https://huggingface.co/google/timesfm-3.0-pytorch) **before downloading the weights**. The downloadable weights are licensed for **non-commercial, non-production** use. Google offers authorized commercial access through BigQuery.

```bash
pip install 'timesfm[torch]'
python -m forecast_lab.timesfm3_eval --dataset synthetic --device cuda \
  --max-origins 6 --output results/real_timesfm3.json
```

For real ETTh1:

```bash
python -m forecast_lab.timesfm3_eval --dataset etth1 --data-path data/ETTh1.csv \
  --device cuda --max-origins 6 --output results/real_timesfm3_etth1.json
```

This step **downloads** the official checkpoint if not present, needs adequate RAM/VRAM, and is not run as part of the shipped benchmark. It runs one multivariate call and three univariate calls per origin, records model-loading time separately from prediction latency, and computes point errors, quantile pinball loss, and empirical 10th-to-90th percentile coverage. **Comparison caveat:** the joint call receives additional legally available covariates, while the three univariate calls receive only their own target histories. This comparison measures the complete multivariate-with-covariates configuration against a target-only control, not an isolated effect of variate attention. On a CPU, change `--device cuda` to `--device cpu`, with potentially much longer runtimes. The TimesFM package and API may change; follow the official repo if installation behavior differs.

## Five-seed synthetic benchmark

Mean MASE (lower is better):

- In-domain: seasonal naive **0.985**, direct ridge **1.386**, mini alternating attention **0.899**.
- Shifted regime: seasonal naive **1.210**, direct ridge **4.390**, mini alternating attention **4.585**.
- Architectural controls (in-domain): mini temporal-only **3.610** and mini retrained without future-known covariates **4.106**.

The toy alternating-attention model improves the baseline before the break but fails after the synthetic structural shift. This is **not** evidence about TimesFM-3 performance. These results are sensitive to the deliberately small network, optimizer steps, data-generating process, and holdout design. `BENCHMARK_NOTES.md` details every assumption.

## Key evaluation safeguards

- **Causal origins:** Input windows include observations at times strictly earlier than the forecast origin. Targets at and after the origin are never revealed.
- **Covariate availability:** Observed weather and target channels are hidden in the horizon, while deterministic calendar and planned pricing are available.
- **No tuning on test:** All models fit on observations ending at or before the fixed training cutoff.
- **Training-only scaling:** Mean and standard deviation are fit to the training portion only.
- **Non-overlapping outcomes:** Evaluation origins are 24 hours apart for a 24-hour horizon.
- **Stable MASE denominator:** The per-series seasonal-naive denominator is estimated only on the original training period, even when evaluating after the shift.
- **Paired uncertainty:** The code bootstraps errors at the origin level. This does not completely address temporal autocorrelation; block-by-week or cluster-aware uncertainty is recommended for serious publication.

## What this lab deliberately does not do

It does not train a 330M parameter foundation model, reproduce Google's 20-layer network, implement the original contiguous-patch masking or exact lookahead tokenization, replicate nine quantile heads, establish causal relationships among channels, or prove competitive zero-shot accuracy. The optional official adapter is the only component intended to execute actual TimesFM-3.

## Sources

- [Google Research: TimesFM-3](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- [Official TimesFM GitHub](https://github.com/google-research/timesfm)
- [Official pretrained checkpoint and license](https://huggingface.co/google/timesfm-3.0-pytorch)
- [BigQuery TimesFM-3 announcement](https://cloud.google.com/blog/products/data-analytics/bigquery-timesfm-3-0-with-multivariate-time-series-forecasting/)
- [GIFT-Eval](https://arxiv.org/abs/2410.10393)
- [fev-bench](https://arxiv.org/abs/2509.26468)
- [ETDataset](https://github.com/zhouhaoyi/ETDataset)

## License

The original educational code in this ZIP may be reused under the terms stated in `LICENSE`. Google TimesFM code, Google pretrained weights, and third-party dataset rights are **separate** and are **not** relicensed by this project.
