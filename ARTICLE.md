# Google Built a 330-Million-Parameter Model That Forecasts Multiple Time Series in One Pass. We Tested What Happens When the World Changes.

## A deep technical tutorial on TimesFM-3, multivariate attention, future covariates, leakage-safe backtesting, and why a seasonal baseline can beat a neural forecaster after distribution shift

*Google Research's TimesFM-3 predicts correlated targets and nine forecast quantiles without task-specific fine-tuning. This tutorial takes apart the architecture, builds a small PyTorch analogue, runs five-seed rolling-origin experiments, and provides a separate adapter for evaluating Google's actual checkpoint.*

A forecasting system can look excellent for months and still be one operational change away from failure.

Imagine a utility forecasting electricity usage across three neighborhoods. It has four years of hourly demand data. The three demand curves move together, respond to temperature, and exhibit predictable daily cycles. A model that sees the three curves jointly should discover patterns that a model looking at each neighborhood independently cannot. It might learn that a rise in demand in one region frequently precedes a rise in another.

Then the utility changes its pricing schedule. A major industrial customer shifts overnight operations. Electric-vehicle charging moves to different hours. The historical relationships remain in the training set, but the future no longer follows them.

The expensive model produces an elegant forecast. Yesterday's value repeated at the same hour might be closer.

That tension is the reason Google's TimesFM-3 deserves a serious engineering investigation rather than another announcement recap.

On August 31, 2026, Google Research introduced a **330-million-parameter time-series foundation model**, trained on more than **one trillion time points**, that can jointly forecast related targets and incorporate two different types of external information. It groups data into patches, alternates attention over time and over variables, and predicts an entire horizon in one model pass rather than generating forecast patches one after another. It produces a median forecast and nine quantiles, from the 10th to the 90th percentile. [1]

Google's reported benchmark results are impressive. But my central question for this tutorial is different:

**How do we know when joint modeling is actually better, when its inputs are legally available at prediction time, and whether the improvement survives a change in the underlying system?**

To answer that, we will reconstruct the core ideas in small, readable PyTorch components, build forecasting baselines that do not have to be fashionable to be competitive, and test a controlled structural break. We will also provide an optional evaluation script that calls Google's real TimesFM-3 weights. The script is deliberately separate from the completed experiments, because claiming to have benchmarked a model that was never run is a much worse research practice than reporting an underwhelming baseline.

**The independent lab's early result is a useful warning:** its small cross-series attention forecaster has a mean MASE of **0.899** before a controlled regime shift, compared with **0.985** for seasonal naive. After the shift, the neural model deteriorates to **4.585**, while seasonal naive records **1.210**. These are five-seed results from our synthetic teaching benchmark. They say nothing directly about TimesFM-3's real-world accuracy, but they reveal exactly the kind of failure a benchmark should be designed to expose.

---

## Table of contents

1. What TimesFM-3 actually changes
2. Univariate forecasting is not the same task
3. The three kinds of time-series information
4. The 2D token grid behind multivariate attention
5. Why patching is important
6. Temporal attention and cross-variate attention
7. The subtle problem with known future variables
8. Contiguous patch masking and single-pass forecasts
9. Nine quantiles and what uncertainty really means
10. What Google's benchmark claims establish
11. The reproducible workshop and research boundaries
12. Building a controlled electricity system
13. Creating leakage-safe rolling forecast origins
14. Implementing seasonal naive
15. Implementing direct ridge regression
16. Building alternating attention in PyTorch
17. Making the forecast horizon visible but not the target
18. Training the miniature forecaster
19. Choosing fair forecasting metrics
20. Five-seed results before and after the shift
21. Ablation experiments and their confounders
22. Reproducing the entire experiment
23. An optional real-world ETTh1 evaluation
24. Running the official TimesFM-3 checkpoint
25. Evaluating probabilistic forecasts
26. How to design a real forecasting benchmark
27. Failure modes, operational risks, and licensing
28. What I would test next
29. Conclusion
30. References

---

## 1. What TimesFM-3 actually changes

Start with what the model is, not what the headline implies.

The original TimesFM family offered pretrained, general-purpose forecasting models. Rather than train a new forecaster on every new dataset, a user supplied recent observations and obtained predictions. Google's earlier releases concentrated on univariate forecasting, where the core input for each forecast was the history of the series being predicted. Even when external regressors were integrated into particular prior workflows, the backbone's native forecasting formulation remained predominantly univariate. [1, 2]

TimesFM-3 changes the interface and the computation. It can process several target series together, alongside past-only measurements and dynamic variables whose future values are genuinely known. Its attention stack mixes information along two axes: historical time and contemporaneous variables. The result is a forecast for many targets in one model invocation. [1]

The official model card identifies a **20-layer** model with hidden size **1,280**, **16 attention heads**, **32-step context patches**, and **64-step forecast patches**. The weights are available from Google on Hugging Face. [3]

This distinction matters: **single-pass horizon generation** means the model need not perform an autoregressive call for each forecast patch. It does not mean constant inference time regardless of sequence length, the number of variables, or hardware. Attention, memory movement, patch count, and output construction still cost compute.

TimesFM-3 is not a causal discovery model. Its cross-variate attention can exploit statistical dependencies, but attending to an input does not establish that the input causes an outcome. If three energy loads correlate because all respond to the same temperature, attention may help forecast them jointly without identifying the temperature-mediated mechanism.

Nor is it the first research model to predict multiple time series. The interesting contribution is the combination of broad pretraining, flexible covariate roles, native multivariate forecasting, and a specific efficient attention and masked-decoding design in a deployable foundation-model family.

## 2. Univariate forecasting is not the same task

Consider three hourly demand series: residential, commercial, and industrial.

A univariate model receives the last four days of residential demand and predicts the next day of residential demand. To forecast the other two curves, it can run separate calls on their own histories. A multivariate model receives all three histories together and may learn that a change in one series is informative about the others.

This is not automatically an advantage.

A correlated series can be noisy, redundant, delayed, or unreliable. A cross-series dependency learned last year can reverse when a new pricing program is introduced. A new covariate can leak the future if its value would not have been available when the original forecast was made. And a much larger group of input series can create new latency and memory costs.

The legitimate question is not simply whether we should always use multiple variables. It is **whether additional variables improve the forecast when those variables are available, relevant, and stable enough to justify their complexity**.

That is also why the best experiment contains a univariate control, a multivariate model, and an explicit experiment with the supposed extra information removed.

## 3. The three kinds of time-series information

TimesFM-3's input categories are operationally meaningful. [1]

**Targets** are the variables we are attempting to forecast. In our energy example, these are the three demand series. All of their historical values are available; their values beyond the forecast origin are unknown.

**Past-only covariates** are other variables observed up to the forecast origin. Historical temperature, a measured sensor value, server utilization, or actual web traffic can help explain past behavior. But their realized future values cannot be smuggled into the model at inference time.

**Past-future covariates** are values legitimately known for both history and the forecast horizon. Day of week, a published public-holiday calendar, a committed promotion schedule, and a previously announced tariff are reasonable examples. This group can also include a genuine *forecast* of a future variable, such as the weather forecast available when predictions are issued. It cannot include future measured weather retrieved retrospectively.

A dataset is not safe merely because a column is labeled `covariate`. Its legal availability depends on when and how the value was produced. A pricing schedule might have been revised after the forecast was issued. A weather forecast retrieved today is not the same forecast that existed three months ago. A vendor feature may secretly incorporate future observations through a rolling aggregation.

If you work on time-series systems professionally, **feature availability is a time-indexed contract**, not an informal naming convention.

The workshop uses six channels:

- Three demand targets.
- One historical-only measured temperature channel.
- One calendar signal known ahead of time.
- One scheduled pricing indicator known ahead of time.

Only the final two are allowed to retain their values within the prediction horizon.

## 4. The 2D token grid behind multivariate attention

Most people picture a Transformer sequence as one line of tokens. TimesFM-3's design is easier to understand as a rectangular grid.

Imagine five series and twelve historical patches. Each position in the grid stores a vector representing one series over one time interval. A temporal attention operation travels along a row. It asks what earlier parts of the same series matter. A variate attention operation travels down a column. It asks which other series at the corresponding patch position supply useful context.

Conceptually:

```text
                         TIME PATCHES
                  p0      p1      p2      p3      ...
Residential        R0      R1      R2      R3
Commercial         C0      C1      C2      C3
Industrial         I0      I1      I2      I3
Temperature        T0      T1      T2      T3
Known calendar     K0      K1      K2      K3

Temporal mixing:  move along each series, causally.
Variate mixing:   move across series at one patch index.
```

Two pieces of information become jointly useful through repeated mixing. The industrial series can inform a residential patch; its representation can then influence subsequent time patches. In an actual architecture, residual projections, normalization, positional information, and the mask construction determine exactly which paths information can follow.

Google's public write-up describes causal temporal attention alternating with full variate attention. The important restriction is that unknown future target values cannot be exposed through any route, including another series' tokens. [1]

## 5. Why patching is important

Processing each hourly observation as an independent token becomes expensive when the historical context spans months or years, particularly with many channels.

Instead, TimesFM-3 packages **32 consecutive steps** into one context token. With 3,072 hours of observations, that is 96 time patches per series rather than 3,072 individual hourly tokens, before considering any additional structure. Patching compresses local patterns into a representation that an attention layer can operate on more economically. [1]

There is a tradeoff. Patch representations need to preserve the information relevant to later forecasts. Sharp transients, irregular sampling, and sparse rare events can be harder to express if the tokenization or training procedure does not preserve them. For a 15-minute workload, a 32-step patch contains eight hours. Whether that is helpful depends on the domain, not on a universal property of Transformers.

Our teaching implementation intentionally uses **8-step patches**, not 32. It has much smaller embeddings and two blocks instead of Google's 20-layer network. The experiment is designed to make the computation inspectable on a CPU, not to replicate the pretrained architecture.

## 6. Temporal attention and cross-variate attention

Inside one educational block, the input tensor has shape:

```text
[batch, variables, patches, hidden_dimension]
```

To perform temporal attention, reshape it to:

```text
[batch * variables, patches, hidden_dimension]
```

Each series is now treated as its own time sequence. We apply a causal mask so patch `p3` cannot attend to `p4`. Then reshape the result back into the grid and transpose the middle axes for variate attention:

```text
[batch * patches, variables, hidden_dimension]
```

This second attention operation mixes the available variables for each patch position.

Here is the core of the independent implementation:

```python
class AltAttentionBlock(nn.Module):
    def __init__(self, d=24, heads=4, cross=True):
        super().__init__()
        self.cross = cross
        self.temporal = nn.MultiheadAttention(
            d, heads, batch_first=True
        )
        self.variate = nn.MultiheadAttention(
            d, heads, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )

    def forward(self, x):
        batch, variables, patches, width = x.shape

        temporal_input = x.reshape(
            batch * variables, patches, width
        )
        future_mask = torch.triu(
            torch.ones(
                patches, patches,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        updated, _ = self.temporal(
            temporal_input,
            temporal_input,
            temporal_input,
            attn_mask=future_mask,
            need_weights=False,
        )
        x = self.ln1(
            x + updated.reshape(batch, variables, patches, width)
        )

        if self.cross:
            across_variables = (
                x.permute(0, 2, 1, 3)
                 .reshape(batch * patches, variables, width)
            )
            updated, _ = self.variate(
                across_variables,
                across_variables,
                across_variables,
                need_weights=False,
            )
            x = self.ln2(
                x + updated.reshape(
                    batch, patches, variables, width
                ).permute(0, 2, 1, 3)
            )

        return self.ln3(x + self.ff(x))
```

The official model uses a more sophisticated architecture and masking system. This block captures the **idea of alternating the two attention axes**, not the complete TimesFM-3 implementation.

Notice an important experimental confounder: if we turn off cross-variate attention entirely, a target's own sequence cannot receive calendar or temperature features stored in other channels. Therefore, a performance drop from disabling cross-variate attention measures a combination of lost cross-series mixing **and lost exogenous information access**. It is not a pure estimate of how much information was contributed by target-to-target correlations.

A better isolated experiment would preserve covariate fusion and only block target-to-target cross-attention. We leave that as an explicit extension rather than claim a result our architecture does not support.

## 7. The subtle problem with known future variables

Google's lookahead token construction makes this issue particularly interesting. For variables known across the forecast horizon, TimesFM-3 can combine current and future patches into its token representation. The model can therefore incorporate a planned promotion or a weekend indicator as it predicts demand. [1]

You might wonder whether looking into the future violates temporal causality.

It does not, **provided the future information is actually available at prediction time**.

Forecasting does not require ignorance of the future calendar. It requires ignorance of future outcomes and unavailable measurements. A demand forecaster is allowed to know that Christmas occurs next Thursday. It is not allowed to know next Thursday's realized sales unless they have somehow already occurred.

This distinction often gets lost in model benchmarking. One code path loads all feature columns for the complete dataset. Another code path slices windows at the forecast origin. If the second path forgets to mask the future of a measured covariate, the model can appear to achieve extraordinary accuracy for a reason that vanishes in production.

Our benchmark makes this boundary explicit:

```python
def input_visibility(x, context, target_count=3, past_only_count=1):
    x = np.asarray(x).copy()
    observed_mask = np.ones_like(x, dtype=np.float32)

    unknown = target_count + past_only_count
    x[:, context:, :unknown] = 0
    observed_mask[:, context:, :unknown] = 0

    return x, observed_mask
```

Two invariants are important. Future target and temperature values become zero and are marked absent. Future calendar and tariff features remain visible. The tests also replace all unknown future measurements with enormous fake values and verify that the model input stays identical. If your forecasting pipeline cannot pass this kind of invariance test, its accuracy metrics are suspect.

## 8. Contiguous patch masking and single-pass forecasts

One of TimesFM-3's most useful changes is its forecast generation strategy.

An autoregressive forecaster may predict the next patch, append that prediction to the sequence, and call the model again. Errors from earlier forecast patches become inputs for later patches. Even if each patch prediction is individually good, accumulated errors can cause drift.

TimesFM-3 instead uses **Contiguous Patch Masking**. Future target patches are included as masked placeholders. Future-known covariates remain available. The model processes the combined context and horizon and produces forecast outputs without running a separate autoregressive pass for each patch. [1]

Think of this as asking one model computation to fill in a contiguous missing region, with the necessary information-access constraints enforced by its input construction and attention pattern.

There are important qualifications. Predicting the entire horizon in one call removes a particular sequential decoding dependency. It does not guarantee identical accuracy at every horizon, eliminate uncertainty accumulation in the underlying system, or make inference memory free. It also does not guarantee that all possible horizon lengths have the same latency.

In the educational project, we mimic the broad interface: the miniature model receives observed context and masked future observations, then its final head emits all 24 future hours at once. It is **not** a reproduction of Google's contiguous-patch masking loss, its 64-step output patches, or its iterative normalization pipeline.

## 9. Nine quantiles and what uncertainty really means

Point accuracy is only half the forecasting problem.

A grid operator cannot plan reserve capacity from a single number alone. A warehouse cannot manage safety stock without a view of demand uncertainty. A network operations team needs to know not only the expected traffic but the plausible extreme traffic.

TimesFM-3 predicts nine quantiles, corresponding to 10%, 20%, and so on through 90%, for each target and horizon step. [1, 3]

For a demand forecast, the 10th percentile is a value that the predicted distribution places near its lower tail. The 90th percentile is near the upper tail. An interval between them represents a nominal central **80% prediction interval**, not a guarantee that 80% of outcomes will fall inside it.

Calibration must be evaluated. On truly unseen forecast origins, does approximately 80% of the realized demand lie between those lower and upper quantiles? Do predictive intervals widen during volatility? Are they systematically too narrow during operational shocks?

For each quantile, one can compute **pinball loss**. If the actual outcome lies above the predicted quantile, the underprediction incurs a penalty weighted by the quantile probability. If it lies below, the overprediction receives the complementary weight. This asymmetric scoring rule encourages a correct conditional quantile rather than a mean forecast.

We implement pinball loss and empirical interval coverage in the optional real-model evaluator. The main synthetic benchmark is point-forecast only, and we do not invent probabilistic results that were never measured.

## 10. What Google's benchmark claims establish

Google reports that TimesFM-3 ranked first among the tested pretrained foundation models on GIFT-Eval, fev-bench, and TIME when it launched, and that full multivariate mode improved on its univariate results. [1, 2]

Those claims should be read in their research context, not as a timeless industry leaderboard. The set of evaluated models, public submissions, benchmark definitions, and available code change. Newer releases have subsequently reported beating TimesFM-3 on at least some public leaderboards. Google was describing comparisons at a particular point in time, not proving permanent superiority. [10]

The benchmarks themselves are worth studying.

**GIFT-Eval** aims to test general-purpose forecasting across diverse datasets, granularities, domains, and horizons. Its paper describes 23 datasets covering roughly 144,000 series and 177 million observed points. [5]

**fev-bench** emphasizes realistic task definitions, including exogenous covariates, and reports aggregated model comparisons with uncertainty-aware evaluation. Its original paper describes 100 tasks across seven domains, including 46 with covariates. [6]

**TIME**, published at ICML 2026, focuses on more recent and carefully curated datasets, realistic predictability assumptions, data integrity, and pattern-level evaluation. It includes 50 datasets and 98 forecasting tasks. [7]

These efforts matter because even a flawless forecasting model can be awarded misleading scores by a benchmark that accidentally leaks future information, blends highly related train and test series, lets developers tune to the test period, or aggregates incomparable tasks in a convenient but uninformative way.

Also distinguish **zero-shot generalization** from an ordinary locally trained neural-network result. TimesFM-3 starts with general pretrained weights and applies them to a new dataset without a task-specific fitting stage. Our miniature Transformer trains directly on our synthetic electricity data. Its accuracy is not directly comparable to zero-shot foundation-model performance.

## 11. The reproducible workshop and research boundaries

The companion repository is named `timesfm3_forecasting_lab`. It contains everything needed to reproduce the completed synthetic experiment on a CPU, along with scripts for optional real data and optional real TimesFM-3 inference.

A good research companion must say what it does **not** do. Ours does not train a 330-million-parameter model, reproduce a trillion-point pretraining corpus, reproduce Google's evaluation leaderboard, or run its checkpoint as part of the saved benchmark. It is a deliberately small demonstration of related engineering principles.

The project includes:

```text
timesfm3_forecasting_lab/
  forecast_lab/
    data.py
    models.py
    train.py
    metrics.py
    benchmark.py
    timesfm3_eval.py
    download_etth1.py
  tests/
    test_lab.py
  data/
  results/
    benchmark.json
    seed0.json ... seed4.json
  figures/
    01_multivariate_vs_shift.png
    02_structural_break.png
  plot_results.py
  README.md
  BENCHMARK_NOTES.md
  ARTICLE.md
  requirements.txt
  LICENSE
```

`ARTICLE.md` is the article you are reading. The benchmark JSON and methodology notes are included so a reader can investigate the results rather than trust a screenshot.

## 12. Building a controlled electricity system

We deliberately use simulated data for the first experiment because synthetic data lets us control what changes and when.

The generator creates 1,488 hourly observations, a little over eight weeks. There are three targets with distinct baseline demand levels and daily amplitudes. Demand also responds to temperature, a known scheduled-tariff indicator, a daily lag, and another channel's recent activity. Noise is added to each series.

Here is the core generator, simplified for readability:

```python
for hour in range(24, n):
    shifted = hour >= shift_start

    for channel in range(3):
        neighbor = (
            channel + (2 if shifted else 1)
        ) % 3

        temperature_effect = (
            sensitivity[channel]
            * (1.7 if shifted else 1.0)
            * abs(weather[hour] - 20)
        )

        base_load = (
            level[channel]
            + amplitude[channel] * daily_cycle[hour]
            + temperature_effect
        )

        if shifted:
            base_load += shift_offset[channel]

        demand[hour, channel] = (
            0.52 * demand[hour - 24, channel]
            + 0.10 * demand[hour - 1, neighbor]
            + 0.38 * base_load
            + tariff_effect[channel] * tariff[hour]
            + noise()
        )
```

At hour **1,152**, three things change. The neighbor relationship used by the cross-series component changes. Temperature sensitivity increases. The level of each target shifts by a different amount. This is a designed structural break, not a random test-set fluctuation.

The point is not to prove that deep learning is fragile in every setting. It is to create a controlled situation where historical correlations become less trustworthy, then ask whether a model knows how to respond.

We hold out the in-domain period before the break and a later period after it. None of the models is retrained or retuned on the shifted period.

## 13. Creating leakage-safe rolling forecast origins

Randomly shuffle rows in a time-series dataset and you are no longer evaluating a realistic forecasting problem. A model trained on next Thursday should not be tested on last Monday as if it had made that prediction before Thursday occurred.

Instead, we define a forecast **origin** as the point in time where the prediction is issued.

At origin `t`, a model may observe `[t - context, t)`, a window that ends immediately before `t`. It predicts `[t, t + horizon)`, which lies in the future from that origin.

With a 96-hour context and 24-hour forecast horizon, the lab creates rolling origins 24 hours apart. Neighboring forecast horizons do not overlap, which simplifies evaluation and reduces dependence caused by scoring the same observed hour multiple times.

```python
def make_windows(ds, context, horizon, start, stop, stride=24):
    origins = np.arange(
        max(context, start),
        min(stop - horizon, len(ds.values) - horizon) + 1,
        stride,
    )

    windows = ds.values[
        np.array([
            np.arange(t - context, t + horizon)
            for t in origins
        ])
    ]

    targets = np.array([
        ds.values[t:t + horizon, :3].T
        for t in origins
    ])

    return windows, targets, origins
```

The full code checks for positive lengths and valid windows. The benchmark then applies the visibility mask before passing each window to the neural model. Ridge regression receives engineered lag features plus future covariates, never the future targets.

Training is limited to hours before index 864. In-domain evaluation starts at index 864 and ends before the structural break at 1,152. Shifted evaluation begins exactly at the structural break.

**One non-obvious point:** the training set consists of overlapping rolling examples. That is acceptable for supervised sequence learning, but those examples are not independent draws. Our five random seeds and bootstrap procedure do not magically turn them into independent datasets.

## 14. Implementing the baseline nobody should skip

The seasonal naive baseline says that the next day's hourly shape will look like the last observed day.

In code:

```python
def seasonal_naive(raw_windows, context=96, horizon=24):
    history = raw_windows[:, :context, :3]
    previous_day = history[:, context - 24:context, :]
    return previous_day.transpose(0, 2, 1)
```

This predictor has no trained parameters. It uses no historical data beyond the context. It makes no attempt to model cross-series relationships, external variables, or price effects.

It is still an essential benchmark.

For time series with strong daily or weekly repetition, a seasonal naive forecast can be surprisingly difficult to beat. A deep model that does not outperform it in the deployment conditions of interest has not earned its complexity. A highly tuned seasonal model might perform even better; our implementation intentionally uses the simplest defensible version.

## 15. Implementing direct ridge regression

The next control is a supervised statistical baseline that has access to information the seasonal predictor ignores.

We construct a feature vector from the most recent target observations, the values 24 and 48 hours ago, the mean over the last 24 hours, observed temperature at the previous hour, and the calendar and tariff schedule for the entire prediction horizon.

The regression is **direct multi-output ridge**: a single linear system predicts all 72 outputs, representing 3 target channels multiplied by 24 forecast hours. It does not recursively predict one step and then feed the result back.

```python
class RidgeDirect:
    def __init__(self, alpha=200.0):
        self.alpha = alpha

    def fit(self, x, y):
        x = np.column_stack([np.ones(len(x)), x])
        y = y.reshape(len(y), -1)

        regularizer = np.eye(x.shape[1])
        regularizer[0, 0] = 0

        self.weights = np.linalg.solve(
            x.T @ x + self.alpha * regularizer,
            x.T @ y,
        )
        return self

    def predict(self, x, targets=3, horizon=24):
        x = np.column_stack([np.ones(len(x)), x])
        return (x @ self.weights).reshape(
            len(x), targets, horizon
        )
```

This is not SARIMA, a Kalman filter, or a sophisticated state-space forecaster. It is an interpretable, regularized direct regression control. It is useful precisely because it can leverage multivariate and known-future information without any attention layer.

The regularization weight is fixed ahead of the reported evaluation rather than selected by scanning the shifted test scores. We also rerun ridge with the future calendar and tariff inputs removed, so the experiment can separate some of their contribution from that of the lagged signals.

## 16. Building the miniature Transformer

Our full miniature model takes arrays of shape:

```text
values:        [batch, 120 hours, 6 channels]
observed_mask: [batch, 120 hours, 6 channels]
output:        [batch, 3 targets, 24 future hours]
```

The 120-hour input combines 96 observed hours and 24 future placeholders. With 8-step patches, each channel becomes 15 patch tokens. Each patch contains the values and the binary observation mask, giving 16 input elements before projection into a hidden representation of size 24.

A trainable channel embedding tells the network which variable it is looking at. A trainable position embedding identifies the time patch. Two alternating-attention blocks process the grid. A projection head converts the final three targets' future patch representations into 24 hourly forecasts per target.

```python
class MiniMaskedForecaster(nn.Module):
    def __init__(self, channels=6, targets=3,
                 context=96, horizon=24, patch=8,
                 d=24, heads=4, layers=2, cross=True):
        super().__init__()
        self.context = context
        self.horizon = horizon
        self.patch = patch
        self.targets = targets
        self.channels = channels

        patches = (context + horizon) // patch
        self.value = nn.Linear(2 * patch, d)
        self.channel_embed = nn.Parameter(
            torch.randn(channels, d) * 0.02
        )
        self.position = nn.Parameter(
            torch.randn(patches, d) * 0.02
        )
        self.blocks = nn.ModuleList([
            AltAttentionBlock(d, heads, cross=cross)
            for _ in range(layers)
        ])
        self.head = nn.Linear(d, patch)

    def forward(self, values, observed_mask):
        batch, length, channels = values.shape
        assert length == self.context + self.horizon
        assert channels == self.channels

        combined = torch.stack([
            values.transpose(1, 2),
            observed_mask.transpose(1, 2),
        ], dim=-1)

        patches = combined.reshape(
            batch, channels,
            length // self.patch,
            2 * self.patch,
        )
        x = (
            self.value(patches)
            + self.channel_embed[None, :, None, :]
            + self.position[None, None, :, :]
        )

        for block in self.blocks:
            x = block(x)

        future = x[
            :, :self.targets,
            self.context // self.patch:,
            :,
        ]
        return self.head(future).reshape(
            batch, self.targets, self.horizon
        )
```

We can now examine the training loop and ablations without needing a proprietary checkpoint or a GPU. Again, this network is independent teaching code. It does **not** have the 330M parameters, 32-step input patches, 64-step forecast patches, or pretrained knowledge of TimesFM-3.

## 17. Making the forecast horizon visible but not the target

The model receives a tensor containing future positions. That might sound like cheating until you inspect what is present at those positions.

Unknown values are replaced with zero and accompanied by mask zero. Known-future calendar and schedule values remain as actual inputs, with mask one. The model can predict against future placeholders but does not receive future demand or measured temperature.

The distinction is crucial for batch-oriented masked forecasting. A naive preprocessing function can accidentally expose a target column because all channels are concatenated before masking. Our tests check both the mask and the model's invariance to arbitrary changes in the hidden portion.

The complete workshop also tests temporal causality: changing future-known inputs for the final output patch must not change the prediction generated at an earlier output patch through the causal temporal path. This is a structural test of the toy model, not a proof that every implementation of multivariate forecasting is causally well designed.

## 18. Training the miniature forecaster

Before fitting the neural model, we compute each channel's mean and standard deviation using the training period only. We apply those same values during evaluation, even when the data distribution shifts. Recomputing the scaler on future test observations would give the model information unavailable at forecast issue time.

We create supervised training examples whose entire target horizon ends at or before the training cutoff. The network minimizes mean squared error on standardized target values. The default setup is deliberately small:

- 96-hour observed context and 24-hour prediction horizon.
- 8-step patches and hidden dimension 24.
- Two alternating-attention blocks, each with four attention heads.
- AdamW with learning rate 0.0015 and weight decay 0.0001.
- Batch size 32 and **160 optimizer updates** per seed.
- Five independently initialized runs with seeds 0 through 4.

The short training budget keeps the experiment reproducible on commodity CPUs. It does not demonstrate the best possible result for the neural architecture. Larger models, longer training, and extensive validation could change rankings substantially. A careful research paper would tune hyperparameters on an earlier validation period, freeze the decisions, and then run the final test exactly once.

Because our task is educational, the training code is intentionally compact:

```python
model = MiniMaskedForecaster(
    context=96, horizon=24, cross=True
)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0015, weight_decay=1e-4
)

for step in range(160):
    indices = sample_training_windows(batch=32)
    prediction = model(
        normalized_inputs[indices],
        visibility_masks[indices],
    )
    loss = (prediction - normalized_targets[indices]).square().mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

The names here communicate the training procedure; the executable indexing and data construction are in `forecast_lab/train.py`.

## 19. Choosing fair forecasting metrics

There is no single metric that makes every time series comparable.

**MAE**, mean absolute error, is straightforward and measured in the same units as the target. If an electricity forecast misses by ten units, MAE reflects that directly. But averaging absolute errors across differently scaled targets can cause the largest-magnitude target to dominate.

**RMSE**, root mean squared error, penalizes large misses more strongly. That can be valuable when rare large failures are especially costly, but it can also be unstable under outliers and make comparisons sensitive to a few spikes.

**MASE**, mean absolute scaled error, divides each target's absolute prediction error by a seasonal naive error scale estimated on training data. In our lab, the seasonal difference uses a lag of 24 hours. We first calculate a training-only scale for each target, then average the scaled errors equally across the three targets. A lower score is better. Scores below one indicate that the model's average absolute error is smaller than the **training-period seasonal-naive scale**, not necessarily smaller than the actual seasonal baseline's test error for the same split.

This last qualification is subtle. A model with MASE below one has not automatically beaten every seasonal naive forecast. That is why we report the actual seasonal naive score on the same evaluation origins.

For statistical uncertainty, we compute mean and sample standard deviation of MASE over five random seeds. We also include exploratory paired origin-level bootstrap comparisons of MAE. Paired comparisons are better than comparing unrelated averages because every forecaster sees the same forecast origins. However, nonoverlapping horizons do not eliminate autocorrelation between nearby origins, so the provided bootstrap intervals are not fully rigorous time-series confidence intervals.

For a production-quality study, use time-block bootstrapping or another dependence-aware uncertainty method, pre-register model selection criteria, and include task-level scores instead of only aggregate averages.

## 20. The actual five-seed results

Now the important part: what happened when the models encountered our controlled system?

The numbers below come from the independent synthetic experiment saved as `results/benchmark.json`. Each result is the mean MASE across seeds 0 through 4, followed by sample standard deviation. **Lower is better.** None of these values represents TimesFM-3.

### Before the regime shift

Seasonal naive achieved **0.985 ± 0.019**. It is a strong result for a method that only repeats yesterday's values.

Direct ridge with multiple targets and permitted covariates achieved **1.386 ± 0.076**. More information and more fitted parameters did not guarantee an improvement over seasonal repetition.

Our miniature alternating-attention forecaster achieved **0.899 ± 0.024**, outperforming seasonal naive on this in-domain synthetic holdout. In relative MASE terms, that is about an **8.8% reduction** against the seasonal predictor on the same experiment.

Ridge without future-known covariates scored **1.979 ± 0.131**, worse than ridge with those inputs.

The miniature temporal-only model scored **3.610 ± 1.251**, and the model separately trained without future-known covariates scored **4.106 ± 1.692**. Those are large, noisy losses of accuracy. They are evidence that these particular controls underperformed in our setup, not a clean causal allocation of performance to a single architectural component.

### After the synthetic structural break

Seasonal naive increased to **1.210 ± 0.031**. It got worse, as expected when the underlying system changes, but remained relatively robust.

Ridge with multivariate features increased to **4.390 ± 0.314**.

The miniature alternating-attention forecaster increased to **4.585 ± 0.332**.

Ridge without future-known covariates reached **4.971 ± 0.417**. The temporal-only neural ablation reached **6.261 ± 0.934**, and the no-future-covariate neural ablation reached **6.183 ± 1.125**.

In this setup, the richer predictive models relied on learned patterns that no longer described the changed system. The seasonal predictor only needed the recent daily shape, so its error rose less dramatically.

There is an especially important limitation in how to interpret this. Our after-shift test begins at the exact known break and uses a fixed, pretrained-on-synthetic-history model without any updating. That is intentionally stressful. A model with online adaptation, drift-aware calibration, covariate shift correction, or periodic refitting could respond differently. The experiment does not show that deep forecasting models cannot handle drift; it shows that **benchmarking without drift can conceal a serious failure mode**.

Figure 1 in the companion repository plots the five-seed MASE before and after the break. Figure 2 shows a sample demand series crossing the structural break. Both figures are generated from the saved experiment, not sourced from Google's paper.

## 21. Ablation experiments and their confounders

Ablations are valuable when they answer a narrow question. They are misleading when several things change at once.

We ran three main architectural controls in addition to the seasonal and ridge baselines.

**Cross-variate attention disabled.** This version keeps the temporal attention mechanism but removes channel mixing. Its in-domain MASE is 3.610, much worse than 0.899 for the full miniature network. However, disabling cross-variate attention also prevents separate calendar and measured-temperature channels from reaching the target representations. Therefore the difference cannot be attributed solely to inter-target dependence.

**Known future covariates masked.** We trained a separate miniature model that could not see the calendar and tariff channels in the future horizon. Its in-domain MASE is 4.106. The large gap suggests that legally known inputs were useful in this controlled task, but the model was trained briefly, and performance is highly seed-dependent. It would be irresponsible to generalize that magnitude to realistic electricity markets.

**Ridge without future covariates.** This version holds the model family fixed while removing known future schedule information. Its in-domain MASE rises from 1.386 to 1.979. Because the estimator class is unchanged, this comparison is easier to interpret as a feature-availability ablation, although feature-set dimensionality and the fixed regularization strength still affect it.

The right follow-up experiment would give each neural model identical access to covariates through a shared embedding and selectively disable **only target-to-target attention**. It would also include training-budget sweeps, multiple context lengths, multiple horizons, and distributions where cross-series relationships are positive, irrelevant, or intentionally adversarial.

An honest research article should state the confounder before someone else discovers it.

## 22. Reproduce the complete experiment

The companion ZIP contains Python source, the five seed outputs, the combined benchmark JSON, and two already generated figures.

From a terminal:

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows, use: .venv\Scripts\activate

python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# Full five-seed synthetic benchmark
python -m forecast_lab.benchmark \
  --seeds 5 \
  --steps 160 \
  --output results/benchmark.json

# Regenerate the figures
python plot_results.py
```

The test suite checks generated-data determinism, the shape and boundaries of forecast windows, nonoverlapping evaluation horizons, future-variable masking, invariance to hidden future targets, training-only scaling, basic metric correctness, model output shape, and a targeted temporal-causality property. It also checks the optional ETTh1 CSV parser using a temporary local fixture.

**Reproducibility note:** these results were produced with PyTorch 2.10.0+cpu and NumPy 2.3.5. Exact neural-network results can vary across platforms, torch kernels, and future dependency releases. The checked-in per-seed outputs provide a concrete reference; reproducing their numerical values to the final decimal is not guaranteed on all machines.

## 23. A real infrastructure dataset: ETTh1

Synthetic systems help us expose mechanisms. They do not establish performance in the wild.

The workshop therefore includes an optional path for the **ETTh1** dataset, part of the Electricity Transformer Temperature collection. Despite its name, ETTh1 is not a dataset of national electricity demand; it describes hourly transformer operating measurements. Our loader uses `HUFL`, `MUFL`, and `OT` as target channels, `HULL` as a past-only observed covariate, and hour-of-day sine and cosine as known-future calendar variables. [8]

```bash
python -m forecast_lab.download_etth1 \
  --output data/ETTh1.csv

python -m forecast_lab.benchmark \
  --dataset etth1 \
  --data-path data/ETTh1.csv \
  --seeds 3 \
  --steps 160 \
  --output results/etth1_benchmark.json
```

The first command needs internet access to retrieve the original public CSV. The benchmark uses chronological splits: first 60% for training, the next 20% for earlier evaluation, and the final 20% for later evaluation.

Do not call that last split a verified **distribution-shift period** simply because it occurs later in time. To make a scientific claim about drift in ETTh1, you would need to measure changing marginals, seasonality, autocorrelation, or cross-series dependencies, and provide evidence that the change matters to forecast error.

The optional real-data command is included for readers to run, but **its output is not part of the five-seed synthetic results reported above**. We are not inventing ETTh1 scores.

## 24. Run the actual TimesFM-3 checkpoint

Google distributes the official PyTorch model in the `google-research/timesfm` repository. The code examples use `TimesFM3Evaluator` and `ModelConfig` through `timesfm3`, with inputs represented as NumPy arrays. [2]

Here is an actual official-API example adapted to our variable roles:

```python
import numpy as np
from timesfm3 import TimesFM3Evaluator, ModelConfig

config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=8,
    device="cuda",
)
model = TimesFM3Evaluator(config)

context = 96
horizon = 24

# Historical demand for three targets: [3, 96]
targets = np.random.randn(3, context).astype(np.float32)

# Measured temperature is only known through the origin.
past_temperature = np.random.randn(1, context).astype(np.float32)

# Calendar and committed tariff are known across the horizon.
known_future = np.random.randn(2, context + horizon).astype(np.float32)

outputs = list(model.predict_batch(
    contexts=[targets],
    horizon=horizon,
    past_only_covariates=[past_temperature],
    past_future_covariates=[known_future],
    return_quantiles=True,
    use_symmetric_averaging=False,
))

forecast = outputs[0].forecast
quantiles = outputs[0].quantiles

print(forecast.shape)   # (3, 24)
print(quantiles.shape)  # (3, 24, 9)
```

In your own experiment, replace the random arrays with actual historical and legally known future values for a particular forecast origin. The separate `forecast_lab/timesfm3_eval.py` script does this for our synthetic dataset or ETTh1, runs a joint multivariate call and independent univariate calls, and records point error, quantile pinball loss, coverage, model-load time, and inference latency. **Do not interpret the difference between these two calls as a pure cross-variate-attention effect:** in this teaching adapter the joint model also receives historical temperature and known-future calendar inputs, while the univariate control receives only each target history. A truly equal-information model ablation requires a more carefully matched input setup.

Install the official dependencies separately:

```bash
python -m pip install 'timesfm[torch]'

python -m forecast_lab.timesfm3_eval \
  --dataset synthetic \
  --device cuda \
  --max-origins 6 \
  --output results/real_timesfm3.json
```

A CPU run may work with `--device cpu`, but can be substantially slower and require significant RAM. On Apple Silicon, the official repository also documents an MLX implementation, though our bundled evaluator is written against the PyTorch API. [2]

**The actual TimesFM-3 checkpoint was not downloaded or executed in the completed benchmark.** No TimesFM-3 error, speed, GPU memory, or energy number in this article is an invented local measurement. If you run the optional script, the results become your own hardware-specific measurements and should be reported with checkpoint revision, dependency versions, device, batch size, input dimensions, precision, and warmup settings.

## 25. Evaluating probabilistic forecasts

Once the real model is installed, evaluating only the median discards much of the value of its output distribution.

Our optional script computes pinball loss at nine quantile levels and coverage of the nominal central 80% interval. A small pinball loss rewards both sharpness and correct placement of quantiles. Coverage reports how frequently the realized target lies between the forecast 10th and 90th percentile.

For a rigorous probabilistic evaluation, also track the empirical coverage at all reported quantiles, interval widths, crossing violations if the model permits them, and proper scoring rules such as weighted quantile loss or CRPS approximations. Report these measures separately for different horizons and especially for shifted or unusually volatile periods.

A 90th percentile that predicts above 90% of realized outcomes on a held-out sample is calibrated marginally, but can still be badly wrong in the particular conditions an operator cares about. For example, a model might be calibrated across ordinary weeks while seriously underestimating peak demand during heat waves. Conditional calibration matters more than an attractive average when the loss of underprediction is asymmetric.

## 26. How to design a real forecasting benchmark

A meaningful comparison of TimesFM-3, seasonal naive, statistical models, and locally trained Transformers needs more than one dataset and one horizon.

**Freeze your task definition.** Decide which variables are targets, which are past-only features, and which will genuinely be known at forecast issue time. Document expected release delays and any missing data behavior.

**Use realistic forecast origins.** A rolling-origin benchmark should simulate repeated predictions from increasing points in time. Specify context length, horizon, stride, retraining policy, and whether the origins overlap. If a model is updated over time, account for the compute and information it receives during each update.

**Avoid tuning on the final test set.** Create a train period, a distinct time-ordered validation period, and a final untouched test interval. The origin schedule and selected metrics should be fixed before evaluating the final interval.

**Compare equivalent information budgets.** A multivariate model with access to the next month's known promotions should not be evaluated as if it were directly comparable to a target-only model, unless the information advantage is itself the object of study. Report both the univariate and multivariate TimesFM-3 modes, and provide multivariate statistical baselines that can consume the same covariates.

**Measure across multiple regimes.** Include holiday periods, system changes, unusual demand, missing sensors, and periods where correlations invert. Separate performance by regime so a good global mean cannot conceal poor behavior during the periods where a business is most vulnerable.

**Report uncertainty and operational cost.** Use paired errors on the same origins, time-block uncertainty estimates where appropriate, and disaggregated metrics. For a foundation model, report cold-load time separately from warm inference. Track latency, peak memory, model deployment costs, and failure rates.

**Treat zero-shot comparisons honestly.** A pretrained model has consumed extensive pretraining compute and data, while a scratch baseline trains only on local observations. Comparisons should make that distinction clear. No single metric answers every deployment question.

The design principles align with concerns raised by the GIFT-Eval, fev-bench, and TIME benchmark authors. [5, 6, 7]

## 27. Failure modes, operational risks, and licensing

A strong forecast can still be operationally unsafe.

**Future leakage** can enter through realized weather measurements, rescaled features computed using future statistics, revised event calendars, post-hoc labels, or joins that ignore data publication timestamps.

**Distribution shift** can result from consumer behavior changes, sensor upgrades, new holidays, pricing changes, infrastructure failures, new regulations, or a change in what a monitored variable measures. A model trained on stable correlations may produce very confident forecasts in a setting where the correlations no longer hold.

**Correlated errors** are particularly relevant to joint forecasting. If a shared latent representation or a bad common covariate pushes every target upward, the errors across targets may occur together. Downstream decisions based on independent forecast-error assumptions can then understate total portfolio risk.

**Data privacy and governance** matter because multivariate inputs can combine sensitive commercial signals across departments or organizations. A model does not have to emit individual records to expose information through predictions, logs, or poorly controlled inference requests. Follow the data permissions and isolation requirements of the deployment environment.

**Model licensing** matters too. The public TimesFM source repository uses Apache-2.0 for its source code, but Google's downloadable TimesFM-3 pretrained weights are under a separate **non-commercial, non-production license**. Google explicitly permits commercial use through authorized Cloud services under their own terms. Do not assume that a locally hosted checkpoint can legally power a paid forecasting product merely because the GitHub source is public. [2, 3, 4]

Finally, forecasting model outputs should not become automatic high-impact decisions without appropriate monitoring, fail-safes, and review. The correct response to an unfamiliar regime may be to widen intervals, ask for human review, or fall back to a known robust baseline.

## 28. What I would test next

If I were extending this workshop into a publishable academic benchmark, I would prioritize experiments that resolve the most important remaining ambiguities.

First, add an **equal-information architecture ablation**. Give every model the same known-future covariates through a shared pathway and disable only cross-target attention. This would distinguish whether performance gains come from related target histories or simply from being able to ingest the calendar.

Second, add a genuine **walk-forward retraining protocol**. Our static-model break experiment is intentionally adverse. Refit ridge and the miniature Transformer at fixed intervals, then measure whether the increased adaptation compute is justified by accuracy improvements. Carefully exclude future labels at each retraining point.

Third, run the **real TimesFM-3 checkpoint** in univariate and multivariate modes on the exact same origins. Since the pretrained model can evaluate without local fitting, compare forecast quality and inference cost against seasonal naive, ridge, and a properly validated small Transformer. Publish code and predictions, not just aggregate scores.

Fourth, evaluate **multiple real datasets and horizons**. ETTh1 provides a useful infrastructure starting point, but electricity-market demand, cloud infrastructure metrics, and retail operations have different sampling behavior and shift mechanisms. The point is to see where the same forecasting method succeeds, not to find the one dataset on which it looks best.

Fifth, add **probabilistic drift evaluation**. Monitor quantile coverage, tail misses, interval sharpness, and conditional calibration before and after structural breaks. A model can maintain reasonable MAE while becoming dangerously overconfident.

Sixth, introduce a **forecast-value metric**, not just an error metric. For a grid operator, the monetary cost of underforecasting reserve demand might differ dramatically from overforecasting. A forecast with slightly higher MAE can be more useful if it performs better under the actual decision loss. Optimize the operational objective only after establishing that the evaluation remains free from lookahead leakage.

## 29. Conclusion: the more serious claim is not that a model can look at more series

TimesFM-3 is an important step in the evolution of pretrained time-series forecasting. It extends a widely used forecasting family to correlated targets and external features, provides joint predictions through alternating attention, and can generate a complete forecast horizon without iterative autoregressive decoding. Google has made the checkpoint and inference code accessible for research, and it offers an enterprise path through BigQuery. [1, 2, 4]

But the part worth remembering is not simply the parameter count, the number of training time points, or a leaderboard position.

**The hard problem in forecasting is deciding what information is actually available at the moment of prediction, what dependencies can be trusted, and how a model behaves when those dependencies stop being true.**

Our small experiment makes that concrete. Joint temporal and variate attention helped before the designed break. A simple seasonal forecast was far more reliable afterward. That does not discredit TimesFM-3. It shows why sophisticated forecasting systems should be evaluated on temporal validity, comparative baselines, uncertainty, and changing operating conditions rather than just one attractive test-period average.

The next time a foundation model claims to forecast many things at once, ask four questions: What was visible at the forecast origin? What was the simplest competitive baseline? What happened under distribution shift? And were the model's predictions, timestamps, and evaluation code saved so anyone else could check the answer?

Those questions turn a model announcement into engineering knowledge.

---

## 30. References and further reading

**[1] Google Research, August 31, 2026.** Ayush Jain and Rajat Sen. *TimesFM-3: A zero-shot foundation model for multivariate forecasting.* Architecture, patches, temporal and variate attention, masking, and launch-time benchmark comparisons. https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/

**[2] Google Research.** *TimesFM official GitHub repository.* API examples for TimesFM 3.0, checkpoint naming, Torch/MLX backends, version history, and license distinctions. https://github.com/google-research/timesfm

**[3] Google Research.** *TimesFM 3.0 PyTorch checkpoint and model card.* Official model configuration and non-commercial license. https://huggingface.co/google/timesfm-3.0-pytorch

**[4] Google Cloud, October 12, 2026.** *Introducing TimesFM 3.0 with multivariate time-series forecasting in BigQuery.* Commercially accessible SQL integration, target and covariate roles, and examples. https://cloud.google.com/blog/products/data-analytics/bigquery-timesfm-3-0-with-multivariate-time-series-forecasting/

**[5] Aksu et al., 2024.** *GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation.* https://arxiv.org/abs/2410.10393

**[6] Shchur et al., 2025.** *fev-bench: A Realistic Benchmark for Time Series Forecasting.* https://arxiv.org/abs/2509.26468

**[7] Qiao et al., ICML 2026.** *It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks.* https://proceedings.mlr.press/v306/qiao26g.html

**[8] Zhou et al.** *ETDataset: Electricity Transformer Dataset*, original public repository and ETTh1 CSV. https://github.com/zhouhaoyi/ETDataset

**[9] Das et al., ICML 2024.** *A Decoder-only Foundation Model for Time-series Forecasting.* Original TimesFM design and motivations. https://arxiv.org/abs/2310.10688

**[10] The Forecasting Company, October 13, 2026.** *t0-gamma: the top time-series foundation model on fev-bench, GIFT-Eval and TIME.* A subsequent model-release benchmark claim, illustrating why rankings must be dated and checked against current benchmark methodology. https://www.theforecastingcompany.com/blog/t0-gamma/

**[11] Google Cloud documentation.** *Forecast multiple time series with a TimesFM multivariate model.* https://docs.cloud.google.com/bigquery/docs/timesfm-multivariate-multi-time-series-forecasting-tutorial

**Companion experiment:** `timesfm3_forecasting_lab` includes source code, test suite, saved five-seed synthetic benchmark JSON, independent architectural teaching model, ETTh1 adapter, optional real TimesFM-3 evaluator, figures, and `BENCHMARK_NOTES.md` for exact reproducibility assumptions.

*Disclosure: This is an independent technical tutorial. The miniature Transformer and synthetic benchmark are not developed or endorsed by Google Research. No actual TimesFM-3 inference has been presented as a completed local experiment. The article distinguishes the source authors' claims from the companion lab's measured results throughout.*
