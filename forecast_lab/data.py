"""Synthetic hourly electric-load benchmark with a controlled late regime shift.

All targets and weather are measured historically; calendar and scheduled tariff
are available for the whole forecasting horizon at the prediction origin.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Dataset:
    values: np.ndarray  # [time, channels] targets (3) + measured weather (1) + known time (1) + known tariff (1)
    train_end: int
    shift_start: int
    target_count: int = 3
    past_only_count: int = 1


def make_energy(seed: int = 0, n: int = 1488) -> Dataset:
    if n < 1200:
        raise ValueError('Use at least 1200 hourly observations for the prescribed holdouts')
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    hour = t % 24
    weekday = (t // 24) % 7
    daily = np.sin(2*np.pi*(hour-7)/24)
    weather = 20 + 8*np.sin(2*np.pi*(hour-4)/24) + 2*np.sin(2*np.pi*t/168) + rng.normal(0, 1, n)
    calendar = np.cos(2*np.pi*(hour-8)/24)
    # Known-ahead scheduled pricing event. No future measurements are used.
    tariff = (((t // 24) % 5 == 0) & (hour >= 17) & (hour <= 21)).astype(float)
    level = np.array([35., 50., 75.]); amplitude = np.array([12., 16., 21.])
    tmp = np.array([0.50, 0.70, 0.85]); event = np.array([-5., -7., -9.])
    target = np.empty((n, 3), dtype=np.float64)
    for i in range(3):
        target[:24, i] = level[i] + amplitude[i]*daily[:24] + rng.normal(0, 1, 24)
    shift_start = 1152
    for j in range(24, n):
        shifted = j >= shift_start
        for i in range(3):
            neighbor = (i + (2 if shifted else 1)) % 3
            temperature = tmp[i] * (1.7 if shifted else 1.) * abs(weather[j] - 20)
            holiday = -2.5 if weekday[j] in (5,6) else 0.
            baseline = level[i] + amplitude[i]*daily[j] + temperature + holiday
            # Shift includes different cross-series driver and baseline offset.
            if shifted: baseline += (5 if i == 0 else -3 if i == 1 else 9)
            target[j,i] = (0.52*target[j-24,i] + 0.10*target[j-1,neighbor]
                           + 0.38*baseline + event[i]*tariff[j]
                           + rng.normal(0, 1.5))
    values = np.column_stack([target, weather, calendar, tariff]).astype(np.float32)
    return Dataset(values, train_end=864, shift_start=shift_start)


def make_windows(ds: Dataset, context: int, horizon: int, start: int, stop: int, stride: int = 1):
    """Origins t have observed data [t-context, t), predict [t,t+horizon).
    The entire horizon is guaranteed to finish strictly before stop.
    """
    if not (context > 0 and horizon > 0 and stride > 0):
        raise ValueError('context/horizon/stride must be positive')
    origins = np.arange(max(context, start), min(stop - horizon, len(ds.values) - horizon) + 1, stride)
    xs = ds.values[np.array([np.arange(t-context, t+horizon) for t in origins])]
    ys = np.array([ds.values[t:t+horizon,:ds.target_count].T for t in origins])
    return xs, ys, origins


class TrainScaler:
    """Global per-channel mean/std fitted only on timestamps before train_end."""
    def fit(self, ds: Dataset):
        self.mu = ds.values[:ds.train_end].mean(0).astype(np.float32)
        self.sd = ds.values[:ds.train_end].std(0).clip(0.05).astype(np.float32)
        return self
    def transform(self, x): return (x-self.mu)/self.sd
    def inverse_targets(self, y):
        return y * self.sd[:3][None,:,None] + self.mu[:3][None,:,None]


def input_visibility(x, context: int, target_count: int=3, past_only_count: int=1):
    """Set future target and past-only covariates to 0 with observed-mask False.
    Future known covariates are kept. x shape [B, L, C]."""
    x = np.asarray(x).copy()
    mask = np.ones_like(x, dtype=np.float32)
    x[:, context:, :target_count+past_only_count] = 0
    mask[:, context:, :target_count+past_only_count] = 0
    return x, mask


def origins_for(ds:Dataset, horizon:int):
    return {
        'in_domain': (ds.train_end, ds.shift_start),
        'shift': (ds.shift_start, len(ds.values)),
    }


def load_etth1(path):
    """Load original ETTh1 hourly CSV. Targets HUFL, MUFL, OT; historical HULL;
    known calendar sin/cos from provided timestamps. All source values retain
    their original units. This is transformer operating data, not grid demand.
    """
    import csv
    from datetime import datetime, timedelta
    rows=[]; times=[]
    with open(path,newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        required={'date','HUFL','MUFL','OT','HULL'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'ETTh1 CSV missing columns {required}')
        for row in reader:
            dt=datetime.strptime(row['date'],'%Y-%m-%d %H:%M:%S')
            hour=dt.hour
            rows.append([float(row['HUFL']),float(row['MUFL']),float(row['OT']),
                         float(row['HULL']),np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24)])
            times.append(dt)
    if len(rows)<500:raise ValueError('Too few hourly observations')
    if any(b-a!=timedelta(hours=1) for a,b in zip(times,times[1:])):
        raise ValueError('ETTh1 timestamps are not contiguous hourly observations')
    n=len(rows)
    return Dataset(np.asarray(rows,dtype=np.float32),int(n*.6),int(n*.8))
