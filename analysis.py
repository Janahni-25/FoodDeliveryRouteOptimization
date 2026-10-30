"""
analysis.py - experiments: algorithm comparison, sensitivity analysis, ablation study and
customer ETA prediction.
"""
from __future__ import annotations

from dataclasses import fields, replace
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from model import AlgoConfig, WorldConfig, preset
from simulator import Simulator

METRIC_LABELS = {
    "distance_km": "Total distance (km)",
    "reposition_km": "Repositioning distance (km)",
    "late_orders": "Late deliveries",
    "on_time_pct": "On-time (%)",
    "total_lateness_min": "Total lateness (min)",
    "avg_delivery_min": "Avg delivery time (min)",
    "max_delivery_min": "Max delivery time (min)",
    "avg_food_age_min": "Avg food age (min)",
    "fresh_violations": "Freshness violations",
    "idle_min": "Rider idle time (min)",
    "rider_km_std": "Workload std (km)",
    "rider_km_gap": "Workload gap max-min (km)",
    "riders_used": "Riders used",
    "batches": "Batches",
    "avg_orders_per_trip": "Orders per trip",
    "avg_first_pickup_min": "First-pickup travel (min)",
    "co2_kg": "CO2 (kg)",
    "fuel_cost_inr": "Fuel cost (INR)",
    "delivered": "Delivered",
    "cancelled": "Cancelled",
    "reassigned": "Re-assigned",
    "undelivered": "Undelivered",
    "runtime_s": "Runtime (s)",
    "composite": "Composite score",
    "late_express": "Late: express",
    "late_normal": "Late: normal",
    "late_scheduled": "Late: scheduled",
}

KEY_METRICS = ["distance_km", "late_orders", "on_time_pct", "avg_delivery_min", "idle_min",
               "rider_km_std", "co2_kg", "fuel_cost_inr", "runtime_s", "composite"]


def run_once(W: WorldConfig, A: AlgoConfig, seed: int = None, dm=None) -> Simulator:
    W2 = replace(W, seed=seed) if seed is not None else W
    return Simulator(W2, A, dm).run()


def _rows(W, algos: Sequence[AlgoConfig], seeds: Sequence[int], dm=None, progress=None):
    rows = []
    total = len(algos) * len(seeds)
    done = 0
    for A in algos:
        for s in seeds:
            sim = run_once(W, A, s, dm)
            row = {"algorithm": A.name, "seed": s}
            row.update(sim.metrics)
            rows.append(row)
            done += 1
            if progress:
                progress(done / total, f"{A.name} / seed {s}")
    return pd.DataFrame(rows)


def compare(W: WorldConfig, algos: Sequence[AlgoConfig], seeds: Sequence[int], dm=None, progress=None):
    """Feature 13: baselines. Returns (raw, mean, std) DataFrames indexed by algorithm."""
    raw = _rows(W, algos, seeds, dm, progress)
    order = [a.name for a in algos]
    g = raw.drop(columns=["seed"]).groupby("algorithm")
    return raw, g.mean().reindex(order), g.std(ddof=0).reindex(order)


def sensitivity(W: WorldConfig, A: AlgoConfig, param: str, values: Sequence[float],
                seeds: Sequence[int], dm=None, progress=None) -> pd.DataFrame:
    """Feature 14: vary one parameter (a WorldConfig or AlgoConfig field) and average over seeds."""
    wf = {f.name for f in fields(WorldConfig)}
    af = {f.name for f in fields(AlgoConfig)}
    rows = []
    total, done = len(values) * len(seeds), 0
    for v in values:
        if param in wf:
            cur = getattr(W, param)
            vv = type(cur)(v) if not isinstance(cur, bool) else bool(v)
            W2, A2 = replace(W, **{param: vv}), A
        elif param in af:
            cur = getattr(A, param)
            vv = type(cur)(v)
            W2, A2 = W, replace(A, **{param: vv})
        else:
            raise KeyError(param)
        for s in seeds:
            sim = run_once(W2, A2, s, dm)
            row = {param: v, "seed": s}
            row.update(sim.metrics)
            rows.append(row)
            done += 1
            if progress:
                progress(done / total, f"{param}={v} / seed {s}")
    df = pd.DataFrame(rows)
    return df.drop(columns=["seed"]).groupby(param).mean().reset_index()


def ablation_variants(A_full: AlgoConfig, W: WorldConfig) -> List[AlgoConfig]:
    """Feature 15: the full method with one feature switched off at a time."""
    v = [replace(A_full, name="TABPC+ (full)")]

    def off(name, **kw):
        v.append(replace(A_full, name=name, **kw))

    off("- prep-time awareness", use_prep=False)
    off("- priority levels", use_priority=False)
    off("- traffic awareness", use_traffic=False)
    off("- fleet-aware speeds", use_hetero=False)
    off("- fairness term", use_fair=False)
    off("- freshness limit", use_fresh=False)
    off("- simulated annealing", use_sa=False)
    off("- inter-route moves", use_inter=False)
    off("- pair-aware moves", use_pair=False)
    off("- adaptive window", adaptive=False)
    off("- idle repositioning", reposition=False)
    if W.cancel_prob > 0 or W.breakdown:
        off("- disruption handling", handle_disruption=False)
    v.append(preset("Original TABPC"))
    return v


def ablation(W: WorldConfig, A_full: AlgoConfig, seeds: Sequence[int], dm=None, progress=None):
    return compare(W, ablation_variants(A_full, W), seeds, dm, progress)


# --------------------------------------------------------------------------- ETA prediction
ETA_FEATURES = ["rc_km", "prep", "hour", "batch_size", "route_orders", "stop_rank", "speed",
                "priority_rank", "peak"]


def eta_dataset(W: WorldConfig, A: AlgoConfig, seeds: Sequence[int], dm=None) -> pd.DataFrame:
    """Delivery records from simulated runs: features known around dispatch -> delivery time."""
    from model import PRIORITY_RANK
    rows = []
    for s in seeds:
        sim = run_once(W, A, s, dm)
        for r in sim.records.values():
            h = (r["t_order"] % 1440) / 60.0
            rows.append({
                "rc_km": r["rc_km"], "prep": r["prep"], "hour": h, "batch_size": r["batch_size"],
                "route_orders": r["route_orders"], "stop_rank": r["stop_rank"], "speed": r["speed"],
                "priority_rank": PRIORITY_RANK[r["priority"]],
                "peak": 1.0 if (12 <= h < 14 or 19 <= h < 21) else 0.0,
                "delivery_min": r["delivered"] - r["t_order"],
            })
    return pd.DataFrame(rows)


class EtaModel:
    """Random Forest if scikit-learn is available, otherwise least squares."""

    def __init__(self):
        self.kind = "random_forest"
        self.model = None
        self.coef = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=120, random_state=0, min_samples_leaf=2)
            self.model.fit(X, y)
        except Exception:
            self.kind = "linear_regression"
            Xb = np.c_[X, np.ones(len(X))]
            self.coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return self.model.predict(X)
        return np.c_[X, np.ones(len(X))] @ self.coef


def train_eta(df: pd.DataFrame, seed: int = 0):
    """Returns (model, metrics dict, holdout DataFrame with actual & predicted)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = max(1, int(0.75 * len(df)))
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    model = EtaModel().fit(tr[ETA_FEATURES].to_numpy(float), tr["delivery_min"].to_numpy(float))
    pred = model.predict(te[ETA_FEATURES].to_numpy(float))
    err = pred - te["delivery_min"].to_numpy(float)
    ss_res = float((err ** 2).sum())
    ss_tot = float(((te["delivery_min"] - te["delivery_min"].mean()) ** 2).sum())
    metrics = {"MAE (min)": float(np.abs(err).mean()),
               "RMSE (min)": float(np.sqrt((err ** 2).mean())),
               "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
               "train rows": len(tr), "test rows": len(te), "model": model.kind}
    hold = te.copy()
    hold["predicted"] = pred
    return model, metrics, hold


if __name__ == "__main__":  # quick console comparison, no GUI needed
    W = WorldConfig(n_orders=40, n_riders=4, batch_size=10, rider_limit=3)
    names = ["TABPC+", "Original TABPC", "Nearest Neighbour", "No batching"]
    _, mean, _ = compare(W, [preset(n) for n in names], seeds=[0, 1, 2])
    pd.set_option("display.width", 200)
    print(mean[KEY_METRICS].round(2))
