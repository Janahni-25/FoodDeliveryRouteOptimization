"""
TABPC+  -  Time-Aware Batching and Precedence-Constrained routing for last-mile food delivery.
Run with:   streamlit run app.py
"""
from __future__ import annotations

from dataclasses import replace

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from analysis import (ETA_FEATURES, KEY_METRICS, METRIC_LABELS, ablation, compare, eta_dataset,
                      run_once, sensitivity, train_eta)
from model import (AlgoConfig, Euclid, PRESET_NAMES, WorldConfig, fmt_time, preset)
from planners import HAS_ORTOOLS

st.set_page_config(page_title="TABPC+ Food Delivery Routing", layout="wide")

st.markdown("<style>.stApp, [data-testid=stSidebar] > div {background:#051d0d}</style>", unsafe_allow_html=True)

PALETTE = ["#92f2ff", "#f85d5d", "#082a14", "#c081fb", "#ea580c", "#e9ff91", "#ffdd93", "#f7a3c9"]
PRIO_COLOR = {"express": "#f85d5d", "normal": "#92f2ff", "scheduled": "#6A7E7F"}

# paper dataset sizes: (orders, riders, batch size, rider limit, span hours)
PAPER = {
    "Paper set 1: 20 orders": (20, 3, 5, 2, 4),
    "Paper set 2: 40 orders": (40, 4, 10, 3, 4),
    "Paper set 3: 50 orders": (50, 5, 10, 3, 4),
    "Paper set 4: 100 orders": (100, 6, 20, 5, 6),
}
DEFAULTS = dict(preset_sel="Paper set 2: 40 orders", n_orders=40, n_riders=4, batch_size=10,
                rider_limit=3, span_h=4, window=10, start_h=11, seed=0)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset():
    p = PAPER.get(st.session_state.preset_sel)
    if p:
        (st.session_state.n_orders, st.session_state.n_riders, st.session_state.batch_size,
         st.session_state.rider_limit, st.session_state.span_h) = p


# =============================================================================== sidebar
with st.sidebar:
    st.title("TABPC+")
    st.header("World")
    st.selectbox("Dataset preset", ["Custom"] + list(PAPER), key="preset_sel", on_change=apply_preset)
    st.slider("Orders", 10, 300, key="n_orders", step=5)
    st.slider("Riders", 1, 15, key="n_riders")
    st.slider("Max batch size", 2, 40, key="batch_size")
    st.slider("Base orders per rider per batch", 1, 8, key="rider_limit")
    st.slider("Order arrival span (hours)", 1, 8, key="span_h")
    st.slider("Start hour", 6, 20, key="start_h")
    st.slider("Batching window (min)", 3, 30, key="window")
    c1, c2 = st.columns(2)
    traffic = c1.checkbox("Traffic", True)
    hetero = c2.checkbox("Mixed fleet", True)
    cancel_prob = st.slider("Cancellation probability", 0.0, 0.5, 0.0, 0.05)
    breakdown = st.checkbox("Rider breakdown mid-run", False)
    dist_model = st.radio("Distances", ["Synthetic grid", "Real roads (OSMnx)"], horizontal=True)
    road_args = None
    if dist_model.startswith("Real"):
        lat = st.number_input("Centre latitude", value=11.0168, format="%.4f")
        lon = st.number_input("Centre longitude", value=76.9558, format="%.4f")
        rad = st.number_input("Radius (m)", value=4000, step=500)
        road_args = (float(lat), float(lon), int(rad))

    st.header("TABPC+ objective")
    alpha = st.number_input("alpha: distance (per km)", 0.0, 20.0, 1.0, 0.1)
    beta = st.number_input("beta: lateness (per min)", 0.0, 20.0, 2.0, 0.5)
    gamma = st.number_input("gamma: food age (per min)", 0.0, 5.0, 0.1, 0.05)
    delta = st.number_input("delta: workload spread (per km)", 0.0, 5.0, 0.5, 0.1)
    fresh = st.slider("Freshness limit (min pickup to drop)", 10, 60, 30)
    sa_iters = st.slider("Simulated annealing iterations / batch", 50, 3000, 400, 50)

    st.header("TABPC+ features")
    f_prep = st.checkbox("Prep-time awareness", True)
    f_prio = st.checkbox("Order priority levels", True)
    f_traf = st.checkbox("Traffic-aware planning", True)
    f_het = st.checkbox("Fleet-aware speeds", True)
    f_fair = st.checkbox("Rider fairness", True)
    f_fresh = st.checkbox("Freshness limit", True)
    f_sa = st.checkbox("Simulated annealing", True)
    f_inter = st.checkbox("Inter-route moves", True)
    f_pair = st.checkbox("Pair-aware moves", True)
    f_adapt = st.checkbox("Adaptive window", True)
    f_repo = st.checkbox("Idle rider repositioning", True)
    f_disr = st.checkbox("Disruption handling", True)


@st.cache_resource(show_spinner="Downloading road network...")
def load_roads(lat, lon, rad):
    from model import RoadNetwork
    return RoadNetwork((lat, lon), rad)


def get_dm(W):
    if road_args is None:
        return Euclid(W.km_per_unit, W.grid)
    return load_roads(*road_args)


W = WorldConfig(
    n_orders=st.session_state.n_orders, n_riders=st.session_state.n_riders,
    batch_size=st.session_state.batch_size, rider_limit=st.session_state.rider_limit,
    sim_start=st.session_state.start_h * 60.0, sim_span=st.session_state.span_h * 60.0,
    window=float(st.session_state.window), window_min=max(2.0, st.session_state.window * 0.4),
    window_max=st.session_state.window * 2.0, traffic=traffic, hetero=hetero,
    cancel_prob=cancel_prob, breakdown=breakdown, fresh_limit=float(fresh),
)
A_PLUS = AlgoConfig(
    name="TABPC+", alpha=alpha, beta=beta, gamma=gamma, delta=delta, sa_iters=sa_iters,
    use_prep=f_prep, use_priority=f_prio, use_traffic=f_traf, use_hetero=f_het, use_fair=f_fair,
    use_fresh=f_fresh, use_sa=f_sa, use_inter=f_inter, use_pair=f_pair, adaptive=f_adapt,
    reposition=f_repo, handle_disruption=f_disr,
)


def algo_by_name(name: str) -> AlgoConfig:
    return A_PLUS if name == "TABPC+" else preset(name)


def available_algos():
    return [n for n in PRESET_NAMES if HAS_ORTOOLS or not n.startswith("GLS")]


def make_progress():
    bar = st.progress(0.0, text="Starting...")
    return lambda f, msg: bar.progress(min(1.0, f), text=msg)


# =============================================================================== map
def build_map(sim, batch_sel, dm):
    ll = dm.to_latlon
    pts = [ll(o.rest) for o in sim.orders] + [ll(o.cust) for o in sim.orders]
    center = (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
    m.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)],
                  [max(p[0] for p in pts), max(p[1] for p in pts)]])
    seen = set()
    for o in sim.orders:
        if o.rest_id not in seen:
            seen.add(o.rest_id)
            folium.Marker(ll(o.rest), tooltip=f"Restaurant {o.rest_id} (prep ~{o.prep:.0f} min)",
                          icon=folium.Icon(color="green", icon="cutlery")).add_to(m)
    plans = [p for p in sim.all_plans() if batch_sel == "All" or p["batch"] == batch_sel]
    for p in plans:
        col = PALETTE[p["rider"] % len(PALETTE)]
        path = [ll(p["start_pos"])] + [ll(s["pos"]) for s in p["stops"]]
        folium.PolyLine(path, color=col, weight=3, opacity=0.8,
                        tooltip=f"Rider {p['rider']} ({p['vehicle']}) batch {p['batch']}: "
                                f"{p['dist']:.1f} km").add_to(m)
        folium.CircleMarker(path[0], radius=6, color=col, fill=True, fill_opacity=1,
                            tooltip=f"Rider {p['rider']} start {fmt_time(p['start'])}").add_to(m)
        for k, s in enumerate(p["stops"], 1):
            o = sim.omap[s["oid"]]
            if s["kind"] == 0:
                txt = f"#{k} PICKUP order {o.id} at {fmt_time(s['arrive'])} (ready {fmt_time(o.ready)})"
                folium.RegularPolygonMarker(ll(s["pos"]), number_of_sides=4, radius=6, color=col,
                                            fill_color=col, fill_opacity=0.4, tooltip=txt).add_to(m)
            else:
                late = max(0.0, s["arrive"] - o.deadline)
                txt = (f"#{k} DROP order {o.id} ({o.priority}) at {fmt_time(s['arrive'])}, "
                       f"deadline {fmt_time(o.deadline)}" + (f", LATE {late:.0f} min" if late else ""))
                folium.CircleMarker(ll(s["pos"]), radius=6, color=PRIO_COLOR[o.priority],
                                    fill=True, fill_color="#f85d5d" if late else col,
                                    fill_opacity=0.9, tooltip=txt).add_to(m)
    return m


def show_html(html: str, height: int):
    """Embed the folium map (works on new and older Streamlit versions)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:  # older Streamlit
        from streamlit.components.v1 import html as legacy_html
        legacy_html(html, height=height)


def bar_panel(mean, std, metrics, ncols=2, title=None):
    n = len(metrics)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, mt in zip(axes, metrics):
        vals = mean[mt].to_numpy()
        ax.bar(range(len(vals)), vals, yerr=std[mt].to_numpy() if std is not None else None,
               color=[PALETTE[i % len(PALETTE)] for i in range(len(vals))], capsize=3)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([str(i) for i in mean.index], rotation=35, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS.get(mt, mt), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def pretty(df):
    return df.rename(columns=METRIC_LABELS)


# =============================================================================== tabs
st.title("Time-Aware Batching and Precedence-Constrained Routing (TABPC+)")
tabs = st.tabs(["Run & map", "Compare algorithms", "Sensitivity", "Ablation", "ETA predictor"])

# ------------------------------------------------------------------ Run & map
with tabs[0]:
    c1, c2, c3 = st.columns([2, 1, 1])
    algo_name = c1.selectbox("Algorithm", available_algos())
    seed = c2.number_input("Seed", 0, 9999, key="seed")
    run_clicked = c3.button("Run simulation", type="primary", use_container_width=True)
    if run_clicked:
        try:
            with st.spinner("Simulating..."):
                dm = get_dm(W)
                st.session_state.sim = run_once(W, algo_by_name(algo_name), int(seed), dm)
                st.session_state.sim_dm = dm
        except Exception as e:  # noqa: BLE001
            st.error(f"Run failed: {e}")
    sim = st.session_state.get("sim")
    if sim is None:
        st.write("Choose settings in the sidebar and press **Run simulation**.")
    else:
        m = sim.metrics
        st.subheader(f"{sim.A.name}  -  {len(sim.orders)} orders, {len(sim.riders)} riders")
        r1 = st.columns(6)
        r1[0].metric("Distance", f"{m['distance_km']:.1f} km")
        r1[1].metric("Late deliveries", f"{m['late_orders']:.0f}", f"{m['on_time_pct']:.0f}% on time",
                     delta_color="off")
        r1[2].metric("Avg delivery", f"{m['avg_delivery_min']:.1f} min")
        r1[3].metric("Rider idle", f"{m['idle_min']:.0f} min")
        r1[4].metric("CO2", f"{m['co2_kg']:.1f} kg")
        r1[5].metric("Runtime", f"{m['runtime_s']:.2f} s")
        r2 = st.columns(6)
        r2[0].metric("Workload std", f"{m['rider_km_std']:.1f} km")
        r2[1].metric("Fuel cost", f"INR {m['fuel_cost_inr']:.0f}")
        r2[2].metric("First pickup travel", f"{m['avg_first_pickup_min']:.1f} min")
        r2[3].metric("Reposition km", f"{m['reposition_km']:.1f}")
        r2[4].metric("Cancelled / reassigned", f"{m['cancelled']:.0f} / {m['reassigned']:.0f}")
        r2[5].metric("Orders per trip", f"{m['avg_orders_per_trip']:.2f}")
        batches = sorted({p["batch"] for p in sim.all_plans()})
        sel = st.selectbox("Show batch", ["All"] + batches)
        show_html(build_map(sim, sel, st.session_state.sim_dm).get_root().render(), 620)
        with st.expander("Rider fleet and routes"):
            st.dataframe(pd.DataFrame([{"rider": r.id, "vehicle": r.vehicle, "speed km/h": r.speed,
                                        "capacity": r.cap, "trips": len(r.plans),
                                        "km": round(r.km(), 1)} for r in sim.riders]),
                         hide_index=True)
            st.dataframe(pd.DataFrame([{
                "batch": p["batch"], "rider": p["rider"], "vehicle": p["vehicle"],
                "start": fmt_time(p["start"]), "orders": p["n_orders"], "km": round(p["dist"], 2),
                "idle min": round(p["idle"], 1),
                "route": " > ".join(("P" if s["kind"] == 0 else "D") + str(s["oid"]) for s in p["stops"]),
            } for p in sim.all_plans()]), hide_index=True)
        with st.expander("Order log"):
            rows = []
            for o in sim.orders:
                rec = sim.records.get(o.id)
                rows.append({"order": o.id, "placed": fmt_time(o.t_order), "priority": o.priority,
                             "prep min": round(o.prep, 1), "deadline": fmt_time(o.deadline),
                             "rider": rec["rider"] if rec else None,
                             "delivered": fmt_time(rec["delivered"]) if rec else "-",
                             "late min": round(rec["late"], 1) if rec else None,
                             "status": "delivered" if rec else
                             ("cancelled" if o in sim.cancelled else "undelivered")})
            st.dataframe(pd.DataFrame(rows), hide_index=True)

# ------------------------------------------------------------------ Compare
with tabs[1]:
    cc1, cc2 = st.columns([3, 1])
    chosen = cc1.multiselect("Algorithms", available_algos(), default=[a for a in available_algos()
                                                                     if not a.startswith("GLS")])
    nseeds = cc2.number_input("Seeds", 1, 20, 3, key="cmp_seeds")
    if st.button("Run comparison", type="primary"):
        try:
            prog = make_progress()
            st.session_state.cmp = compare(W, [algo_by_name(n) for n in chosen], range(int(nseeds)),
                                           get_dm(W), prog)
        except Exception as e:  # noqa: BLE001
            st.error(f"Comparison failed: {e}")
    if "cmp" in st.session_state:
        raw, mean, std = st.session_state.cmp
        st.dataframe(pretty(mean[KEY_METRICS]).round(2))
        st.pyplot(bar_panel(mean, std, ["distance_km", "late_orders", "avg_delivery_min", "idle_min",
                                        "rider_km_std", "composite"], title="Mean over seeds (error bar = std)"))
        st.subheader("Cost and CO2")
        if "No batching" in mean.index:
            base = mean.loc["No batching"]
            sav = pd.DataFrame({
                "CO2 (kg)": mean["co2_kg"], "Fuel (INR)": mean["fuel_cost_inr"],
                "CO2 saved vs no batching (%)": 100 * (1 - mean["co2_kg"] / base["co2_kg"]),
                "Fuel saved vs no batching (%)": 100 * (1 - mean["fuel_cost_inr"] / base["fuel_cost_inr"]),
            })
            st.dataframe(sav.round(1))
        else:
            st.dataframe(mean[["co2_kg", "fuel_cost_inr"]].round(1))
        with st.expander("All metrics"):
            st.dataframe(pretty(mean).round(2))
        with st.expander("Raw per-seed results"):
            st.dataframe(raw.round(2), hide_index=True)

# ------------------------------------------------------------------ Sensitivity
with tabs[2]:
    PARAMS = {
        "batch_size": "5,10,20,30", "n_riders": "3,4,5,6,8", "window": "5,10,15,20",
        "rider_limit": "1,2,3,4,5", "beta": "0,1,2,5,10", "gamma": "0,0.1,0.5,1",
        "alpha": "0.5,1,2,4", "delta": "0,0.5,1,2", "sa_iters": "0,100,400,1000",
    }
    s1, s2, s3 = st.columns([2, 3, 1])
    param = s1.selectbox("Parameter", list(PARAMS))
    values_txt = s2.text_input("Values (comma separated)", PARAMS[param], key=f"vals_{param}")
    sseeds = s3.number_input("Seeds", 1, 10, 3, key="sens_seeds")
    if st.button("Run sensitivity", type="primary"):
        try:
            vals = [float(v) for v in values_txt.split(",") if v.strip()]
            st.session_state.sens = (param, sensitivity(W, A_PLUS, param, vals, range(int(sseeds)),
                                                         get_dm(W), make_progress()))
        except Exception as e:  # noqa: BLE001
            st.error(f"Sensitivity failed: {e}")
    if "sens" in st.session_state:
        p, df = st.session_state.sens
        fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
        for ax, mt, col in zip(axes, ["distance_km", "late_orders", "avg_delivery_min"], PALETTE):
            ax.plot(df[p], df[mt], marker="o", color=col)
            ax.set_xlabel(p)
            ax.set_title(METRIC_LABELS[mt], fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        st.dataframe(pretty(df[[p] + KEY_METRICS]).round(2), hide_index=True)

# ------------------------------------------------------------------ Ablation
with tabs[3]:
    aseeds = st.number_input("Seeds", 1, 10, 2, key="abl_seeds")
    if st.button("Run ablation", type="primary"):
        try:
            st.session_state.abl = ablation(W, A_PLUS, range(int(aseeds)), get_dm(W), make_progress())
        except Exception as e:  # noqa: BLE001
            st.error(f"Ablation failed: {e}")
    if "abl" in st.session_state:
        _, mean, std = st.session_state.abl
        cols = ["distance_km", "late_orders", "avg_delivery_min", "rider_km_std",
                "avg_first_pickup_min", "idle_min", "composite"]
        st.dataframe(pretty(mean[cols]).round(2))
        full = mean.loc["TABPC+ (full)", "composite"]
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(mean) + 1))
        comp = mean["composite"]
        ax.barh(range(len(comp)), comp.to_numpy(),
                color=["#082a14" if n == "TABPC+ (full)" else ("#6A7E7F" if n.startswith("Original") else "#92f2ff")
                       for n in comp.index])
        ax.axvline(full, color="#082a14", ls="--", lw=1)
        ax.set_yticks(range(len(comp)))
        ax.set_yticklabels(comp.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Composite score (lower is better)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

# ------------------------------------------------------------------ ETA
with tabs[4]:
    e1, e2 = st.columns([1, 3])
    eseeds = e1.number_input("Training simulations", 2, 30, 8, key="eta_seeds")
    if e2.button("Generate data and train", type="primary"):
        try:
            with st.spinner("Simulating and training..."):
                df = eta_dataset(W, A_PLUS, range(int(eseeds)), get_dm(W))
                st.session_state.eta = (df,) + train_eta(df)
        except Exception as e:  # noqa: BLE001
            st.error(f"Training failed: {e}")
    if "eta" in st.session_state:
        df, model, met, hold = st.session_state.eta
        mc = st.columns(4)
        mc[0].metric("MAE", f"{met['MAE (min)']:.1f} min")
        mc[1].metric("RMSE", f"{met['RMSE (min)']:.1f} min")
        mc[2].metric("R2", f"{met['R2']:.2f}")
        mc[3].metric("Model", met["model"].replace("_", " "))
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(hold["delivery_min"], hold["predicted"], s=10, alpha=0.6, color="#92f2ff")
        lim = [0, float(max(hold["delivery_min"].max(), hold["predicted"].max())) * 1.05]
        ax.plot(lim, lim, color="#f85d5d", lw=1)
        ax.set_xlabel("Actual delivery time (min)")
        ax.set_ylabel("Predicted (min)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        st.subheader("Predict for a new order")
        p1, p2, p3 = st.columns(3)
        rc = p1.slider("Restaurant to customer (km)", 0.2, 8.0, 2.0, 0.1)
        prep = p1.slider("Prep time (min)", 3, 25, 10)
        hour = p2.slider("Hour of order", 6.0, 23.0, 13.0, 0.25)
        bsz = p2.slider("Batch size", 1, 30, 8)
        ro = p3.slider("Orders on the route", 1, 6, 3)
        rank = p3.slider("Position of this drop on the route", 0, 5, 1)
        veh = st.selectbox("Vehicle", ["cycle (12 km/h)", "scooter (22 km/h)", "bike (30 km/h)"], index=1)
        prio = st.selectbox("Priority", ["express", "normal", "scheduled"], index=1)
        speed = {"cycle": 12.0, "scooter": 22.0, "bike": 30.0}[veh.split()[0]]
        peak = 1.0 if (12 <= hour < 14 or 19 <= hour < 21) else 0.0
        x = np.array([[rc, prep, hour, bsz, ro, rank, speed,
                       {"express": 0, "normal": 1, "scheduled": 2}[prio], peak]], float)
        st.metric("Predicted delivery time", f"{float(model.predict(x)[0]):.0f} min")
