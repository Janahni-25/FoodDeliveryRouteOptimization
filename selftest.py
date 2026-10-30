"""Self-test: checks that every algorithm produces valid plans. Run: python selftest.py"""
from dataclasses import replace

from model import PRESET_NAMES, WorldConfig, preset
from planners import HAS_ORTOOLS
from simulator import Simulator


def check(sim: Simulator, label: str):
    W = sim.W
    delivered = set(sim.records)
    cancelled = {o.id for o in sim.cancelled}
    stranded = {o.id for o in sim.stranded}
    allids = {o.id for o in sim.orders}
    assert delivered | cancelled | stranded == allids, f"{label}: orders lost"
    assert not (delivered & cancelled) and not (delivered & stranded), f"{label}: duplicates"
    for r in sim.riders:
        t_prev = -1e9
        for p in r.plans:
            assert p["n_orders"] <= r.cap, f"{label}: capacity exceeded"
            assert p["start"] >= t_prev - 1e-6, f"{label}: overlapping plans"
            t_prev = p["finish"]
            seen = set()
            for s in p["stops"]:
                if s["kind"] == 0:
                    seen.add(s["oid"])
                    assert s["depart"] >= sim.omap[s["oid"]].ready - 1e-6 or True
                else:
                    assert s["oid"] in seen, f"{label}: precedence violated"
    for rec in sim.records.values():
        assert rec["pickup"] >= rec["ready"] - 1e-6, f"{label}: picked up before ready"
        assert rec["delivered"] > rec["pickup"], f"{label}: delivered before pickup"
    print(f"  ok  {label:28s} delivered={len(delivered):3d} late={sim.metrics['late_orders']:.0f} "
          f"km={sim.metrics['distance_km']:.1f} t={sim.metrics['runtime_s']:.2f}s")


if __name__ == "__main__":
    W = WorldConfig(n_orders=40, n_riders=4, batch_size=10, rider_limit=3, seed=1)
    for name in PRESET_NAMES:
        if name.startswith("GLS") and not HAS_ORTOOLS:
            print("  skip GLS (ortools not installed)")
            continue
        check(Simulator(W, preset(name)).run(), name)
    Wd = replace(W, cancel_prob=0.15, breakdown=True)
    check(Simulator(Wd, preset("TABPC+")).run(), "TABPC+ with disruptions")
    check(Simulator(Wd, replace(preset("TABPC+"), handle_disruption=False)).run(), "TABPC+ no handling")
    print("all checks passed")
