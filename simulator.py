"""
simulator.py - dynamic discrete-event simulation of last-mile food delivery.

Orders arrive over time (lunch / dinner peaks). A sliding batching window (fixed or adaptive)
collects them, a planner routes the batch, riders then drive the planned routes in the TRUE world
(prep time, traffic, vehicle speeds). Optional: cancellations, a rider breakdown with
re-assignment, and idle-rider repositioning towards demand hotspots.
"""
from __future__ import annotations

import bisect
import math
import random
import time
from collections import Counter, deque
from typing import List

from model import (AlgoConfig, DEFAULT_SPEED, Euclid, PRIORITY_RANK, PRIORITY_WEIGHT, Rider,
                   VEHICLES, WorldConfig, eval_route, generate_world)
from planners import PLANNERS, Ctx, Slot


class Simulator:
    def __init__(self, W: WorldConfig, A: AlgoConfig, dm=None):
        self.W, self.A = W, A
        self.dm = dm if dm is not None else Euclid(W.km_per_unit, W.grid)
        self.orders, self.riders, self.breakdown = generate_world(W, self.dm)
        self.omap = {o.id: o for o in self.orders}
        self.arr_times = [o.t_order for o in self.orders]
        self.rng = random.Random(W.seed * 1000 + 17)
        self.records = {}
        self.dispatched = set()
        self.arrived: List = []
        self.cancelled: List = []
        self.stranded: List = []
        self.reassigned = 0
        self.batches = 0
        self.batch_sizes: List[int] = []
        self.reposition_km = 0.0
        self.metrics = {}

    # ------------------------------------------------------------------ helpers
    def window_now(self, t: float) -> float:
        W, A = self.W, self.A
        if A.mode == "nobatch":
            return 0.0
        if not A.adaptive:
            return W.window
        look = 15.0
        lo = bisect.bisect_right(self.arr_times, t - look)
        hi = bisect.bisect_right(self.arr_times, t)
        rate = (hi - lo) / look
        ref = W.n_orders / max(1.0, W.sim_span)
        w = W.window * ref / max(rate, 1e-6)
        return min(W.window_max, max(W.window_min, w))

    def reposition(self, t_d: float):
        """Feature: idle riders drift towards the busiest restaurants of the last 15 arrivals."""
        if not self.A.reposition or not self.arrived:
            return
        recent = self.arrived[-15:]
        spots = {o.rest_id: o.rest for o in recent}
        top = [spots[rid] for rid, _ in Counter(o.rest_id for o in recent).most_common(3)]
        for r in self.riders:
            gap = t_d - r.available_at
            if gap <= 1.0:
                continue
            target = min(top, key=lambda p: self.dm.dist(r.pos, p))
            newp = self.dm.move_toward(r.pos, target, r.speed * gap / 60.0)
            moved = self.dm.dist(r.pos, newp)
            if moved > 1e-9:
                r.pos = newp
                r.repo_km += moved
                self.reposition_km += moved
            r.available_at = t_d

    def commit(self, ridx: int, stops, t_d: float, bno: int, bsize: int):
        r = self.riders[ridx]
        start = max(t_d, r.available_at)
        ev = eval_route(stops, r.pos, start, r.speed, self.omap, self.dm, self.W,
                        prep=True, traffic=True, detail=True)
        plan = {"rider": ridx, "vehicle": r.vehicle, "batch": bno, "dispatch": t_d, "start": start,
                "start_pos": r.pos, "stops": ev.timeline, "dist": ev.dist, "idle": ev.idle,
                "finish": ev.finish, "last_pos": ev.last_pos, "n_orders": len(stops) // 2,
                "first_pickup": ev.timeline[0]["arrive"] - start}
        r.plans.append(plan)
        r.pos, r.available_at = ev.last_pos, ev.finish
        for d in ev.delivs:
            o = self.omap[d["oid"]]
            self.dispatched.add(o.id)
            rec = dict(d)
            rec.update({"rider": ridx, "vehicle": r.vehicle, "speed": r.speed, "batch": bno,
                        "batch_size": bsize, "route_orders": len(stops) // 2, "t_order": o.t_order,
                        "prep": o.prep, "priority": o.priority, "deadline": o.deadline,
                        "dispatch": t_d, "rc_km": self.dm.dist(o.rest, o.cust)})
            self.records[o.id] = rec

    def breakdown_event(self):
        """A rider breaks down: everything not delivered by then is released; the rider is
        out of action for `downtime` minutes."""
        tb, ridx, down = self.breakdown
        r = self.riders[ridx]
        released_ids = set()
        kept = []
        for plan in r.plans:
            rel = {s["oid"] for s in plan["stops"] if s["kind"] == 1 and s["arrive"] > tb}
            if not rel:
                kept.append(plan)
                continue
            released_ids |= rel
            stops = [s for s in plan["stops"] if s["arrive"] <= tb]
            if not stops:
                continue
            plan["stops"] = stops
            plan["dist"] = stops[-1]["cum_dist"]
            plan["idle"] = sum(s["idle"] for s in stops)
            plan["last_pos"] = stops[-1]["pos"]
            plan["n_orders"] = sum(1 for s in stops if s["kind"] == 1)
            plan["finish"] = stops[-1]["depart"]
            kept.append(plan)
        r.plans = kept
        r.pos = kept[-1]["last_pos"] if kept else r.init_pos
        r.available_at = tb + down
        out = []
        for oid in sorted(released_ids):
            self.records.pop(oid, None)
            out.append(self.omap[oid])
        return out

    # ------------------------------------------------------------------ main loop
    def run(self):
        t_wall = time.perf_counter()
        A, W = self.A, self.W
        nobatch = A.mode == "nobatch"
        B = 1 if nobatch else W.batch_size
        planner = PLANNERS[A.mode]
        future = deque(sorted(self.orders, key=lambda o: o.t_order))
        pending: List = []
        bd = self.breakdown
        bd_done = False
        next_t = W.sim_start + (0.0 if nobatch else W.window)
        guard = 0
        while (future or pending) and guard < 200000:
            guard += 1
            if not pending and future and future[0].t_order > next_t:
                if nobatch:
                    next_t = future[0].t_order
                else:
                    w = max(1.0, self.window_now(next_t))
                    next_t += w * math.ceil((future[0].t_order - next_t) / w)
            t_d = next_t
            if bd and not bd_done and bd[0] <= t_d:
                rel = self.breakdown_event()
                bd_done = True
                if A.handle_disruption:
                    pending.extend(rel)
                    self.reassigned += len(rel)
                else:
                    self.stranded.extend(rel)
            while future and future[0].t_order <= t_d:
                o = future.popleft()
                self.arrived.append(o)
                pending.append(o)
            keep = []
            for o in pending:
                if o.cancel_at is not None and o.cancel_at <= t_d and o.id not in self.dispatched:
                    self.cancelled.append(o)
                else:
                    keep.append(o)
            pending = keep
            if not pending:
                next_t = t_d + (0.0 if nobatch else max(1.0, self.window_now(t_d)))
                if not future:
                    break
                continue
            if A.use_priority:
                pending.sort(key=lambda o: (PRIORITY_RANK[o.priority], o.deadline))
            else:
                pending.sort(key=lambda o: o.deadline)
            batch, pending = pending[:B], pending[B:]
            self.reposition(t_d)
            slots = []
            for r in self.riders:
                start = max(t_d, r.available_at)
                slots.append(Slot(r.id, r.pos, start, r.speed if A.use_hetero else DEFAULT_SPEED,
                                  r.cap, r.km(), start - t_d))
            ctx = Ctx(self.omap, self.dm, W, A, self.rng, t_d)
            routes, left = planner(batch, slots, ctx)
            self.batches += 1
            self.batch_sizes.append(len(batch))
            for ridx, stops in routes.items():
                self.commit(ridx, stops, t_d, self.batches, len(batch))
            pending = left + pending
            w = 0.0 if nobatch else max(1.0, self.window_now(t_d))
            next_t = t_d + (w if routes else max(w, 1.0))
        self.compute_metrics(time.perf_counter() - t_wall)
        return self

    # ------------------------------------------------------------------ metrics
    def compute_metrics(self, runtime: float):
        W = self.W
        recs = list(self.records.values())
        n = len(recs)
        late = [x for x in recs if x["late"] > 1e-6]
        plans = [p for r in self.riders for p in r.plans]
        dist = sum(p["dist"] for p in plans)
        km_r = [r.km() for r in self.riders]
        mean = sum(km_r) / len(km_r)
        std = math.sqrt(sum((k - mean) ** 2 for k in km_r) / len(km_r))
        fuel = co2 = 0.0
        for r in self.riders:
            info = VEHICLES[r.vehicle]
            k = r.km() + r.repo_km
            fuel += k * info["fuel"]
            co2 += k * info["co2"]
        active = W.n_orders - len(self.cancelled)
        trips = len(plans)
        delivery = [x["delivered"] - x["t_order"] for x in recs]
        late_w = sum(x["late"] * PRIORITY_WEIGHT[x["priority"]] for x in recs)
        food = sum(x["age"] for x in recs)
        m = {
            "distance_km": dist,
            "reposition_km": self.reposition_km,
            "late_orders": float(len(late)),
            "on_time_pct": 100.0 * (n - len(late)) / active if active else 0.0,
            "total_lateness_min": sum(x["late"] for x in recs),
            "avg_delivery_min": sum(delivery) / n if n else 0.0,
            "max_delivery_min": max(delivery) if n else 0.0,
            "avg_food_age_min": food / n if n else 0.0,
            "fresh_violations": float(sum(1 for x in recs if x["transit"] > W.fresh_limit + 1e-6)),
            "idle_min": sum(p["idle"] for p in plans),
            "rider_km_std": std,
            "rider_km_gap": max(km_r) - min(km_r),
            "riders_used": float(sum(1 for k in km_r if k > 0)),
            "batches": float(self.batches),
            "avg_orders_per_trip": (sum(p["n_orders"] for p in plans) / trips) if trips else 0.0,
            "avg_first_pickup_min": (sum(p["first_pickup"] for p in plans) / trips) if trips else 0.0,
            "co2_kg": co2 / 1000.0,
            "fuel_cost_inr": fuel,
            "delivered": float(n),
            "cancelled": float(len(self.cancelled)),
            "reassigned": float(self.reassigned),
            "undelivered": float(len(self.stranded)),
            "runtime_s": runtime,
            "composite": dist + 2.0 * late_w + 0.1 * food + 100.0 * len(self.stranded),
        }
        for p in ("express", "normal", "scheduled"):
            m["late_" + p] = float(sum(1 for x in late if x["priority"] == p))
        self.metrics = m
        return m

    def all_plans(self):
        out = [p for r in self.riders for p in r.plans]
        return sorted(out, key=lambda p: (p["batch"], p["rider"]))
