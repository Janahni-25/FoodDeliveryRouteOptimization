"""
planners.py - batch planners.

  plan_tabpc    TABPC (original, distance-only) and TABPC+ (time-aware multi-objective with
                simulated annealing, inter-route and pair-aware moves) - controlled by AlgoConfig flags
  plan_nn       Nearest Neighbour baseline
  plan_nobatch  one order per trip, earliest-arriving rider (no batching baseline)
  plan_gls      Guided Local Search from Google OR-Tools (as used in the paper)

Every planner: planner(batch, slots, ctx) -> (routes {rider_index: stops}, leftover_orders)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from model import (AlgoConfig, Order, PRIORITY_RANK, Point, Stop, WorldConfig, eval_route)

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    HAS_ORTOOLS = True
except Exception:  # pragma: no cover
    HAS_ORTOOLS = False


@dataclass
class Slot:
    """A rider as seen by the planner for one batch."""
    idx: int
    pos: Point
    start: float       # time the rider can start this route
    speed: float       # speed the PLANNER assumes (actual if use_hetero, else a default)
    cap: int           # max orders in this route
    base_km: float     # distance already driven (for the fairness term)
    delay: float       # start - dispatch time (how busy the rider still is)


@dataclass
class Ctx:
    orders: Dict[int, Order]
    dm: object
    W: WorldConfig
    A: AlgoConfig
    rng: random.Random
    t_d: float


def valid(stops: List[Stop]) -> bool:
    seen = set()
    for oid, k in stops:
        if k == 0:
            seen.add(oid)
        elif oid not in seen:
            return False
    return True


def _order_ids(route: List[Stop]) -> List[int]:
    return [oid for oid, k in route if k == 0]


def _remove(route: List[Stop], oid: int) -> List[Stop]:
    return [s for s in route if s[0] != oid]


def _insert_random(route: List[Stop], oid: int, rng: random.Random) -> List[Stop]:
    L = len(route)
    p = rng.randint(0, L)
    q = rng.randint(p, L)
    return route[:p] + [(oid, 0)] + route[p:q] + [(oid, 1)] + route[q:]


# =========================================================================== TABPC / TABPC+
class TabpcSolver:
    def __init__(self, ctx: Ctx, slots: List[Slot]):
        self.ctx, self.slots, self.R = ctx, slots, len(slots)
        self.routes: List[List[Stop]] = [[] for _ in slots]
        self.cache: Dict[tuple, object] = {}
        self.evs = [self.ev(i, []) for i in range(self.R)]

    # -- evaluation ---------------------------------------------------------
    def ev(self, i: int, stops: List[Stop]):
        key = (i, tuple(stops))
        e = self.cache.get(key)
        if e is None:
            c, s = self.ctx, self.slots[i]
            e = eval_route(stops, s.pos, s.start, s.speed, c.orders, c.dm, c.W,
                           prep=c.A.use_prep, traffic=c.A.use_traffic, A=c.A)
            self.cache[key] = e
        return e

    def fair(self, evs) -> float:
        A = self.ctx.A
        if not A.use_fair or A.delta <= 0 or self.R < 2:
            return 0.0
        vals = [s.base_km + e.dist for s, e in zip(self.slots, evs)]
        m = sum(vals) / len(vals)
        return A.delta * math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    def total(self, evs) -> float:
        return sum(e.cost for e in evs) + self.fair(evs)

    def has_room(self, i: int, route: List[Stop] = None) -> bool:
        r = self.routes[i] if route is None else route
        return len(r) // 2 < self.slots[i].cap

    # -- insertion ----------------------------------------------------------
    def best_insert(self, oid: int, i: int, base: List[Stop]):
        L = len(base)
        best = None
        for p in range(L + 1):
            for q in range(p, L + 1):
                new = base[:p] + [(oid, 0)] + base[p:q] + [(oid, 1)] + base[q:]
                e = self.ev(i, new)
                if best is None or e.cost < best[0].cost:
                    best = (e, new)
        return best

    def insert_order(self, oid: int) -> bool:
        cur = self.total(self.evs)
        best = None
        for i in range(self.R):
            if not self.has_room(i):
                continue
            e, new = self.best_insert(oid, i, self.routes[i])
            evs2 = self.evs[:]
            evs2[i] = e
            delta = self.total(evs2) - cur
            if best is None or delta < best[0]:
                best = (delta, i, new, e)
        if best is None:
            return False
        _, i, new, e = best
        self.routes[i], self.evs[i] = new, e
        return True

    # -- construction phase -------------------------------------------------
    def construct(self, ordered: List[Order]) -> List[Order]:
        c = self.ctx
        time_aware = c.A.beta > 0 or c.A.gamma > 0
        free = set(range(self.R))
        first, rest = ordered[: self.R], list(ordered[self.R:])
        for o in first:
            if not free:
                rest.insert(0, o)
                continue

            def key(i, o=o):
                s = self.slots[i]
                tt = (c.dm.dist(s.pos, o.rest) + c.dm.dist(o.rest, o.cust)) / s.speed * 60.0
                return tt + (s.delay if time_aware else 0.0)

            i = min(free, key=key)
            free.discard(i)
            stops = [(o.id, 0), (o.id, 1)]
            self.routes[i] = stops
            self.evs[i] = self.ev(i, stops)
        left = []
        for o in rest:
            if not self.insert_order(o.id):
                left.append(o)
        return left

    # -- improvement: original paper (intra-route swap hill climbing) -------
    def improve_swap(self, iters: int):
        rng = self.ctx.rng
        cur = self.total(self.evs)
        for _ in range(iters):
            cand = [i for i in range(self.R) if len(self.routes[i]) >= 3]
            if not cand:
                break
            i = rng.choice(cand)
            s = self.routes[i][:]
            a, b = rng.sample(range(len(s)), 2)
            s[a], s[b] = s[b], s[a]
            if not valid(s):
                continue
            evs2 = self.evs[:]
            evs2[i] = self.ev(i, s)
            t = self.total(evs2)
            if t < cur - 1e-9:
                self.routes[i], self.evs, cur = s, evs2, t

    # -- improvement: TABPC+ (simulated annealing) --------------------------
    def propose(self, mv: str):
        rng = self.ctx.rng
        R = self.R
        routes = self.routes
        if mv == "swap":
            cand = [i for i in range(R) if len(routes[i]) >= 3]
            if not cand:
                return None
            i = rng.choice(cand)
            s = routes[i][:]
            a, b = rng.sample(range(len(s)), 2)
            s[a], s[b] = s[b], s[a]
            return {i: s} if valid(s) else None
        if mv == "relocate":      # inter-route: whole pickup-drop pair moves to another rider
            src = [i for i in range(R) if routes[i]]
            if not src or R < 2:
                return None
            i = rng.choice(src)
            tgt = [j for j in range(R) if j != i and self.has_room(j)]
            if not tgt:
                return None
            j = rng.choice(tgt)
            oid = rng.choice(_order_ids(routes[i]))
            return {i: _remove(routes[i], oid), j: _insert_random(routes[j], oid, rng)}
        if mv == "exchange":      # inter-route: two orders swap riders
            src = [i for i in range(R) if routes[i]]
            if len(src) < 2:
                return None
            i, j = rng.sample(src, 2)
            a = rng.choice(_order_ids(routes[i]))
            b = rng.choice(_order_ids(routes[j]))
            ni = _insert_random(_remove(routes[i], a), b, rng)
            nj = _insert_random(_remove(routes[j], b), a, rng)
            return {i: ni, j: nj}
        if mv == "pairshift":     # pair-aware: re-position a pickup+drop pair inside its route
            src = [i for i in range(R) if len(routes[i]) >= 4]
            if not src:
                return None
            i = rng.choice(src)
            oid = rng.choice(_order_ids(routes[i]))
            new = _insert_random(_remove(routes[i], oid), oid, rng)
            return {i: new} if new != routes[i] else None
        if mv == "pairbest":      # pair-aware: remove a pair, re-insert it at its cheapest place
            src = [i for i in range(R) if routes[i]]
            if not src:
                return None
            i = rng.choice(src)
            oid = rng.choice(_order_ids(routes[i]))
            base_i = _remove(routes[i], oid)
            cur_routes = routes[:]
            cur_routes[i] = base_i
            evs_base = self.evs[:]
            evs_base[i] = self.ev(i, base_i)
            best = None
            for j in range(R):
                if not self.has_room(j, cur_routes[j]):
                    continue
                e, new = self.best_insert(oid, j, cur_routes[j])
                evs2 = evs_base[:]
                evs2[j] = e
                tot = self.total(evs2)
                if best is None or tot < best[0]:
                    best = (tot, j, new)
            if best is None:
                return None
            _, j, new = best
            if j == i:
                return {i: new} if new != routes[i] else None
            return {i: base_i, j: new}
        return None

    def anneal(self, iters: int):
        A, rng = self.ctx.A, self.ctx.rng
        moves = ["swap"]
        if A.use_inter:
            moves += ["relocate", "exchange"]
        if A.use_pair:
            moves += ["pairshift", "pairbest"]
        cur = self.total(self.evs)
        best = cur
        best_routes = [r[:] for r in self.routes]
        T = max(1.0, 0.05 * cur)
        cool = (0.01) ** (1.0 / max(1, iters))
        for _ in range(iters):
            ch = self.propose(rng.choice(moves))
            if ch:
                evs2 = self.evs[:]
                for i, s in ch.items():
                    evs2[i] = self.ev(i, s)
                new = self.total(evs2)
                d = new - cur
                if d <= 0 or rng.random() < math.exp(-d / T):
                    for i, s in ch.items():
                        self.routes[i] = s
                    self.evs, cur = evs2, new
                    if cur < best - 1e-9:
                        best, best_routes = cur, [r[:] for r in self.routes]
            T *= cool
        self.routes = best_routes
        self.evs = [self.ev(i, r) for i, r in enumerate(self.routes)]


def plan_tabpc(batch: List[Order], slots: List[Slot], ctx: Ctx):
    A = ctx.A
    if A.use_priority:
        ordered = sorted(batch, key=lambda o: (PRIORITY_RANK[o.priority], o.deadline))
    else:
        ordered = sorted(batch, key=lambda o: o.deadline)
    sol = TabpcSolver(ctx, slots)
    left = sol.construct(ordered)
    n_stops = sum(len(r) for r in sol.routes)
    if A.use_sa:
        sol.anneal(A.sa_iters)
    else:
        sol.improve_swap(8 * max(1, n_stops))
    routes = {slots[i].idx: r for i, r in enumerate(sol.routes) if r}
    return routes, left


# =========================================================================== baselines
def plan_nn(batch: List[Order], slots: List[Slot], ctx: Ctx):
    dm, orders = ctx.dm, ctx.orders
    remaining = {o.id: o for o in batch}
    st = [{"pos": s.pos, "onboard": [], "n": 0, "stops": []} for s in slots]
    while True:
        best = None
        for i, s in enumerate(st):
            for oid in s["onboard"]:
                d = dm.dist(s["pos"], orders[oid].cust)
                if best is None or d < best[0]:
                    best = (d, i, oid, 1)
            if s["n"] < slots[i].cap:
                for oid, o in remaining.items():
                    d = dm.dist(s["pos"], o.rest)
                    if best is None or d < best[0]:
                        best = (d, i, oid, 0)
        if best is None:
            break
        _, i, oid, kind = best
        s = st[i]
        if kind == 0:
            o = remaining.pop(oid)
            s["onboard"].append(oid)
            s["n"] += 1
            s["pos"] = o.rest
        else:
            s["onboard"].remove(oid)
            s["pos"] = orders[oid].cust
        s["stops"].append((oid, kind))
    routes = {slots[i].idx: s["stops"] for i, s in enumerate(st) if s["stops"]}
    return routes, list(remaining.values())


def plan_nobatch(batch: List[Order], slots: List[Slot], ctx: Ctx):
    o = batch[0]
    dm = ctx.dm

    def eta(s: Slot):
        return s.delay + dm.dist(s.pos, o.rest) / s.speed * 60.0

    s = min(slots, key=eta)
    return {s.idx: [(o.id, 0), (o.id, 1)]}, list(batch[1:])


def plan_gls(batch: List[Order], slots: List[Slot], ctx: Ctx):
    if not HAS_ORTOOLS:
        raise RuntimeError("ortools is not installed: pip install ortools")
    n, R = len(batch), len(slots)
    pts = [s.pos for s in slots] + [o.rest for o in batch] + [o.cust for o in batch]
    end = len(pts)
    manager = pywrapcp.RoutingIndexManager(end + 1, R, list(range(R)), [end] * R)
    routing = pywrapcp.RoutingModel(manager)
    SC = 1000.0
    dm = ctx.dm
    memo: Dict[Tuple[int, int], int] = {}

    def dist_cb(fi, ti):
        a, b = manager.IndexToNode(fi), manager.IndexToNode(ti)
        if a == end or b == end:
            return 0
        v = memo.get((a, b))
        if v is None:
            v = int(dm.dist(pts[a], pts[b]) * SC)
            memo[(a, b)] = v
        return v

    tid = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tid)
    routing.AddDimension(tid, 0, 10 ** 9, True, "Dist")
    dim = routing.GetDimensionOrDie("Dist")

    def cnt_cb(fi):
        node = manager.IndexToNode(fi)
        return 1 if R <= node < R + n else 0

    cid = routing.RegisterUnaryTransitCallback(cnt_cb)
    routing.AddDimensionWithVehicleCapacity(cid, 0, [s.cap for s in slots], True, "Count")
    for k in range(n):
        p, d = manager.NodeToIndex(R + k), manager.NodeToIndex(R + n + k)
        routing.AddPickupAndDelivery(p, d)
        routing.solver().Add(routing.VehicleVar(p) == routing.VehicleVar(d))
        routing.solver().Add(dim.CumulVar(p) <= dim.CumulVar(d))
        routing.AddDisjunction([p, d], 10 ** 7, 2)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(ctx.A.gls_seconds * 1000))
    sol = routing.SolveWithParameters(params)
    if sol is None:
        return {}, list(batch)
    routes, served = {}, set()
    for v in range(R):
        idx = routing.Start(v)
        stops: List[Stop] = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if R <= node < R + n:
                oid = batch[node - R].id
                stops.append((oid, 0))
                served.add(oid)
            elif R + n <= node < end:
                stops.append((batch[node - R - n].id, 1))
            idx = sol.Value(routing.NextVar(idx))
        if stops:
            routes[slots[v].idx] = stops
    return routes, [o for o in batch if o.id not in served]


PLANNERS = {"tabpc": plan_tabpc, "nn": plan_nn, "nobatch": plan_nobatch, "gls": plan_gls}
