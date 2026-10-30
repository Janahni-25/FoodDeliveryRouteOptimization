"""
model.py - data model, distance models, physical route evaluator and world generator
for TABPC+ (Time-Aware Batching and Precedence-Constrained routing, enhanced).

Units: time in minutes since midnight, distance in km, speed in km/h.
A route is a list of stops; a stop is (order_id, kind) with kind 0 = pickup, 1 = drop.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace  # noqa: F401  (replace re-exported)
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]
Stop = Tuple[int, int]

PRIORITIES = ("express", "normal", "scheduled")
PRIORITY_RANK = {"express": 0, "normal": 1, "scheduled": 2}
PRIORITY_WEIGHT = {"express": 3.0, "normal": 1.0, "scheduled": 0.5}

# Heterogeneous fleet. fuel = INR per km, co2 = grams per km (assumed illustrative factors).
VEHICLES = {
    "cycle": {"speed": 12.0, "cap_delta": -2, "fuel": 0.0, "co2": 0.0},
    "scooter": {"speed": 22.0, "cap_delta": 0, "fuel": 2.0, "co2": 45.0},
    "bike": {"speed": 30.0, "cap_delta": 1, "fuel": 3.0, "co2": 70.0},
}
DEFAULT_SPEED = 22.0


# --------------------------------------------------------------------------- configs
@dataclass
class WorldConfig:
    """Everything that describes the (true) world a plan is executed in."""
    n_orders: int = 40
    n_riders: int = 4
    batch_size: int = 10            # max orders processed in one batch
    rider_limit: int = 3            # base orders per rider per batch
    seed: int = 0
    grid: float = 200.0             # synthetic grid (0,0)-(grid,grid) as in the paper
    km_per_unit: float = 0.03       # 200 units = 6 km, a compact delivery zone
    n_restaurants: int = 8
    sim_start: float = 660.0        # 11:00
    sim_span: float = 240.0         # orders arrive over 4 hours
    window: float = 10.0            # base batching window (min)
    window_min: float = 4.0
    window_max: float = 20.0
    traffic: bool = True            # time-of-day slowdowns exist in the world
    peak_factor: float = 1.4        # travel-time multiplier at 12-14h and 19-21h
    hetero: bool = True             # mixed fleet (cycle / scooter / bike)
    fleet: Tuple[str, ...] = ("bike", "scooter", "scooter", "cycle", "bike")
    prep_range: Tuple[float, float] = (6.0, 18.0)
    p_priority: Tuple[float, float, float] = (0.2, 0.65, 0.15)
    deadlines: Tuple[float, float, float] = (45.0, 65.0, 95.0)   # minutes after order
    handling: float = 1.0           # minutes spent at each pickup / drop
    fresh_limit: float = 30.0       # max minutes between pickup and drop
    cancel_prob: float = 0.0
    breakdown: bool = False
    breakdown_frac: float = 0.4
    breakdown_downtime: float = 30.0


@dataclass
class AlgoConfig:
    """Planner settings. use_* flags control what the PLANNER knows / does."""
    name: str = "TABPC+"
    mode: str = "tabpc"             # tabpc | nn | nobatch | gls
    alpha: float = 1.0              # weight on distance (km)
    beta: float = 2.0               # weight on lateness (min)
    gamma: float = 0.1              # weight on food age (min)
    delta: float = 0.5              # weight on rider workload spread (km std)
    fresh_penalty: float = 5.0      # per minute over the freshness limit
    use_prep: bool = True
    use_traffic: bool = True
    use_priority: bool = True
    use_hetero: bool = True
    use_fair: bool = True
    use_fresh: bool = True
    use_sa: bool = True
    use_inter: bool = True
    use_pair: bool = True
    sa_iters: int = 400
    adaptive: bool = True
    reposition: bool = True
    handle_disruption: bool = True
    gls_seconds: float = 1.0


def _naive(name: str, mode: str) -> AlgoConfig:
    return AlgoConfig(
        name=name, mode=mode, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0,
        use_prep=False, use_traffic=False, use_priority=False, use_hetero=False,
        use_fair=False, use_fresh=False, use_sa=False, use_inter=False, use_pair=False,
        adaptive=False, reposition=False, handle_disruption=False,
    )


PRESET_NAMES = ["TABPC+", "Original TABPC", "GLS (OR-Tools)", "Nearest Neighbour", "No batching"]


def preset(name: str) -> AlgoConfig:
    if name == "TABPC+":
        return AlgoConfig(name=name)
    if name == "Original TABPC":
        return _naive(name, "tabpc")
    if name == "GLS (OR-Tools)":
        return _naive(name, "gls")
    if name == "Nearest Neighbour":
        return _naive(name, "nn")
    if name == "No batching":
        return _naive(name, "nobatch")
    raise KeyError(name)


# --------------------------------------------------------------------------- entities
@dataclass
class Order:
    id: int
    t_order: float
    rest_id: int
    rest: Point
    cust: Point
    prep: float
    priority: str
    deadline: float
    cancel_at: Optional[float] = None

    @property
    def ready(self) -> float:
        return self.t_order + self.prep


@dataclass
class Rider:
    id: int
    vehicle: str
    speed: float
    cap: int
    pos: Point
    available_at: float
    init_pos: Point = None
    plans: list = field(default_factory=list)
    repo_km: float = 0.0

    def __post_init__(self):
        if self.init_pos is None:
            self.init_pos = self.pos

    def km(self) -> float:
        return sum(p["dist"] for p in self.plans)


def fmt_time(t: float) -> str:
    t = t % 1440.0
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


# --------------------------------------------------------------------------- distance models
class Euclid:
    """Synthetic grid (as in the paper) placed on a map around Chennai for display."""
    name = "Synthetic grid"

    def __init__(self, km_per_unit: float = 0.03, grid: float = 200.0, origin=(13.0827, 80.2707)):
        self.k = km_per_unit
        self.grid = grid
        self.origin = origin

    def dist(self, a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1]) * self.k

    def sample_point(self, rng: random.Random) -> Point:
        return (rng.uniform(0, self.grid), rng.uniform(0, self.grid))

    def move_toward(self, p: Point, target: Point, max_km: float) -> Point:
        d = self.dist(p, target)
        if d <= max_km or d == 0:
            return target
        f = max_km / d
        return (p[0] + (target[0] - p[0]) * f, p[1] + (target[1] - p[1]) * f)

    def to_latlon(self, p: Point) -> Point:
        lat0, lon0 = self.origin
        north = (p[1] - self.grid / 2) * self.k
        east = (p[0] - self.grid / 2) * self.k
        return (lat0 + north / 110.574, lon0 + east / (111.320 * math.cos(math.radians(lat0))))


class RoadNetwork:
    """Real road distances from OpenStreetMap (needs `pip install osmnx` and internet).
    Points are (lon, lat) of road-network nodes."""
    name = "Real roads (OpenStreetMap)"

    def __init__(self, center=(11.0168, 76.9558), radius_m: int = 4000):  # Coimbatore centre
        import osmnx as ox
        import networkx as nx
        G = ox.graph_from_point(center, dist=radius_m, network_type="drive")
        G = ox.truncate.largest_component(G, strongly=True)
        self.ox, self.nx, self.G = ox, nx, G
        self.nodes = list(G.nodes)
        self.xy = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in self.nodes}
        self.node_of = {v: k for k, v in self.xy.items()}
        self._near: Dict[Point, int] = {}
        self._sp: Dict[int, dict] = {}

    def snap(self, p: Point) -> int:
        n = self.node_of.get(p)
        if n is not None:
            return n
        n = self._near.get(p)
        if n is None:
            n = self.ox.distance.nearest_nodes(self.G, X=p[0], Y=p[1])
            self._near[p] = n
        return n

    def dist(self, a: Point, b: Point) -> float:
        na, nb = self.snap(a), self.snap(b)
        if na == nb:
            return 0.0
        d = self._sp.get(na)
        if d is None:
            d = self.nx.single_source_dijkstra_path_length(self.G, na, weight="length", cutoff=40000)
            self._sp[na] = d
        v = d.get(nb)
        if v is None:  # unreachable inside cutoff: fall back to crow-flies x 1.4
            v = 1.4 * 111000 * math.hypot(a[0] - b[0], a[1] - b[1])
        return v / 1000.0

    def sample_point(self, rng: random.Random) -> Point:
        return self.xy[rng.choice(self.nodes)]

    def move_toward(self, p: Point, target: Point, max_km: float) -> Point:
        return target if self.dist(p, target) <= max_km else p

    def to_latlon(self, p: Point) -> Point:
        return (p[1], p[0])


# --------------------------------------------------------------------------- physics
def traffic_mult(t: float, W: WorldConfig) -> float:
    h = (t % 1440.0) / 60.0
    return W.peak_factor if (12.0 <= h < 14.0 or 19.0 <= h < 21.0) else 1.0


class RouteEval:
    __slots__ = ("dist", "finish", "idle", "food", "late", "late_w", "n_late", "exc",
                 "cost", "last_pos", "timeline", "delivs")


def eval_route(stops: List[Stop], pos: Point, t0: float, speed: float, orders: Dict[int, Order],
               dm, W: WorldConfig, prep: bool = True, traffic: bool = True,
               A: Optional[AlgoConfig] = None, detail: bool = False) -> RouteEval:
    """Simulate one rider driving `stops` starting at `pos` at time `t0`.

    prep / traffic say whether the evaluator *knows* about preparation time and traffic
    (planners that are unaware pass False; the true world always passes True).
    """
    t = t0
    dist = idle = food = late = late_w = exc = 0.0
    n_late = 0
    pk: Dict[int, Tuple[float, float]] = {}
    tl = [] if detail else None
    dv = [] if detail else None
    hand = W.handling
    use_tr = traffic and W.traffic
    cum = 0.0
    rank = 0
    for oid, kind in stops:
        o = orders[oid]
        p = o.rest if kind == 0 else o.cust
        d = dm.dist(pos, p)
        tt = d / speed * 60.0
        if use_tr:
            tt *= traffic_mult(t, W)
        arr = t + tt
        dist += d
        cum += d
        pos = p
        if kind == 0:
            ready = (o.t_order + o.prep) if prep else arr
            pt = arr if arr >= ready else ready
            wait = pt - arr
            idle += wait
            pk[oid] = (pt, ready)
            t = pt + hand
            if detail:
                tl.append({"oid": oid, "kind": 0, "pos": p, "arrive": arr, "depart": t,
                           "cum_dist": cum, "idle": wait})
        else:
            pt, ready = pk[oid]
            age = arr - ready
            food += age
            ld = arr - o.deadline
            if ld > 0:
                late += ld
                late_w += ld * PRIORITY_WEIGHT[o.priority]
                n_late += 1
            else:
                ld = 0.0
            ex = (arr - pt) - W.fresh_limit
            if ex > 0:
                exc += ex
            t = arr + hand
            if detail:
                tl.append({"oid": oid, "kind": 1, "pos": p, "arrive": arr, "depart": t,
                           "cum_dist": cum, "idle": 0.0})
                dv.append({"oid": oid, "pickup": pt, "delivered": arr, "ready": ready,
                           "transit": arr - pt, "age": age, "late": ld, "stop_rank": rank})
                rank += 1
    r = RouteEval()
    r.dist, r.finish, r.idle, r.food = dist, t, idle, food
    r.late, r.late_w, r.n_late, r.exc = late, late_w, n_late, exc
    r.last_pos, r.timeline, r.delivs = pos, tl, dv
    if A is not None:
        r.cost = (A.alpha * dist + A.beta * (late_w if A.use_priority else late)
                  + A.gamma * food + (A.fresh_penalty * exc if A.use_fresh else 0.0))
    else:
        r.cost = 0.0
    return r


# --------------------------------------------------------------------------- world generation
def _demand(t: float) -> float:
    """Relative order intensity: base + lunch peak (12:45) + dinner peak (19:45)."""
    t = t % 1440.0
    return (1.0 + 3.0 * math.exp(-0.5 * ((t - 765.0) / 35.0) ** 2)
            + 3.5 * math.exp(-0.5 * ((t - 1185.0) / 45.0) ** 2))


def sample_arrivals(n: int, start: float, span: float, rng: random.Random) -> List[float]:
    lam_max = 1.0 + 3.0 + 3.5
    out: List[float] = []
    while len(out) < n:
        t = rng.uniform(start, start + span)
        if rng.random() < _demand(t) / lam_max:
            out.append(t)
    return sorted(out)


def generate_world(W: WorldConfig, dm):
    """Returns (orders, riders, breakdown). breakdown = (time, rider_index, downtime) or None.
    Random draws do not depend on the disruption flags, so toggling them keeps the same orders."""
    rng = random.Random(W.seed)
    rests = [dm.sample_point(rng) for _ in range(W.n_restaurants)]
    base_prep = [rng.uniform(*W.prep_range) for _ in rests]
    arrivals = sample_arrivals(W.n_orders, W.sim_start, W.sim_span, rng)
    orders: List[Order] = []
    for i, t in enumerate(arrivals):
        r = rng.randrange(len(rests))
        prio = rng.choices(PRIORITIES, weights=W.p_priority)[0]
        prep = max(3.0, base_prep[r] + rng.uniform(-2.0, 2.0))
        cust = dm.sample_point(rng)
        cdraw, coff = rng.random(), rng.uniform(2.0, 12.0)
        cancel = t + coff if cdraw < W.cancel_prob else None
        dl = W.deadlines[PRIORITIES.index(prio)]
        orders.append(Order(i, t, r, rests[r], cust, prep, prio, t + dl, cancel))
    riders: List[Rider] = []
    for k in range(W.n_riders):
        veh = W.fleet[k % len(W.fleet)] if W.hetero else "scooter"
        info = VEHICLES[veh]
        cap = max(1, W.rider_limit + info["cap_delta"]) if W.hetero else W.rider_limit
        riders.append(Rider(k, veh, info["speed"], cap, dm.sample_point(rng), W.sim_start))
    bd_rider = rng.randrange(W.n_riders)
    bd = (W.sim_start + W.breakdown_frac * W.sim_span, bd_rider, W.breakdown_downtime) if W.breakdown else None
    return orders, riders, bd
