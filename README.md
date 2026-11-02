# TABPC+ : Time-Aware Batching and Precedence-Constrained Routing (enhanced)

Extension of the paper "Time and Precedence-Constrained Routing for Last-Mile Food Delivery"
(TABPC). Python engine plus a Streamlit GUI.

## Run

```
pip install -r requirements.txt
streamlit run app.py          # GUI
python selftest.py            # checks every algorithm produces valid plans
python analysis.py            # quick console comparison without the GUI
```

`ortools` is only needed for the Guided Local Search baseline and `osmnx` only for real roads;
the rest works without them.

## Files

| File | Purpose |
|------|---------|
| `model.py` | configs, entities, distance models (grid / OpenStreetMap), physical route evaluator, world generator |
| `planners.py` | TABPC / TABPC+, Nearest Neighbour, No batching, GLS (OR-Tools) |
| `simulator.py` | dynamic simulation, adaptive window, disruptions, repositioning, metrics |
| `analysis.py` | comparison, sensitivity, ablation, ETA prediction |
| `app.py` | Streamlit GUI |
| `selftest.py` | validity checks (precedence, capacity, no lost orders) |

## The 17 added features

1. Multi-objective cost: alpha x distance + beta x lateness + gamma x food age, plus a freshness limit
2. Restaurant preparation time (riders wait for food, idle time reported)
3. Order priority levels (express / normal / scheduled)
4. Rider fairness term (workload spread)
5. Simulated annealing improvement phase (replaces the single random swap)
6. Move families: intra-route swap, inter-route relocate and exchange, pair-aware shift and best re-insertion
7. Dynamic order arrival (lunch/dinner peaks) with adaptive batching window
8. Order cancellations and rider breakdown with re-assignment
9. Idle rider repositioning to demand hotspots
10. Heterogeneous riders (cycle / scooter / bike)
11. Traffic-aware travel times
12. Real road distances via OpenStreetMap (optional)
13. Baselines (original TABPC, GLS, Nearest Neighbour, No batching), many metrics, multiple seeds
14. Sensitivity analysis
15. Ablation study
16. Fuel cost and CO2 estimate
17. Map dashboard and customer ETA prediction (Random Forest, MAE)

## Notes for your report

* Planner flags (prep, traffic, priority, fleet speeds ...) control what the **planner knows**. Every plan is
  executed and scored in the same true world, which makes the ablation a fair test.
* "Original TABPC" here is the paper's method: distance-only cost, cluster-first/route-second construction,
  cheapest insertion, random intra-route swaps, fixed window. It is scored with the same metrics as TABPC+.
* The "composite score" = distance + 2 x priority-weighted lateness + 0.1 x food age (+ penalty for stranded orders).
* The Original TABPC and baselines have no disruption handling, so a breakdown strands their orders.
  Only TABPC+ re-assigns them.
* GLS minimises distance only (as in the paper). It gets about 1 second of search per batch (`gls_seconds`).
* CO2 and fuel factors are illustrative assumptions (`VEHICLES` in `model.py`). Replace them with real values.
* Results are on synthetic data (the paper has no public benchmark). Always report the mean over several seeds.
* Paper dataset 4 (100 orders) is overloaded with only 5 riders in this simulator, so the GUI preset uses 6 riders over 6 hours.
