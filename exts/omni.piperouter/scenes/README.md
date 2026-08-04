# Constraint test scenes

Three USD scenes, each built so that one constraint is the only thing deciding the route.
Geometry is plain boxes with known dimensions, so a wrong answer is visible by eye and the
numbers can be checked by hand.

Open a `.usd` in Kit, open the PipeRouter panel, press ROUTE ALL. Each scene carries its
own wires and a sensible grid resolution.

Regenerate with:

```bash
python3 exts/omni.piperouter/scenes/make_test_scenes.py
```

## test_bend.usd

Five channels sealed at both ends, each with a central divider that stops short of the far
wall. A cable starting in one lane can only reach the other by running the length of the
channel and turning 180° around the divider tip, so the turn cannot be avoided and its
radius is fixed by geometry: `R = divider/2 + lane/2`, printed in each wire name.

Every channel is routed with `cooling_main_24`, **rated for a 120 mm minimum bend radius**.
Available radii are **152, 103, 73, 42 and 25 mm**, so only the first can hold a legal turn.
The other four are impossible by construction.

Also in the scene:

- **`uturn_R152mm_bend1` vs `uturn_R152mm_bend8`** — identical geometry and pipe, bend
  weight 1 against 8. Tests whether the bend slider actually buys a wider turn.
- **`corner_sig_can` / `corner_brake_line_6` / `corner_cooling_main_24`** — one right-angle
  corner routed with ratings of 20, 30 and 120 mm. A stiffer pipe should cut the corner
  noticeably wider.

What to look for: the router does **not** refuse the impossible channels — the bend rating
is a soft penalty, so it takes the tight turn and reports success. Measured at 8.5 mm cells,
even the legal 152 mm channel comes back with turns far tighter than the rating, because
the length saved outweighs the penalty. This scene is the reproducer for that.

## test_clearance.usd

One wall pierced by slots of **150, 80, 40, 24 and 14 mm**, routed with `sig_can`
(4 mm outer diameter). All five pass at zero clearance.

Raise the safety clearance in the panel and the slots should close from the narrowest
upward: at 10 mm clearance the 14 and 24 mm slots become too tight, and so on. That
staircase is the test. It also exercises the endpoint clearance waiver, since the markers
sit in open space either side of the wall.

## test_thermal_em.usd

A 6 × 5 m bay with no walls at all, so the only thing bending a route is a field. Two
sources sit 2.6 m apart: a 300 °C manifold and an EM emitter. The bay has to be this large
because every tagged prim radiates over its own size plus a one-metre margin, and in a
small bay the field swamps everything and nothing routes.

**Melt rating** — three wires share a straight 5.2 m crossing of the manifold, with the
soft thermal weight set to **0** so the hard cutoff at each wire's own rating is the only
thing acting. Expected: detour length ordered by rating. Measured at resolution 200:

| wire | rating | length | detour |
|---|---|---|---|
| `hot_sig_can_90C` | 90 °C | 6097 mm | 897 mm |
| `hot_ac_pipe_135C` | 135 °C | 5915 mm | 715 mm |
| `hot_brake_160C` | 160 °C | 5834 mm | 634 mm |

**EM sensitivity** — two wires share a crossing past the emitter. `sig_can` has a
sensitivity of 0.9, `brake_line_6` has 0.0.

| wire | sensitivity | length |
|---|---|---|
| `em_sig_can_sens09` | 0.9 | 6423 mm |
| `em_brake_sens00` | 0.0 | 5728 mm |

The 695 mm spread is the EM constraint doing its job on identical endpoints.

## A note on resolution

The scenes save a resolution that works. Raising it a lot can make wires report `no_path`
rather than route: above roughly one billion lattice edges the planner declines to build
the exhaustive fallback and gives up instead. The thermal scene at resolution 300 hits this
and drops three wires whose endpoints are provably in open, cool, connected space — worth
knowing before blaming the constraint.
