"""
Step 3: Simulation on top of the Kuhn Poker CFR solver (kuhn_cfr.py).

Part A - Matchups: the equilibrium strategy vs five exploitative opponents,
         and an "exploiter" (exact best response) vs the same opponents.
         Each reported both as exact expected value and as Monte Carlo play,
         with seats alternating so neither side gets the positional edge.
Part B - Deviation tests: what happens to a player's guaranteed value when
         they deviate from the equilibrium bluff / call frequencies.

Run:  python kuhn_simulation.py --hands 200000
Needs outputs/strategy_table.csv from kuhn_cfr.py (or retrains if missing).
Outputs (in ./outputs): matchups.csv, cumulative_profit.csv,
    exploit_strategies.csv, deviation_sweeps.csv, simulation_summary.json
"""

import argparse
import csv
import json
import os
import random
from itertools import product

from kuhn_cfr import (
    ALL_INFOSETS,
    P1_INFOSETS,
    P2_INFOSETS,
    average_sigma,
    infoset_key,
    p1_expected_value,
    p1_payoff,
    train,
)

NEG_INF = float("-inf")


# --------------------------------------------------------------------------
# Strategies: dict infoset -> P(bet). The same dict works for either seat,
# because P1 and P2 information sets have different keys.
# --------------------------------------------------------------------------
def make_strategy(**overrides):
    """Start from all-pass; override individual infosets."""
    sigma = {k: 0.0 for k in ALL_INFOSETS}
    sigma.update(overrides)
    return sigma


ARCHETYPES = {
    # Never bluffs, calls everything, bets only with the King.
    "Calling station": make_strategy(
        K=1, Kp=1, Jb=1, Qb=1, Kb=1, Jpb=1, Qpb=1, Kpb=1
    ),
    # Folds to every bet unless holding the King, bets only the King.
    "Over-folder": make_strategy(K=1, Kp=1, Kb=1, Kpb=1),
    # Bets and calls with every hand.
    "Maniac": make_strategy(**{k: 1.0 for k in ALL_INFOSETS}),
    # Bluffs the Jack every time, value bets the King, calls only with King.
    "Over-bluffer": make_strategy(J=1, K=1, Jp=1, Kp=1, Kb=1, Kpb=1),
    # Coin flip at every decision.
    "Random": make_strategy(**{k: 0.5 for k in ALL_INFOSETS}),
}


def load_equilibrium(path):
    """Load the CFR average strategy, or retrain if the file is missing."""
    if os.path.exists(path):
        with open(path) as f:
            return {row["infoset"]: float(row["p_bet"]) for row in csv.DictReader(f)}
    nodes, _ = train(100_000, 100_000)
    return average_sigma(nodes)


def merge(p1_strategy, p2_strategy):
    """Player 1's keys from one strategy, Player 2's keys from another."""
    merged = {k: p1_strategy[k] for k in P1_INFOSETS}
    merged.update({k: p2_strategy[k] for k in P2_INFOSETS})
    return merged


# --------------------------------------------------------------------------
# Exact evaluation
# --------------------------------------------------------------------------
def seat_averaged_ev(a, b):
    """Exact EV per hand for strategy a against b, averaged over both seats."""
    as_p1 = p1_expected_value(merge(a, b))
    as_p2 = -p1_expected_value(merge(b, a))
    return (as_p1 + as_p2) / 2.0, as_p1, as_p2


def best_p1_value(p2_strategy):
    """Best EV Player 1 can get against a fixed Player 2 (pure strategies)."""
    best, best_bits = NEG_INF, None
    for bits in product([0.0, 1.0], repeat=len(P1_INFOSETS)):
        trial = dict(p2_strategy)
        trial.update(zip(P1_INFOSETS, bits))
        v = p1_expected_value(trial)
        if v > best:
            best, best_bits = v, bits
    return best, dict(zip(P1_INFOSETS, best_bits))


def best_p2_value(p1_strategy):
    """Best EV Player 2 can get against a fixed Player 1 (pure strategies)."""
    best, best_bits = NEG_INF, None
    for bits in product([0.0, 1.0], repeat=len(P2_INFOSETS)):
        trial = dict(p1_strategy)
        trial.update(zip(P2_INFOSETS, bits))
        v = -p1_expected_value(trial)
        if v > best:
            best, best_bits = v, bits
    return best, dict(zip(P2_INFOSETS, best_bits))


def exploiter_against(opponent):
    """Exact best response to an opponent in both seats, as one strategy."""
    v1, s1 = best_p1_value(opponent)
    v2, s2 = best_p2_value(opponent)
    strategy = dict(s1)
    strategy.update(s2)
    return strategy, (v1 + v2) / 2.0


# --------------------------------------------------------------------------
# Monte Carlo play
# --------------------------------------------------------------------------
DECK = [0, 1, 2]


def play_hand(sigma_p1, sigma_p2, rng):
    """Play one hand; returns Player 1's payoff."""
    cards = rng.sample(DECK, 2)
    history = ""
    while True:
        payoff = p1_payoff(cards, history)
        if payoff is not None:
            return payoff
        player = len(history) % 2
        sigma = sigma_p1 if player == 0 else sigma_p2
        p_bet = sigma[infoset_key(cards, history)]
        history += "b" if rng.random() < p_bet else "p"


def simulate(a, b, hands, rng, checkpoint):
    """
    Strategy a plays b for `hands` hands, alternating seats each hand.
    Returns (profit per hand, [(hands_played, cumulative_profit), ...]).
    """
    total = 0.0
    curve = []
    for i in range(1, hands + 1):
        if i % 2 == 1:
            total += play_hand(a, b, rng)       # a is Player 1
        else:
            total -= play_hand(b, a, rng)       # a is Player 2
        if i % checkpoint == 0:
            curve.append((i, total))
    return total / hands, curve


# --------------------------------------------------------------------------
# Part B: deviation tests (theoretical equilibrium, exact fractions)
# --------------------------------------------------------------------------
def theory_p2():
    return make_strategy(Jb=0, Qb=1 / 3, Kb=1, Jp=1 / 3, Qp=0, Kp=1)


def theory_p1(alpha):
    return make_strategy(
        J=alpha, Q=0, K=min(1.0, 3 * alpha),
        Jpb=0, Qpb=min(1.0, alpha + 1 / 3), Kpb=1,
    )


def deviation_sweeps(grid_size=60):
    grid = [i / grid_size for i in range(grid_size + 1)]
    rows = []

    # Player 2 deviates on one frequency; Player 1 replies with a best response.
    for label, key in (
        ("P2 bluff freq with Jack after check (eq 1/3)", "Jp"),
        ("P2 call freq with Queen facing bet (eq 1/3)", "Qb"),
    ):
        for x in grid:
            p2 = theory_p2()
            p2[key] = x
            p1_best, _ = best_p1_value(p2)
            rows.append(
                {"sweep": label, "frequency": x,
                 "guaranteed_ev_of_deviator": -p1_best}
            )

    # Player 1 varies its bluff frequency alpha (other frequencies adjusted
    # to stay on the equilibrium family); Player 2 replies with a best response.
    label = "P1 bluff freq with Jack, alpha (eq range 0 to 1/3)"
    for x in grid:
        p2_best, _ = best_p2_value(theory_p1(x))
        rows.append(
            {"sweep": label, "frequency": x,
             "guaranteed_ev_of_deviator": -p2_best}
        )
    return rows


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=200_000)
    parser.add_argument("--checkpoint", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    equilibrium = load_equilibrium(os.path.join(args.out, "strategy_table.csv"))

    matchups, curves, exploit_rows = [], {}, []
    for name, opp in ARCHETYPES.items():
        exploiter, exploiter_ev_exact = exploiter_against(opp)
        eq_ev_exact, eq_as_p1, eq_as_p2 = seat_averaged_ev(equilibrium, opp)

        eq_mc, eq_curve = simulate(equilibrium, opp, args.hands, rng, args.checkpoint)
        ex_mc, ex_curve = simulate(exploiter, opp, args.hands, rng, args.checkpoint)
        curves[name] = (eq_curve, ex_curve)

        # How badly the opponent loses to a perfect counter-player
        # (their exploitability, seat-averaged).
        matchups.append(
            {
                "opponent": name,
                "equilibrium_ev_exact": round(eq_ev_exact, 5),
                "equilibrium_ev_simulated": round(eq_mc, 5),
                "exploiter_ev_exact": round(exploiter_ev_exact, 5),
                "exploiter_ev_simulated": round(ex_mc, 5),
                "extra_profit_from_exploiting": round(exploiter_ev_exact - eq_ev_exact, 5),
            }
        )
        for key in ALL_INFOSETS:
            exploit_rows.append(
                {"opponent": name, "infoset": key,
                 "equilibrium_p_bet": round(equilibrium[key], 4),
                 "exploiter_p_bet": exploiter[key]}
            )

    # ----- write outputs
    with open(os.path.join(args.out, "matchups.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(matchups[0].keys()))
        w.writeheader()
        w.writerows(matchups)

    with open(os.path.join(args.out, "exploit_strategies.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(exploit_rows[0].keys()))
        w.writeheader()
        w.writerows(exploit_rows)

    names = list(ARCHETYPES)
    n_points = len(curves[names[0]][0])
    with open(os.path.join(args.out, "cumulative_profit.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hands"] + [f"{n} | {who}" for n in names for who in ("Equilibrium", "Exploiter")])
        for i in range(n_points):
            row = [curves[names[0]][0][i][0]]
            for n in names:
                row += [round(curves[n][0][i][1], 1), round(curves[n][1][i][1], 1)]
            w.writerow(row)

    sweeps = deviation_sweeps()
    with open(os.path.join(args.out, "deviation_sweeps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweeps[0].keys()))
        w.writeheader()
        for r in sweeps:
            w.writerow({"sweep": r["sweep"], "frequency": round(r["frequency"], 4),
                        "guaranteed_ev_of_deviator": round(r["guaranteed_ev_of_deviator"], 6)})

    # ----- checks
    se = 1.5 / args.hands ** 0.5            # rough standard error of the mean
    tol = 4 * se
    checks = []
    for m in matchups:
        checks.append({
            "check": f"{m['opponent']}: equilibrium never loses (exact EV >= -0.002)",
            "pass": m["equilibrium_ev_exact"] >= -0.002,
        })
        checks.append({
            "check": f"{m['opponent']}: simulated equilibrium EV matches exact (within 4 SE = {tol:.4f})",
            "pass": abs(m["equilibrium_ev_simulated"] - m["equilibrium_ev_exact"]) <= tol,
        })
        checks.append({
            "check": f"{m['opponent']}: simulated exploiter EV matches exact (within 4 SE = {tol:.4f})",
            "pass": abs(m["exploiter_ev_simulated"] - m["exploiter_ev_exact"]) <= tol,
        })
        checks.append({
            "check": f"{m['opponent']}: exploiting earns at least as much as equilibrium",
            "pass": m["exploiter_ev_exact"] >= m["equilibrium_ev_exact"] - 1e-9,
        })

    for label in {r["sweep"] for r in sweeps}:
        pts = [r for r in sweeps if r["sweep"] == label]
        best = max(p["guaranteed_ev_of_deviator"] for p in pts)
        if label.startswith("P1"):
            flat = [p for p in pts if p["frequency"] <= 1 / 3 + 1e-9]
            checks.append({
                "check": "P1 alpha: guaranteed EV flat at -1/18 for alpha in [0, 1/3]",
                "pass": all(abs(p["guaranteed_ev_of_deviator"] + 1 / 18) < 1e-9 for p in flat),
            })
            beyond = [p for p in pts if p["frequency"] > 1 / 3 + 1e-9]
            checks.append({
                "check": "P1 alpha: guaranteed EV falls below -1/18 once alpha > 1/3",
                "pass": all(p["guaranteed_ev_of_deviator"] < -1 / 18 - 1e-9 for p in beyond),
            })
        else:
            at_eq = [p for p in pts if abs(p["frequency"] - 1 / 3) < 1e-9][0]
            checks.append({
                "check": f"{label}: guaranteed EV is maximised at 1/3 (= +1/18)",
                "pass": abs(at_eq["guaranteed_ev_of_deviator"] - best) < 1e-9
                and abs(best - 1 / 18) < 1e-9,
            })

    summary = {
        "hands_per_matchup": args.hands,
        "seed": args.seed,
        "matchups": matchups,
        "all_checks_passed": all(c["pass"] for c in checks),
        "checks": checks,
    }
    with open(os.path.join(args.out, "simulation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ----- console report
    print(f"Hands per matchup: {args.hands:,} (seats alternate)\n")
    print(f"{'Opponent':<17}{'Equil. exact':>13}{'Equil. sim':>12}{'Exploit exact':>15}{'Exploit sim':>13}")
    for m in matchups:
        print(f"{m['opponent']:<17}{m['equilibrium_ev_exact']:>13.4f}{m['equilibrium_ev_simulated']:>12.4f}"
              f"{m['exploiter_ev_exact']:>15.4f}{m['exploiter_ev_simulated']:>13.4f}")
    print("\nDeviation sweeps (guaranteed EV of the deviating player):")
    for label in sorted({r["sweep"] for r in sweeps}):
        pts = [r for r in sweeps if r["sweep"] == label]
        print(f"  {label}")
        for x in (0.0, 1 / 6, 1 / 3, 0.5, 2 / 3, 1.0):
            p = min(pts, key=lambda r: abs(r["frequency"] - x))
            print(f"    freq {p['frequency']:.3f}: {p['guaranteed_ev_of_deviator']:+.4f}")
    print("\nChecks:")
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}")
    print(f"\nAll checks passed: {summary['all_checks_passed']}")


if __name__ == "__main__":
    main()
