"""
Kuhn Poker solver using Counterfactual Regret Minimization (CFR).

Game: 3 cards (J < Q < K), 2 players, each antes 1, one betting round.
Actions: 0 = pass (check/fold), 1 = bet (bet/call).

Information-set key = <own card><public history>, e.g. "Jb" = holding J,
facing a bet. History letters: p = pass, b = bet.

Run:  python kuhn_cfr.py --iterations 100000
Outputs (in ./outputs): strategy_table.csv, convergence.csv, summary.json
"""

import argparse
import csv
import json
import os
from itertools import permutations, product

CARD_NAMES = "JQK"
ACTIONS = ["pass", "bet"]
NUM_ACTIONS = 2


# --------------------------------------------------------------------------
# Game rules
# --------------------------------------------------------------------------
def p1_payoff(cards, history):
    """Payoff to Player 1 at a terminal history, or None if not terminal."""
    p1_higher = cards[0] > cards[1]
    if history == "pp":      # check, check -> showdown for 1
        return 1 if p1_higher else -1
    if history == "bp":      # P1 bets, P2 folds
        return 1
    if history == "bb":      # bet, call -> showdown for 2
        return 2 if p1_higher else -2
    if history == "pbp":     # check, bet, P1 folds
        return -1
    if history == "pbb":     # check, bet, call -> showdown for 2
        return 2 if p1_higher else -2
    return None


def infoset_key(cards, history):
    player = len(history) % 2
    return f"{CARD_NAMES[cards[player]]}{history}"


ALL_INFOSETS = sorted(
    [f"{c}{h}" for c in CARD_NAMES for h in ("", "pb")]   # Player 1
    + [f"{c}{h}" for c in CARD_NAMES for h in ("p", "b")]  # Player 2
)
P1_INFOSETS = [k for k in ALL_INFOSETS if len(k) == 1 or k.endswith("pb")]
P2_INFOSETS = [k for k in ALL_INFOSETS if k not in P1_INFOSETS]


# --------------------------------------------------------------------------
# CFR
# --------------------------------------------------------------------------
class Node:
    """Regret and strategy accumulators for one information set."""

    def __init__(self):
        self.regret_sum = [0.0] * NUM_ACTIONS
        self.strategy_sum = [0.0] * NUM_ACTIONS

    def current_strategy(self, reach_weight):
        """Regret matching: play actions in proportion to positive regret."""
        positive = [max(r, 0.0) for r in self.regret_sum]
        total = sum(positive)
        strategy = (
            [p / total for p in positive] if total > 0 else [1.0 / NUM_ACTIONS] * NUM_ACTIONS
        )
        for a in range(NUM_ACTIONS):
            self.strategy_sum[a] += reach_weight * strategy[a]
        return strategy

    def average_strategy(self):
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / NUM_ACTIONS] * NUM_ACTIONS


def cfr(nodes, cards, history, p1_reach, p2_reach):
    """Returns the expected utility for the player to act at this node."""
    player = len(history) % 2

    payoff = p1_payoff(cards, history)
    if payoff is not None:
        return payoff if player == 0 else -payoff

    key = infoset_key(cards, history)
    node = nodes.setdefault(key, Node())

    own_reach = p1_reach if player == 0 else p2_reach
    opp_reach = p2_reach if player == 0 else p1_reach
    strategy = node.current_strategy(own_reach)

    action_util = [0.0] * NUM_ACTIONS
    node_util = 0.0
    for a in range(NUM_ACTIONS):
        next_history = history + ("p" if a == 0 else "b")
        if player == 0:
            action_util[a] = -cfr(nodes, cards, next_history, p1_reach * strategy[a], p2_reach)
        else:
            action_util[a] = -cfr(nodes, cards, next_history, p1_reach, p2_reach * strategy[a])
        node_util += strategy[a] * action_util[a]

    for a in range(NUM_ACTIONS):
        node.regret_sum[a] += opp_reach * (action_util[a] - node_util)

    return node_util


# --------------------------------------------------------------------------
# Exact evaluation and exploitability
# --------------------------------------------------------------------------
def _walk(cards, history, sigma):
    payoff = p1_payoff(cards, history)
    if payoff is not None:
        return payoff
    p_bet = sigma[infoset_key(cards, history)]
    return (1 - p_bet) * _walk(cards, history + "p", sigma) + p_bet * _walk(
        cards, history + "b", sigma
    )


def p1_expected_value(sigma):
    """Exact expected payoff to Player 1. sigma maps infoset -> P(bet)."""
    deals = list(permutations(range(3), 2))
    return sum(_walk(d, "", sigma) for d in deals) / len(deals)


def best_response_values(sigma):
    """
    Exact best-response values against a fixed strategy profile, found by
    enumerating every pure strategy (2^6 = 64 per player).

    Returns (br1, br2): the best expected payoff Player 1 can get against
    sigma's Player-2 strategy, and the best Player 2 can get against
    sigma's Player-1 strategy (from Player 2's own perspective).
    """
    br1 = float("-inf")
    for bits in product([0.0, 1.0], repeat=len(P1_INFOSETS)):
        trial = dict(sigma)
        trial.update(dict(zip(P1_INFOSETS, bits)))
        br1 = max(br1, p1_expected_value(trial))

    br2 = float("-inf")
    for bits in product([0.0, 1.0], repeat=len(P2_INFOSETS)):
        trial = dict(sigma)
        trial.update(dict(zip(P2_INFOSETS, bits)))
        br2 = max(br2, -p1_expected_value(trial))

    return br1, br2


def exploitability(sigma):
    """
    NashConv-style exploitability: (BR1 + BR2) / 2.
    Zero exactly at a Nash equilibrium of this zero-sum game.
    """
    br1, br2 = best_response_values(sigma)
    return (br1 + br2) / 2.0


def average_sigma(nodes):
    """Average strategy as infoset -> P(bet)."""
    return {k: nodes[k].average_strategy()[1] for k in ALL_INFOSETS}


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
def train(iterations, log_every):
    nodes = {}
    deals = list(permutations(range(3), 2))
    convergence = []
    running_util = 0.0

    for i in range(1, iterations + 1):
        for cards in deals:
            running_util += cfr(nodes, cards, "", 1.0, 1.0)

        if i % log_every == 0 or i == iterations:
            sigma = average_sigma(nodes)
            convergence.append(
                {
                    "iteration": i,
                    "exploitability": exploitability(sigma),
                    "p1_expected_value": p1_expected_value(sigma),
                    "training_avg_game_value": running_util / (i * len(deals)),
                }
            )
    return nodes, convergence


# --------------------------------------------------------------------------
# Validation against the known analytical solution
# --------------------------------------------------------------------------
def validate(sigma, convergence, tol=0.02):
    """
    Known equilibrium family (alpha in [0, 1/3]):
      P1:  J bets alpha | Q checks, calls a bet w.p. alpha + 1/3 | K bets 3*alpha
      P2:  after bet:  J folds, Q calls 1/3, K calls
           after check: J bets 1/3, Q checks, K bets
    Game value for P1 = -1/18.
    """
    alpha = sigma["J"]
    checks = []

    def add(name, got, expected):
        ok = abs(got - expected) <= tol
        checks.append({"check": name, "got": round(got, 4), "expected": round(expected, 4), "pass": ok})

    final = convergence[-1]
    add("Game value for P1 (-1/18)", final["p1_expected_value"], -1 / 18)
    add("Exploitability (~0)", final["exploitability"], 0.0)
    checks.append(
        {
            "check": "P1 bluff freq with J (alpha) in [0, 1/3]",
            "got": round(alpha, 4),
            "expected": "0 to 0.3333",
            "pass": -tol <= alpha <= 1 / 3 + tol,
        }
    )
    add("P1 bets K with prob 3*alpha", sigma["K"], 3 * alpha)
    add("P1 bets Q first move (0)", sigma["Q"], 0.0)
    add("P1 calls with Q after check-bet (alpha + 1/3)", sigma["Qpb"], alpha + 1 / 3)
    add("P1 calls with K after check-bet (1)", sigma["Kpb"], 1.0)
    add("P1 calls with J after check-bet (0)", sigma["Jpb"], 0.0)
    add("P2 calls with J facing bet (0)", sigma["Jb"], 0.0)
    add("P2 calls with Q facing bet (1/3)", sigma["Qb"], 1 / 3)
    add("P2 calls with K facing bet (1)", sigma["Kb"], 1.0)
    add("P2 bluffs with J after check (1/3)", sigma["Jp"], 1 / 3)
    add("P2 bets Q after check (0)", sigma["Qp"], 0.0)
    add("P2 bets K after check (1)", sigma["Kp"], 1.0)
    return checks


def describe(key):
    card, history = key[0], key[1:]
    player = "P1" if key in P1_INFOSETS else "P2"
    situations = {
        "": "first to act",
        "p": "after P1 checks",
        "b": "facing P1's bet",
        "pb": "after check, facing P2's bet",
    }
    return player, f"{card}, {situations[history]}"


def main():
    parser = argparse.ArgumentParser(description="Kuhn Poker CFR solver")
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nodes, convergence = train(args.iterations, args.log_every)
    sigma = average_sigma(nodes)

    with open(os.path.join(args.out, "strategy_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["infoset", "player", "situation", "p_pass", "p_bet"])
        for key in sorted(ALL_INFOSETS, key=lambda k: (k not in P1_INFOSETS, len(k), k)):
            player, situation = describe(key)
            p_bet = sigma[key]
            w.writerow([key, player, situation, round(1 - p_bet, 4), round(p_bet, 4)])

    with open(os.path.join(args.out, "convergence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(convergence[0].keys()))
        w.writeheader()
        w.writerows(convergence)

    checks = validate(sigma, convergence)
    summary = {
        "iterations": args.iterations,
        "final": convergence[-1],
        "all_checks_passed": all(c["pass"] for c in checks),
        "checks": checks,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Iterations: {args.iterations}")
    print(f"P1 expected value: {convergence[-1]['p1_expected_value']:.5f}  (theory: {-1/18:.5f})")
    print(f"Exploitability:    {convergence[-1]['exploitability']:.6f}  (theory: 0)")
    print("\nAverage strategy (probability of betting/calling):")
    for key in sorted(ALL_INFOSETS, key=lambda k: (k not in P1_INFOSETS, len(k), k)):
        player, situation = describe(key)
        print(f"  {player} {situation:<32} {sigma[key]:.4f}")
    print("\nValidation:")
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: got {c['got']}, expected {c['expected']}")
    print(f"\nAll checks passed: {summary['all_checks_passed']}")


if __name__ == "__main__":
    main()
