"""
Poker & Game Theory Lab: an interactive dashboard for Kuhn Poker.

Run:  streamlit run app.py
Needs kuhn_cfr.py and kuhn_simulation.py in the same folder. The saved
results in ./outputs are used if present (otherwise the solver retrains).

Pages
  1. Game & Rules          rules, payoffs, game tree, information sets
  2. Equilibrium Solver    run CFR live, watch exploitability fall to zero
  3. Strategy Explorer     the equilibrium family, indifference principle
  4. Opponent Lab          build an opponent, see the exact best response
  5. Simulation            Monte Carlo play: equilibrium vs exploiter
  6. Deviation Tests       what deviating from equilibrium costs
  7. Play vs the AI        play Kuhn Poker yourself, with optional hints
"""

import os
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kuhn_cfr import (
    ALL_INFOSETS,
    CARD_NAMES,
    P1_INFOSETS,
    P2_INFOSETS,
    average_sigma,
    exploitability,
    p1_expected_value,
    p1_payoff,
    train,
    validate,
)
from kuhn_simulation import (
    ARCHETYPES,
    deviation_sweeps,
    exploiter_against,
    load_equilibrium,
    merge,
    seat_averaged_ev,
    simulate,
    theory_p1,
    theory_p2,
)

st.set_page_config(page_title="Poker & Game Theory Lab", page_icon="♠️", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
GAME_VALUE = -1 / 18

BLUE = "#2f6fdd"
ORANGE = "#e8743b"
GREY = "#8a8a8a"
GREEN = "#2e9e5b"

CARD_WORD = {"J": "Jack", "Q": "Queen", "K": "King"}
SITUATION = {
    "": "first move",
    "p": "after P1 checks",
    "b": "facing P1's bet",
    "pb": "check, then facing P2's bet",
}
SITUATION_SHORT = {"": "1st move", "p": "after check", "b": "vs bet", "pb": "check→vs bet"}
TERMINAL = {"pp", "bp", "bb", "pbp", "pbb"}


# ---------------------------------------------------------------------------
# Helpers: labels, strategy tables, cached computations
# ---------------------------------------------------------------------------
def player_of(key):
    return "P1" if key in P1_INFOSETS else "P2"


def action_labels(key):
    """(first action, second action) names at an information set."""
    return ("Fold", "Call") if key[1:].endswith("b") else ("Check", "Bet")


def infoset_label(key):
    return f"{player_of(key)} {key[0]} {SITUATION_SHORT[key[1:]]} ({action_labels(key)[1].lower()})"


def sort_keys(keys):
    return sorted(keys, key=lambda k: (k not in P1_INFOSETS, len(k), "JQK".index(k[0])))


def sigma_to_key(sigma):
    return tuple(round(float(sigma[k]), 6) for k in ALL_INFOSETS)


def key_to_sigma(key):
    return dict(zip(ALL_INFOSETS, key))


@st.cache_data(show_spinner=False)
def run_cfr(iterations, log_every):
    nodes, convergence = train(iterations, log_every)
    return average_sigma(nodes), convergence


@st.cache_data(show_spinner="Loading the equilibrium strategy…")
def get_equilibrium():
    """The CFR strategy saved by kuhn_cfr.py (retrained if the file is missing)."""
    return load_equilibrium(os.path.join(OUT, "strategy_table.csv"))


@st.cache_data(show_spinner=False)
def cached_exploiter(opp_key):
    return exploiter_against(key_to_sigma(opp_key))


@st.cache_data(show_spinner=False)
def cached_sweeps(grid_size):
    return pd.DataFrame(deviation_sweeps(grid_size))


@st.cache_data(show_spinner=False)
def cached_overview():
    """Exact seat-averaged EV for the equilibrium and an exploiter vs each archetype."""
    eq = get_equilibrium()
    rows = []
    for name, opp in ARCHETYPES.items():
        _, ex_ev = exploiter_against(opp)
        rows.append(
            {"Opponent": name, "Equilibrium": seat_averaged_ev(eq, opp)[0], "Exploiter": ex_ev}
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cached_simulation(opp_name, hands, seed):
    rng = random.Random(seed)
    eq = get_equilibrium()
    opp = ARCHETYPES[opp_name]
    exploiter, ex_exact = exploiter_against(opp)
    eq_exact = seat_averaged_ev(eq, opp)[0]
    checkpoint = max(hands // 200, 1)
    eq_mc, eq_curve = simulate(eq, opp, hands, rng, checkpoint)
    ex_mc, ex_curve = simulate(exploiter, opp, hands, rng, checkpoint)
    return {
        "eq_exact": eq_exact, "ex_exact": ex_exact,
        "eq_mc": eq_mc, "ex_mc": ex_mc,
        "eq_curve": eq_curve, "ex_curve": ex_curve,
    }


def strategy_frame(sigma):
    rows = []
    for key in sort_keys(ALL_INFOSETS):
        a0, a1 = action_labels(key)
        rows.append(
            {
                "Player": player_of(key),
                "Hand": CARD_WORD[key[0]],
                "Situation": SITUATION[key[1:]],
                f"Actions": f"{a0} / {a1}",
                "P(2nd action)": round(sigma[key], 4),
            }
        )
    return pd.DataFrame(rows)


def strategy_heatmap(sigma, player):
    if player == "P1":
        cols = [("", "First move: bet"), ("pb", "Check → facing bet: call")]
    else:
        cols = [("p", "After P1 checks: bet"), ("b", "Facing P1's bet: call")]
    z = [[sigma[f"{c}{h}"] for h, _ in cols] for c in CARD_NAMES]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=[label for _, label in cols], y=[CARD_WORD[c] for c in CARD_NAMES],
            zmin=0, zmax=1, colorscale="Blues", showscale=False,
            text=[[f"{v:.0%}" for v in row] for row in z], texttemplate="%{text}",
            textfont={"size": 16},
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=f"Player {player[1]}: probability of the aggressive action",
        height=270, margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


# ---------------------------------------------------------------------------
# Exact action values at an information set (used for indifference and hints)
# ---------------------------------------------------------------------------
def _tail_value(c, seat, weights, history, sigma, optimise):
    """
    Mass-weighted expected payoff to `seat` from `history` onward. `seat` holds
    card c; `weights` maps each possible opponent card to its probability mass.
    If `optimise`, the seat's own later decisions are chosen to maximise value
    (one decision per information set, since all opponent cards share it).
    """
    def cards(o):
        return (c, o) if seat == 0 else (o, c)

    if history in TERMINAL:
        total = 0.0
        for o, w in weights.items():
            pay = p1_payoff(cards(o), history)
            total += w * (pay if seat == 0 else -pay)
        return total

    if len(history) % 2 == seat:
        v_pass = _tail_value(c, seat, weights, history + "p", sigma, optimise)
        v_bet = _tail_value(c, seat, weights, history + "b", sigma, optimise)
        if optimise:
            return max(v_pass, v_bet)
        p = sigma[CARD_NAMES[c] + history]
        return (1 - p) * v_pass + p * v_bet

    w_pass = {o: w * (1 - sigma[CARD_NAMES[o] + history]) for o, w in weights.items()}
    w_bet = {o: w * sigma[CARD_NAMES[o] + history] for o, w in weights.items()}
    return _tail_value(c, seat, w_pass, history + "p", sigma, optimise) + _tail_value(
        c, seat, w_bet, history + "b", sigma, optimise
    )


def action_values(sigma, key, optimise=False):
    """
    Expected payoff (in chips, net of the ante) of each action at information
    set `key`, given the opponent's strategy in `sigma`. Returns
    (value of first action, value of second action, reachable).
    """
    c = CARD_NAMES.index(key[0])
    history = key[1:]
    seat = 0 if key in P1_INFOSETS else 1

    weights = {o: 1.0 for o in range(3) if o != c}
    for t, ch in enumerate(history):
        if t % 2 != seat:                       # an opponent action already taken
            for o in weights:
                p = sigma[CARD_NAMES[o] + history[:t]]
                weights[o] *= p if ch == "b" else 1 - p
    total = sum(weights.values())
    reachable = total > 1e-12
    weights = {o: (w / total if reachable else 0.5) for o, w in weights.items()}

    v_pass = _tail_value(c, seat, weights, history + "p", sigma, optimise)
    v_bet = _tail_value(c, seat, weights, history + "b", sigma, optimise)
    return v_pass, v_bet, reachable


# ---------------------------------------------------------------------------
# Page 1: Game & Rules
# ---------------------------------------------------------------------------
GAME_TREE = r"""
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.5;
  node [shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#4a6fa5",
        fontname="Helvetica", fontcolor="#1a1a1a", fontsize=15];
  edge [fontname="Helvetica", color="#888888", fontcolor="#888888", fontsize=14];
  p1  [label="P1 acts\n(sees own card)"];
  p2c [label="P2 acts\n(P1 checked)"];
  p2b [label="P2 acts\n(P1 bet)"];
  p1r [label="P1 acts\n(P2 bet after check)"];
  node [shape=ellipse, fillcolor="#fdf1e6", color="#c98a52"];
  t1 [label="Showdown\nwinner +1"];
  t2 [label="P1 folds\nP2 +1"];
  t3 [label="Showdown\nwinner +2"];
  t4 [label="P2 folds\nP1 +1"];
  t5 [label="Showdown\nwinner +2"];
  p1  -> p2c [label="check"];  p1  -> p2b [label="bet"];
  p2c -> t1  [label="check"];  p2c -> p1r [label="bet"];
  p1r -> t2  [label="fold"];   p1r -> t3  [label="call"];
  p2b -> t4  [label="fold"];   p2b -> t5  [label="call"];
}
"""


def page_game():
    st.title("♠ Poker & Game Theory Lab")
    st.markdown(
        "An interactive study of **Kuhn Poker**, the smallest poker game that still has "
        "bluffing, and a classic way to see game theory at work. The solver, simulation "
        "and every chart in this app are computed live from the Python engine."
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("The rules")
        st.markdown(
            """
- **Deck:** three cards, Jack < Queen < King. Each player is dealt one card privately.
- **Ante:** both players put 1 chip into the pot.
- **Player 1 acts first:** *check* or *bet 1*.
- **If P1 checks:** P2 can *check* (showdown) or *bet 1*. If P2 bets, P1 must *fold* or *call*.
- **If P1 bets:** P2 must *fold* or *call*.
- **Showdown:** the higher card wins the pot.
- It is **zero-sum**: what one player wins, the other loses.
            """
        )
    with right:
        st.subheader("Payoffs to Player 1")
        st.dataframe(
            pd.DataFrame(
                [
                    ["check, check", "Showdown for the antes", "+1 or −1"],
                    ["check, bet, fold", "P1 gives up", "−1"],
                    ["check, bet, call", "Showdown with a bet each", "+2 or −2"],
                    ["bet, fold", "P2 gives up", "+1"],
                    ["bet, call", "Showdown with a bet each", "+2 or −2"],
                ],
                columns=["Action history", "What happens", "P1 payoff"],
            ),
            hide_index=True,
        )

    st.subheader("The game tree (one deal)")
    st.graphviz_chart(GAME_TREE, width="stretch")
    st.caption(
        "Players choose between two actions at each decision. Neither player sees the "
        "other's card, which is what makes this a game of imperfect information."
    )

    st.subheader("Information sets")
    st.markdown(
        "A player can't tell apart the situations that look identical to them. An "
        "**information set** is *(your card, the actions so far)*. Each player has 6: "
        "three cards times two decision points. A **strategy** gives a probability of "
        "betting (or calling) at each one, so a full strategy profile is 12 numbers, "
        "and those 12 numbers are what the solver finds."
    )
    st.dataframe(
        pd.DataFrame(
            [
                ["Player 1", "Jack / Queen / King", "First move", "check or bet"],
                ["Player 1", "Jack / Queen / King", "After check, P2 bets", "fold or call"],
                ["Player 2", "Jack / Queen / King", "After P1 checks", "check or bet"],
                ["Player 2", "Jack / Queen / King", "After P1 bets", "fold or call"],
            ],
            columns=["Player", "Own card", "Decision point", "Choice"],
        ),
        hide_index=True,
    )

    st.subheader("Where each game theory concept appears")
    st.dataframe(
        pd.DataFrame(
            [
                ["Extensive form, imperfect information", "1. Game & Rules", "Game tree and information sets"],
                ["Mixed strategy Nash equilibrium", "2. Equilibrium Solver", "CFR converges to an unexploitable strategy"],
                ["Indifference principle", "3. Strategy Explorer", "Mixed actions have equal expected value"],
                ["Best response / exploitation", "4. Opponent Lab", "Exact counter-strategy to any opponent"],
                ["Minimax theorem, law of large numbers", "5. Simulation", "Equilibrium never loses; results converge to exact EV"],
                ["Why equilibrium is unexploitable", "6. Deviation Tests", "Any deviation lowers your guaranteed value"],
                ["Applying the theory", "7. Play vs the AI", "Play the game, with optional EV hints"],
            ],
            columns=["Concept", "Page", "What you see"],
        ),
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Page 2: Equilibrium Solver
# ---------------------------------------------------------------------------
def page_solver():
    st.title("Equilibrium Solver")
    st.markdown(
        "**Counterfactual Regret Minimization (CFR)** has the program play itself repeatedly. "
        "At every information set it tracks how much better each action *would have* done "
        "(its *regret*) and shifts toward the actions it regrets not taking. The **average** "
        "strategy over all iterations converges to a Nash equilibrium."
    )

    with st.form("cfr_form"):
        iters_in = st.slider(
            "CFR iterations", 1_000, 100_000, st.session_state.get("cfr_iters", 20_000), step=1_000
        )
        submitted = st.form_submit_button("Run solver")
    if submitted:
        st.session_state.cfr_iters = iters_in
    iters = st.session_state.get("cfr_iters", 20_000)

    with st.spinner(f"Running {iters:,} iterations…"):
        sigma, conv = run_cfr(iters, max(iters // 100, 10))
    final = conv[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("P1 game value", f"{final['p1_expected_value']:.4f}", f"theory {GAME_VALUE:.4f}", delta_color="off", delta_arrow="off")
    c2.metric("Exploitability", f"{final['exploitability']:.5f}", "theory 0", delta_color="off", delta_arrow="off")
    c3.metric("P1 Jack bluff frequency (α)", f"{sigma['J']:.3f}", "any value in 0 to 1/3", delta_color="off", delta_arrow="off")

    df = pd.DataFrame(conv)
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Scatter(x=df["iteration"], y=df["exploitability"], mode="lines", line={"color": BLUE, "width": 2.5}))
        fig.update_yaxes(type="log", title="Exploitability (log scale)")
        fig.update_xaxes(title="Iterations")
        fig.update_layout(title="Exploitability falls toward zero", height=360, margin={"t": 50})
        st.plotly_chart(fig)
        st.caption(
            "Exploitability is how much a perfect opponent could win per hand against the "
            "current strategy. It is exactly 0 at a Nash equilibrium."
        )
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["iteration"], y=df["p1_expected_value"], mode="lines", name="P1 value of average strategy", line={"color": BLUE, "width": 2.5}))
        fig.add_hline(y=GAME_VALUE, line_dash="dash", line_color=ORANGE, annotation_text="theory: −1/18", annotation_position="top right")
        fig.update_xaxes(title="Iterations")
        fig.update_yaxes(title="Expected payoff to Player 1 per hand")
        fig.update_layout(title="Game value converges to −1/18", height=360, margin={"t": 50}, showlegend=False)
        st.plotly_chart(fig)
        st.caption("Player 1 loses about 1/18 of a chip per hand: acting second is a built-in advantage.")

    st.subheader("The learned strategy")
    t1, t2 = st.columns(2)
    t1.plotly_chart(strategy_heatmap(sigma, "P1"))
    t2.plotly_chart(strategy_heatmap(sigma, "P2"))

    with st.expander("Validation against the known analytical solution", expanded=True):
        checks = validate(sigma, conv)
        passed = sum(c["pass"] for c in checks)
        (st.success if passed == len(checks) else st.warning)(f"{passed} of {len(checks)} checks pass")
        st.dataframe(
            pd.DataFrame(
                [{"Check": c["check"], "Computed": str(c["got"]), "Expected": str(c["expected"]), "Pass": "✅" if c["pass"] else "❌"} for c in checks]
            ),
            hide_index=True,
        )
        st.caption("Tolerance is 0.02. With few iterations some checks may fail; raise the iteration count to tighten them.")

    with st.expander("Full strategy table"):
        st.dataframe(strategy_frame(sigma), hide_index=True)


# ---------------------------------------------------------------------------
# Page 3: Strategy Explorer
# ---------------------------------------------------------------------------
def page_explorer():
    st.title("Strategy Explorer")
    st.markdown(
        "Kuhn Poker's equilibrium is not a single strategy but a **family**: Player 2's strategy "
        "is unique, while Player 1 has a free parameter **α**, the frequency of bluffing with the Jack."
    )

    source = st.radio(
        "Strategy to inspect",
        ["Theoretical equilibrium (choose α)", "Learned by CFR (saved 100,000-iteration run)"],
        horizontal=True,
    )
    if source.startswith("Theoretical"):
        k = st.slider("Player 1 Jack bluff frequency α", 0, 20, 10, help="α = k/60, so the range runs from 0 to 1/3")
        alpha = k / 60
        sigma = merge(theory_p1(alpha), theory_p2())   # P1's keys from the first, P2's from the second
        st.markdown(
            f"**α = {alpha:.3f}** → P1 bets the King **{min(1.0, 3 * alpha):.3f}** (= 3α), checks the Queen, "
            f"and calls with the Queen after a check-bet **{alpha + 1/3:.3f}** (= α + 1/3). "
            "P2's frequencies don't depend on α."
        )
    else:
        sigma = get_equilibrium()

    c1, c2, c3 = st.columns(3)
    c1.metric("P1 game value", f"{p1_expected_value(sigma):.4f}", f"theory {GAME_VALUE:.4f}", delta_color="off", delta_arrow="off")
    c2.metric("Exploitability", f"{exploitability(sigma):.5f}", "0 at equilibrium", delta_color="off", delta_arrow="off")
    c3.metric("P1 Jack bluff α", f"{sigma['J']:.3f}")

    h1, h2 = st.columns(2)
    h1.plotly_chart(strategy_heatmap(sigma, "P1"))
    h2.plotly_chart(strategy_heatmap(sigma, "P2"))

    st.subheader("The indifference principle")
    st.markdown(
        "A player only mixes between two actions if they are **equally good**; otherwise they'd "
        "always take the better one. This table gives the exact expected value of both actions at every "
        "information set, given the opponent's strategy. Where the strategy mixes, the two values match."
    )
    rows = []
    for key in sort_keys(ALL_INFOSETS):
        v0, v1, reachable = action_values(sigma, key)
        a0, a1 = action_labels(key)
        p = sigma[key]
        mixed = 0.02 < p < 0.98
        rows.append(
            {
                "Player": player_of(key),
                "Hand": CARD_WORD[key[0]],
                "Situation": SITUATION[key[1:]],
                "Actions": f"{a0} / {a1}",
                "P(2nd action)": round(p, 3),
                "EV of 1st": round(v0, 4),
                "EV of 2nd": round(v1, 4),
                "Gap": round(abs(v0 - v1), 4),
                "Type": ("Mixed: indifferent" if mixed else "Pure") if reachable else "Not reached",
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True)
    st.caption(
        "Pure rows: the chosen action has the higher (or equal) EV. Mixed rows: the gap is zero for the "
        "theoretical equilibrium and close to zero for the CFR approximation."
    )


# ---------------------------------------------------------------------------
# Page 4: Opponent Lab
# ---------------------------------------------------------------------------
LAB_PRESETS = list(ARCHETYPES) + ["Equilibrium (CFR)", "Custom"]


def _lab_apply_preset():
    name = st.session_state.lab_preset
    if name == "Custom":
        return
    src = get_equilibrium() if name == "Equilibrium (CFR)" else ARCHETYPES[name]
    for key in ALL_INFOSETS:
        st.session_state[f"lab_{key}"] = round(float(src[key]), 2)


def _lab_set_custom():
    st.session_state.lab_preset = "Custom"


def _lab_slider(key, label):
    st.slider(label, 0.0, 1.0, step=0.01, key=f"lab_{key}", on_change=_lab_set_custom)


def page_lab():
    st.title("Opponent Lab")
    st.markdown(
        "Build an opponent by setting how often they bet or call in every situation. The app then "
        "computes the **exact best response**: the counter-strategy that earns the most against them. "
        "It is compared with the equilibrium strategy, which doesn't adapt."
    )

    if "lab_J" not in st.session_state:
        st.session_state.lab_preset = "Calling station"
        _lab_apply_preset()

    st.selectbox("Load a preset opponent", LAB_PRESETS, key="lab_preset", on_change=_lab_apply_preset)

    with st.expander("Edit the opponent's strategy (probability of betting or calling)", expanded=False):
        a, b = st.columns(2)
        with a:
            st.markdown("**When the opponent is Player 1**")
            for c in CARD_NAMES:
                _lab_slider(c, f"Bet with {CARD_WORD[c]} (first move)")
            for c in CARD_NAMES:
                _lab_slider(f"{c}pb", f"Call with {CARD_WORD[c]} (after check, facing a bet)")
        with b:
            st.markdown("**When the opponent is Player 2**")
            for c in CARD_NAMES:
                _lab_slider(f"{c}p", f"Bet with {CARD_WORD[c]} (after P1 checks)")
            for c in CARD_NAMES:
                _lab_slider(f"{c}b", f"Call with {CARD_WORD[c]} (facing P1's bet)")

    opp = {k: float(st.session_state[f"lab_{k}"]) for k in ALL_INFOSETS}
    eq = get_equilibrium()
    _, ex_ev = cached_exploiter(sigma_to_key(opp))
    eq_ev = seat_averaged_ev(eq, opp)[0]

    # Best response, defined at EVERY situation: take the higher-EV action given the opponent's play.
    # (A brute-force search only pins down situations the exploiter itself can reach.) On an exact
    # tie any action is a best response, so the equilibrium's choice is shown.
    exploiter, gaps = {}, {}
    for k in ALL_INFOSETS:
        v0, v1, _ = action_values(opp, k, optimise=True)
        gaps[k] = abs(v0 - v1)
        exploiter[k] = 1.0 if v1 > v0 + 1e-9 else 0.0 if v0 > v1 + 1e-9 else eq[k]

    st.subheader("Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Equilibrium player's EV / hand", f"{eq_ev:+.4f}")
    m2.metric("Best-response player's EV / hand", f"{ex_ev:+.4f}", help="This is also the opponent's exploitability: what a perfect opponent wins per hand against them.")
    m3.metric("Extra profit from adapting", f"{ex_ev - eq_ev:+.4f}")
    st.caption("EVs are in chips per hand, averaged over both seats (so neither side gets the second-mover advantage).")

    keys = sort_keys(ALL_INFOSETS)
    labels = [infoset_label(k) for k in keys]
    fig = go.Figure()
    fig.add_bar(name="Opponent", x=labels, y=[opp[k] for k in keys], marker_color=GREY)
    fig.add_bar(name="Equilibrium", x=labels, y=[eq[k] for k in keys], marker_color=BLUE)
    fig.add_bar(name="Best response", x=labels, y=[exploiter[k] for k in keys], marker_color=ORANGE)
    fig.update_layout(
        barmode="group", height=440, margin={"t": 30},
        yaxis={"title": "Probability of betting / calling", "range": [0, 1.05]},
        legend={"orientation": "h", "y": 1.08},
    )
    st.plotly_chart(fig)

    st.subheader("How the best response differs from equilibrium")
    notes = []
    for k in keys:
        if abs(exploiter[k] - eq[k]) >= 0.5 and gaps[k] > 0.02:   # the choice really matters here
            notes.append(
                f"- **{infoset_label(k)}**: equilibrium {eq[k]:.0%} → best response **{exploiter[k]:.0%}** "
                f"(worth {gaps[k]:.2f} chips per time you're in this spot)"
            )
    if notes:
        st.markdown("\n".join(notes))
    else:
        st.info("The best response is essentially the equilibrium strategy: this opponent has no big leaks to exploit.")
    st.caption(
        "Only situations where the choice clearly matters are listed. Where two actions earn exactly the same "
        "against this opponent, the chart shows the equilibrium's choice, since any action is a best response there."
    )


# ---------------------------------------------------------------------------
# Page 5: Simulation
# ---------------------------------------------------------------------------
def page_sim():
    st.title("Simulation")
    st.markdown(
        "Monte Carlo play: real hands, dealt at random, with seats alternating every hand. It checks the "
        "exact math, and shows the **minimax guarantee** in action: the equilibrium player never loses in "
        "the long run, while a player who knows the opponent's leaks can win much more."
    )

    tab_run, tab_over, tab_full = st.tabs(["Run a matchup", "All opponents (exact)", "Saved 200,000-hand run"])

    with tab_run:
        c1, c2, c3 = st.columns(3)
        opp_name = c1.selectbox("Opponent", list(ARCHETYPES))
        hands = c2.slider("Hands to play", 2_000, 100_000, 20_000, step=2_000)
        seed = c3.number_input("Random seed", 0, 9999, 7)
        with st.spinner("Playing hands…"):
            res = cached_simulation(opp_name, int(hands), int(seed))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Equilibrium: simulated", f"{res['eq_mc']:+.4f}", f"exact {res['eq_exact']:+.4f}", delta_color="off", delta_arrow="off")
        m2.metric("Exploiter: simulated", f"{res['ex_mc']:+.4f}", f"exact {res['ex_exact']:+.4f}", delta_color="off", delta_arrow="off")
        m3.metric("Equilibrium total chips", f"{res['eq_curve'][-1][1]:+,.0f}")
        m4.metric("Exploiter total chips", f"{res['ex_curve'][-1][1]:+,.0f}")

        x = [p[0] for p in res["eq_curve"]]
        eq_y = [p[1] for p in res["eq_curve"]]
        ex_y = [p[1] for p in res["ex_curve"]]
        view = st.radio("Chart", ["Cumulative profit", "Running average (law of large numbers)"], horizontal=True)
        fig = go.Figure()
        if view == "Cumulative profit":
            fig.add_scatter(x=x, y=eq_y, name="Equilibrium", line={"color": BLUE, "width": 2.5})
            fig.add_scatter(x=x, y=ex_y, name="Exploiter", line={"color": ORANGE, "width": 2.5})
            fig.add_scatter(x=x, y=[res["eq_exact"] * h for h in x], name="Equilibrium: exact expectation", line={"color": BLUE, "dash": "dot", "width": 1.5})
            fig.add_scatter(x=x, y=[res["ex_exact"] * h for h in x], name="Exploiter: exact expectation", line={"color": ORANGE, "dash": "dot", "width": 1.5})
            fig.update_yaxes(title="Cumulative chips won")
        else:
            fig.add_scatter(x=x, y=[v / h for v, h in zip(eq_y, x)], name="Equilibrium", line={"color": BLUE, "width": 2.5})
            fig.add_scatter(x=x, y=[v / h for v, h in zip(ex_y, x)], name="Exploiter", line={"color": ORANGE, "width": 2.5})
            fig.add_hline(y=res["eq_exact"], line_dash="dot", line_color=BLUE)
            fig.add_hline(y=res["ex_exact"], line_dash="dot", line_color=ORANGE)
            fig.update_yaxes(title="Average chips per hand")
        fig.update_xaxes(title="Hands played")
        fig.update_layout(height=430, margin={"t": 30}, legend={"orientation": "h", "y": 1.1})
        st.plotly_chart(fig)
        st.caption(
            "Dotted lines are the exact expected values. Early results are noisy, but the running "
            "average settles onto them as more hands are played."
        )

    with tab_over:
        df = cached_overview()
        fig = go.Figure()
        fig.add_bar(name="Equilibrium", x=df["Opponent"], y=df["Equilibrium"], marker_color=BLUE)
        fig.add_bar(name="Exploiter (best response)", x=df["Opponent"], y=df["Exploiter"], marker_color=ORANGE)
        fig.update_layout(barmode="group", height=400, margin={"t": 30}, yaxis={"title": "Exact EV per hand (chips)"}, legend={"orientation": "h", "y": 1.1})
        st.plotly_chart(fig)
        st.markdown(
            "- The **equilibrium never loses**, whatever the opponent does.\n"
            "- An **exploiter** earns more against every opponent, but only because it knows their exact leaks.\n"
            "- Against the **over-folder** and **over-bluffer**, equilibrium earns about zero: their mistakes "
            "are real, but only a player who *adapts* can profit from them."
        )

    with tab_full:
        path = os.path.join(OUT, "matchups.csv")
        if os.path.exists(path):
            st.dataframe(pd.read_csv(path), hide_index=True)
            st.caption("Produced by kuhn_simulation.py: 200,000 hands per matchup, seats alternating.")
        else:
            st.info("Run kuhn_simulation.py to generate outputs/matchups.csv.")


# ---------------------------------------------------------------------------
# Page 6: Deviation Tests
# ---------------------------------------------------------------------------
SWEEP_INFO = {
    "P2 bluff freq with Jack after check (eq 1/3)": (
        "Player 2 bluffs with the Jack after a check",
        "Too little and Player 1 learns to fold the Queen less; too much and Player 1 calls and punishes the bluffs. "
        "Only 1/3 leaves Player 1 indifferent, so only 1/3 guarantees +1/18.",
    ),
    "P2 call freq with Queen facing bet (eq 1/3)": (
        "Player 2 calls with the Queen facing a bet",
        "Call too rarely and Player 1 bluffs profitably; call too often and Player 1 simply stops bluffing and bets "
        "only strong hands. Again 1/3 is the unique best choice.",
    ),
    "P1 bluff freq with Jack, alpha (eq range 0 to 1/3)": (
        "Player 1 bluffs with the Jack (α)",
        "Player 1 has a *range* of equally good choices: any α from 0 to 1/3 guarantees −1/18. "
        "Bluff more than 1/3 of the time and Player 2 can profit from calling.",
    ),
}


def page_dev():
    st.title("Deviation Tests")
    st.markdown(
        "Why is equilibrium play 'unexploitable'? Here one player changes a single frequency while "
        "every other frequency stays at equilibrium, and the opponent replies with a **perfect best response**. "
        "The chart shows the deviating player's **guaranteed** expected value per hand."
    )

    df = cached_sweeps(60)
    names = list(SWEEP_INFO)
    choice = st.radio("Which frequency deviates?", names, format_func=lambda n: SWEEP_INFO[n][0])
    sub = df[df["sweep"] == choice].reset_index(drop=True)
    title, explanation = SWEEP_INFO[choice]

    fig = go.Figure()
    fig.add_scatter(x=sub["frequency"], y=sub["guaranteed_ev_of_deviator"], mode="lines", line={"color": BLUE, "width": 3}, name="Guaranteed EV")
    if choice.startswith("P1"):
        fig.add_vrect(x0=0, x1=1 / 3, fillcolor=GREEN, opacity=0.12, line_width=0, annotation_text="equilibrium range", annotation_position="top left")
        ref = GAME_VALUE
    else:
        fig.add_vline(x=1 / 3, line_dash="dash", line_color=GREEN, annotation_text="equilibrium: 1/3", annotation_position="bottom right")
        ref = -GAME_VALUE
    fig.add_hline(y=ref, line_dash="dot", line_color=GREY, annotation_text=f"equilibrium value {ref:+.4f}", annotation_position="bottom right")
    fig.update_xaxes(title="Frequency (probability)")
    fig.update_yaxes(title="Guaranteed EV per hand (opponent best-responds)")
    fig.update_layout(height=430, margin={"t": 30}, showlegend=False, title=title)
    st.plotly_chart(fig)
    st.markdown(explanation)

    st.subheader("Try a deviation")
    k = st.slider("Frequency", 0, 60, 20, help="frequency = k/60")
    row = sub.iloc[k]
    loss = row["guaranteed_ev_of_deviator"] - ref
    if abs(loss) < 1e-9:
        loss = 0.0                                   # avoid showing "-0.0000"
    c1, c2, c3 = st.columns(3)
    c1.metric("Frequency you pick", f"{row['frequency']:.3f}")
    c2.metric("Guaranteed EV", f"{row['guaranteed_ev_of_deviator']:+.4f}")
    c3.metric("Compared with equilibrium", f"{loss:+.4f}", delta_color="off", delta_arrow="off")
    if loss < -1e-9:
        st.warning(f"A best-responding opponent takes {abs(loss):.4f} chips per hand off you, about {abs(loss) / abs(GAME_VALUE):.1f}× the size of the whole 1/18 edge.")
    else:
        st.success("No loss: this frequency is part of an equilibrium.")


# ---------------------------------------------------------------------------
# Page 7: Play vs the AI
# ---------------------------------------------------------------------------
BOT_CHOICES = ["Equilibrium (CFR)"] + list(ARCHETYPES)


def _bot_sigma(name):
    return get_equilibrium() if name == "Equilibrium (CFR)" else ARCHETYPES[name]


def _play_state():
    if "play" not in st.session_state:
        st.session_state.play = {
            "hand": 0, "profit": 0.0, "log": [], "cards": None, "history": "",
            "settled": True, "seat": 0, "bot": "Equilibrium (CFR)",
        }
    return st.session_state.play


def _deal():
    p = _play_state()
    p["hand"] += 1
    p["cards"] = tuple(random.sample([0, 1, 2], 2))
    p["history"] = ""
    p["settled"] = False
    p["bot"] = st.session_state.get("play_bot", "Equilibrium (CFR)")
    mode = st.session_state.get("play_seat", "Alternate each hand")
    p["seat"] = (p["hand"] + 1) % 2 if mode.startswith("Alternate") else (0 if mode.endswith("1") else 1)


def _act(action):
    p = _play_state()
    p["history"] += action


def _reset():
    st.session_state.pop("play", None)


def _bot_moves(p):
    sigma = _bot_sigma(p["bot"])
    while (
        p["cards"] is not None
        and p1_payoff(p["cards"], p["history"]) is None
        and len(p["history"]) % 2 != p["seat"]
    ):
        key = CARD_NAMES[p["cards"][1 - p["seat"]]] + p["history"]
        p["history"] += "b" if random.random() < sigma[key] else "p"


def _history_text(history, seat):
    parts = []
    for t, ch in enumerate(history):
        who = "You" if t % 2 == seat else "Bot"
        facing = t > 0 and history[t - 1] == "b"
        verb = ("folds", "calls")[ch == "b"] if facing else ("checks", "bets")[ch == "b"]
        parts.append(f"**{who}** {verb}")
    return " → ".join(parts) if parts else "*No actions yet*"


def page_play():
    st.title("Play vs the AI")
    st.markdown(
        "Play Kuhn Poker against the bot. Against the **equilibrium bot** you can't win in the long run "
        "(over alternating seats you expect to break even at best). Against a **leaky opponent** you can "
        "win by spotting and exploiting its habits, just like the Opponent Lab's best response."
    )

    p = _play_state()
    s1, s2, s3 = st.columns(3)
    s1.selectbox("Opponent bot", BOT_CHOICES, key="play_bot", help="Applies from the next hand")
    s2.radio("Your seat", ["Alternate each hand", "Always Player 1", "Always Player 2"], key="play_seat", help="Applies from the next hand")
    show_hint = s3.checkbox("Show EV hint (spoiler)", help="Exact expected value of each option against this bot, assuming you play the rest of the hand well")

    if p["cards"] is None:
        st.button("Deal first hand", on_click=_deal, type="primary")
    else:
        _bot_moves(p)
        terminal = p1_payoff(p["cards"], p["history"]) is not None
        me = p["seat"]
        my_card, bot_card = p["cards"][me], p["cards"][1 - me]

        pot = 2 + p["history"].count("b")
        st.subheader(f"Hand {p['hand']}: you are Player {me + 1}" + (" (you act first)" if me == 0 else " (bot acts first)"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Your card", CARD_WORD[CARD_NAMES[my_card]])
        c2.metric("Bot's card", CARD_WORD[CARD_NAMES[bot_card]] if terminal else "🂠 hidden")
        c3.metric("Pot", f"{pot} chips")
        st.markdown(_history_text(p["history"], me))

        if not terminal:
            facing = p["history"].endswith("b")
            a0, a1 = ("Fold", "Call") if facing else ("Check", "Bet 1 chip")
            b1, b2, _ = st.columns([1, 1, 3])
            b1.button(a0, on_click=_act, args=("p",), key="act_pass", width="stretch")
            b2.button(a1, on_click=_act, args=("b",), key="act_bet", width="stretch", type="primary")
            if show_hint:
                key = CARD_NAMES[my_card] + p["history"]
                v0, v1, _ = action_values(_bot_sigma(p["bot"]), key, optimise=True)
                better = a0 if v0 > v1 + 1e-9 else a1 if v1 > v0 + 1e-9 else "either (equal EV)"
                st.info(f"Hint: **{a0}** has EV {v0:+.3f} chips, **{a1}** has EV {v1:+.3f}. Best: {better}.")
        else:
            payoff = p1_payoff(p["cards"], p["history"])
            mine = payoff if me == 0 else -payoff
            if not p["settled"]:
                p["profit"] += mine
                p["log"].append(
                    {
                        "Hand": p["hand"], "Seat": f"P{me + 1}", "Bot": p["bot"],
                        "Your card": CARD_NAMES[my_card], "Bot card": CARD_NAMES[bot_card],
                        "Actions": p["history"].replace("p", "·").replace("b", "B"), "Result": mine,
                    }
                )
                p["settled"] = True
            folded = p["history"] in ("bp", "pbp")
            chips = f"{abs(mine)} chip{'s' if abs(mine) != 1 else ''}"
            if mine > 0:
                st.success(f"You won {chips}: " + ("the bot folded." if folded else "you had the higher card at showdown."))
            else:
                st.error(f"You lost {chips}: " + ("you folded." if folded else "the bot had the higher card at showdown."))
            st.button("Next hand", on_click=_deal, type="primary")

    if p["log"]:
        st.divider()
        n = len(p["log"])
        k1, k2, k3 = st.columns(3)
        k1.metric("Hands played", n)
        k2.metric("Total profit", f"{p['profit']:+.0f} chips")
        k3.metric("Average per hand", f"{p['profit'] / n:+.3f}")
        cum, run = [], 0
        for row in p["log"]:
            run += row["Result"]
            cum.append(run)
        fig = go.Figure(go.Scatter(x=list(range(1, n + 1)), y=cum, mode="lines+markers", line={"color": BLUE}))
        fig.add_hline(y=0, line_dash="dot", line_color=GREY)
        fig.update_layout(height=300, margin={"t": 30}, xaxis_title="Hand", yaxis_title="Your cumulative chips")
        st.plotly_chart(fig)
        with st.expander("Hand history"):
            st.dataframe(pd.DataFrame(p["log"]), hide_index=True)
            st.caption("Actions: · = check or fold, B = bet or call.")
    st.button("Reset session", on_click=_reset)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "1. Game & Rules": page_game,
    "2. Equilibrium Solver": page_solver,
    "3. Strategy Explorer": page_explorer,
    "4. Opponent Lab": page_lab,
    "5. Simulation": page_sim,
    "6. Deviation Tests": page_dev,
    "7. Play vs the AI": page_play,
}

with st.sidebar:
    st.markdown("### ♠ Poker & Game Theory Lab")
    page = st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")
    st.divider()
    st.caption(
        "Kuhn Poker solved with Counterfactual Regret Minimization. Game value for Player 1 is "
        "−1/18 chip per hand."
    )

PAGES[page]()
