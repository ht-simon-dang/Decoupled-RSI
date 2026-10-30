"""Render the four editable README figures with Matplotlib.

From the repository root:
    python docs/assets/decoupled-rsi/render_figures.py

Writes SVG (editable text) and PNG beside this script. Only Matplotlib is
required. The PFM counts are recomputed from the curated public example.
No model inference, installation, network access, or original run logs are used.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MPath
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parents[2]
INK = "#17243b"
MUTED = "#53627a"
LINE = "#dce3ed"
TEAL = "#087f8c"
TEAL_BG = "#edf8f8"
PURPLE = "#7045b8"
PURPLE_BG = "#f4effb"
AMBER = "#986016"
AMBER_BG = "#fff7e8"
GRAY_BG = "#f3f5f8"
GRAY = "#637189"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "svg.hashsalt": "decoupled-rsi-draft-v1",
    "font.size": 13,
})


def canvas(w, h, number, title, subtitle):
    fig = plt.figure(figsize=(w / 100, h / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(h, 0))
    ax.axis("off")
    text(ax, 60, 30, f"DECOUPLED RSI  /  FIGURE {number:02}", 11, PURPLE, weight="bold")
    text(ax, 60, 76, title, 26, INK, weight="bold")
    text(ax, 60, 116, subtitle, 13, MUTED)
    return fig, ax


def text(ax, x, y, s, size=14, color=INK, weight="normal", ha="left", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", zorder=6, **kw)


def panel(ax, x, y, w, h, label, color, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=18",
                              facecolor=fill, edgecolor="none", zorder=0))
    text(ax, x + 24, y + 30, label, 11, color, weight="bold")


def card(ax, x, y, w, h, title, lines=(), color=GRAY, fill="white",
         title_size=17, body_size=12.5, body_step=25):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=12",
                              facecolor=fill, edgecolor=LINE, linewidth=1.1, zorder=3))
    ax.plot([x + 1, x + 1], [y + 16, y + h - 16], color=color,
            linewidth=3.5, solid_capstyle="round", zorder=4)
    def fitted(s, size, weight="normal"):
        font = FontProperties(family="DejaVu Sans", weight=weight)
        width = TextPath((0, 0), s, size=size, prop=font).get_extents().width * 100 / 72
        return min(size, size * (w - 48) / max(width, 1))
    text(ax, x + 22, y + 30, title, fitted(title, title_size, "bold"), color, weight="bold")
    for i, line in enumerate(lines):
        text(ax, x + 22, y + 62 + i * body_step, line, fitted(line, body_size), MUTED)


def arrow(ax, points, color=GRAY, dashed=False, width=1.8):
    path = MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=15,
                                linewidth=width, color=color,
                                linestyle=(0, (4, 4)) if dashed else "-",
                                zorder=2))


def label(ax, x, y, s, color=MUTED, size=11, ha="center"):
    text(ax, x, y, s, size, color, ha=ha,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})


def footer(ax, h, note):
    ax.plot([60, 1540], [h - 69, h - 69], color=LINE, linewidth=1)
    text(ax, 60, h - 38, note, 11, MUTED)


def save(fig, stem):
    # Catch off-canvas labels before exporting. Visual inspection is still needed.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for obj in fig.findobj(matplotlib.text.Text):
        if not obj.get_text() or not obj.get_visible():
            continue
        box = obj.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1:
            raise ValueError(f"Text outside canvas in {stem}: {obj.get_text()}")
    fig.savefig(HERE / f"{stem}.svg", metadata={"Date": None, "Creator": "Decoupled RSI"})
    fig.savefig(HERE / f"{stem}.png", dpi=140)
    plt.close(fig)
    print(stem)


def architecture():
    fig, ax = canvas(1600, 1150, 1, "Give self-evolution its own agent",
                     "Separate task execution, execution evidence, improvement memory, and adoption.")
    panel(ax, 60, 165, 740, 315, "TASK EXECUTION  /  SELECTED VERSION", TEAL, TEAL_BG)
    panel(ax, 850, 165, 690, 610, "SELF-EVOLUTION  /  INDEPENDENT INVOCATION", PURPLE, PURPLE_BG)
    card(ax, 85, 230, 170, 105, "User goal", ["Objective", "Constraints"], TEAL,
         title_size=15, body_size=12, body_step=22)
    card(ax, 300, 225, 290, 120, "Master Agent",
         ["Plan, dispatch, inspect feedback", "Refine the current task"], TEAL, body_size=11.5)
    card(ax, 635, 230, 140, 105, "Task result", ["Output +", "evidence"], TEAL,
         title_size=13, body_size=12, body_step=22)
    card(ax, 300, 385, 290, 65, "Domain execution & tools", color=TEAL, title_size=14)
    arrow(ax, [(255, 282), (300, 282)], TEAL)
    arrow(ax, [(590, 282), (635, 282)], TEAL)
    arrow(ax, [(400, 345), (400, 385)], TEAL)
    arrow(ax, [(490, 385), (490, 345)], TEAL)
    card(ax, 125, 565, 605, 160, "Trace Database",
         ["Request + plan + tool events",
          "Results + errors + cost + active version",
          "Journal files → normalized episodes"], AMBER, AMBER_BG, title_size=20)
    arrow(ax, [(445, 450), (445, 565)], TEAL)
    label(ax, 445, 515, "execution evidence", TEAL)
    card(ax, 895, 230, 600, 145, "Self-Evolve Agent",
         ["Study evidence → form a hypothesis",
          "Build candidates within allowed editable surfaces"],
         PURPLE, title_size=21, body_size=13)
    arrow(ax, [(650, 565), (650, 515), (820, 515), (820, 300), (895, 300)], AMBER)
    label(ax, 730, 500, "selected experience", AMBER)
    card(ax, 895, 420, 600, 75, "Candidate version",
         ["Policy / prompt overlay / scoped code or decoder change"],
         PURPLE, title_size=16, body_size=11.7)
    arrow(ax, [(1195, 375), (1195, 420)], PURPLE)
    card(ax, 895, 545, 600, 165, "Evolution Memory",
         ["Hypotheses · candidates · parent links",
          "Evaluation outcomes · decisions · costs",
          "Retain unsuccessful attempts as well as gains"],
         AMBER, AMBER_BG, title_size=20, body_size=13)
    arrow(ax, [(1400, 495), (1400, 545)], AMBER)
    label(ax, 1430, 520, "record", AMBER)
    arrow(ax, [(1495, 600), (1520, 600), (1520, 280), (1495, 280)], AMBER)
    text(ax, 1520, 430, "reuse", 10, AMBER, ha="center", rotation=90,
         bbox={"facecolor": PURPLE_BG, "edgecolor": "none", "pad": 2})
    panel(ax, 60, 845, 1480, 180, "EVALUATION & ADOPTION", GRAY, GRAY_BG)
    card(ax, 825, 900, 380, 90, "Independent evaluation",
         ["Fixed rules · measured outcomes"], GRAY, title_size=17)
    arrow(ax, [(895, 458), (840, 458), (840, 810), (1015, 810), (1015, 900)], PURPLE)
    arrow(ax, [(1100, 900), (1100, 750), (1200, 750), (1200, 710)], AMBER)
    label(ax, 1255, 785, "evaluation evidence", AMBER)
    card(ax, 420, 900, 325, 90, "Adoption decision",
         ["Explicit policy / operator choice"], GRAY, title_size=17, body_size=11.8)
    arrow(ax, [(825, 945), (745, 945)], GRAY)
    card(ax, 90, 900, 250, 90, "Selected version",
         ["Used by subsequent tasks"], TEAL, title_size=16, body_size=11.5)
    arrow(ax, [(420, 945), (340, 945)], TEAL)
    arrow(ax, [(90, 945), (70, 945), (70, 350), (280, 350), (280, 320), (300, 320)], TEAL)
    text(ax, 70, 800, "adopted version", 10, TEAL, rotation=90, ha="center",
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    footer(ax, 1150, "Architecture view: the Drug implementation is primary; a shared interface across all four Task Agents remains future work.")
    save(fig, "01-architecture")


def memories():
    fig, ax = canvas(1600, 1030, 2, "Two memories, two kinds of experience",
                     "Execution traces explain task behavior. Evolution records explain improvement decisions.")
    panel(ax, 60, 175, 660, 510, "TRACE DATABASE", TEAL, TEAL_BG)
    panel(ax, 880, 175, 660, 510, "EVOLUTION MEMORY", PURPLE, PURPLE_BG)
    text(ax, 90, 255, "What happened during the task?", 20, TEAL, weight="bold")
    text(ax, 910, 255, "What did we learn from each attempt?", 18, PURPLE, weight="bold")
    left = [
        ("Unit", "A task execution / episode"),
        ("Intent", "Goal, constraints, plan"),
        ("Actions", "Tool calls and returned evidence"),
        ("Outcome", "Result, failure, cost, timing"),
        ("Identity", "Request and active-version references"),
        ("Use", "Reconstruct behavior; locate problems"),
    ]
    right = [
        ("Unit", "An improvement attempt / candidate"),
        ("Hypothesis", "Problem, rationale, proposed change"),
        ("Lineage", "Parent candidate and editable surface"),
        ("Evidence", "Evaluation result and consumed cost"),
        ("Decision", "Retain, reject, nominate, or adopt"),
        ("Use", "Track attempts; guide later choices"),
    ]
    for x, rows, color in [(90, left, TEAL), (910, right, PURPLE)]:
        for i, (key, value) in enumerate(rows):
            y = 320 + i * 50
            if i % 2 == 0:
                ax.add_patch(FancyBboxPatch((x, y - 20), 600, 42,
                                          boxstyle="round,pad=0,rounding_size=6",
                                          facecolor="white", edgecolor="none", zorder=1))
            text(ax, x + 14, y, key, 12, color, weight="bold")
            text(ax, x + 140, y, value, 11.6, INK)
    arrow(ax, [(720, 420), (880, 420)], PURPLE)
    label(ax, 800, 385, "analyze", PURPLE)
    label(ax, 800, 451, "and propose", PURPLE)
    text(ax, 90, 645, "Record task facts, including failures.", 12, TEAL)
    text(ax, 910, 645, "Keep negative and unchanged results, too.", 12, PURPLE)
    text(ax, 60, 755, "CONNECT RECORDS THROUGH EVIDENCE REFERENCES", 11, GRAY, weight="bold")
    chain = [
        (60, "Trace reference", TEAL),
        (360, "Hypothesis", PURPLE),
        (660, "Candidate", PURPLE),
        (960, "Evaluation", GRAY),
        (1260, "Decision", TEAL),
    ]
    for x, name, color in chain:
        card(ax, x, 800, 280, 65, name, color=color, title_size=16)
    for x in [340, 640, 940, 1240]:
        arrow(ax, [(x, 833), (x + 20, 833)], GRAY)
    text(ax, 60, 910, "Current stores: request journals + episode views; candidate archives + cycle logs + per-target experiment history.",
         12, MUTED)
    footer(ax, 1030, "Record categories and evidence links are schematic. The release does not yet expose one unified cross-domain Memory API.")
    save(fig, "02-two-memories")


def pfm_counts():
    data = json.loads((PUBLIC / "examples/pfm_replay/recorded_evaluations.json").read_text())
    change = json.loads((PUBLIC / "examples/pfm_replay/candidate_change.json").read_text())
    counts = {}
    for record in data["records"]:
        # Gate names and nested layout are read from the public replay's schema.
        passing = set()
        for row in record["rows"]:
            if all(row["gates"].values()):
                passing.add(row["scaffold_sha256"])
        counts[record["cycle"]] = len(passing)
    return change, [(str(pair["seed"]), counts[pair["stock"]], counts[pair["shaped"]])
                    for pair in change["paired_cycles"]]


def pfm():
    change, results = pfm_counts()
    constants = change["recorded_constants"]
    assert constants["weights_changed"] is False
    assert constants["optimiser_steps"] == 0
    assert constants["submission_budget"] == 150
    fig, ax = canvas(1600, 1120, 3, "A concrete intervention: evolve the PFM decoder",
                     "A recorded per-target experiment, with independent evaluation of the resulting submissions.")
    card(ax, 60, 165, 1480, 90, "Same recorded PFM checkpoint",
         ["0 optimizer steps · 150 submissions per arm · development partition"],
         GRAY, GRAY_BG, title_size=20, body_size=14)
    card(ax, 60, 310, 660, 170, "Stock decoder",
         ["As recorded in c04 and c06", "Original PFM sampling procedure"],
         TEAL, TEAL_BG, title_size=22, body_size=15)
    card(ax, 880, 310, 660, 170, "Shaped decoder",
         ["Size- and atom-type-biased prior",
          "Partial-structure rejection and resampling",
          "Checks at steps 15 and 17 of 20; up to 3 attempts"],
         PURPLE, PURPLE_BG, title_size=22, body_size=14)
    arrow(ax, [(390, 255), (390, 310)], TEAL)
    arrow(ax, [(1210, 255), (1210, 310)], PURPLE)
    card(ax, 450, 540, 700, 90, "Independent evaluator",
         ["Same recorded qualification rules · seven gates"], GRAY,
         title_size=20, body_size=14)
    arrow(ax, [(390, 480), (390, 585), (450, 585)], TEAL)
    arrow(ax, [(1210, 480), (1210, 585), (1150, 585)], PURPLE)
    panel(ax, 60, 690, 1480, 300, "RECORDED QUALIFYING SCAFFOLDS  /  OUT OF 150 SUBMISSIONS", GRAY, GRAY_BG)
    for x, c, s in [(1060, TEAL, "stock"), (1250, PURPLE, "shaped")]:
        ax.plot(x, 723, "o", color=c, markersize=7, zorder=5)
        text(ax, x + 18, 723, s, 12, c)
    x0, x1 = 460, 1320
    position = lambda n: x0 + (x1 - x0) * n / 150
    for tick in range(0, 151, 30):
        x = position(tick)
        ax.plot([x, x], [767, 914], color=LINE, linewidth=1, zorder=1)
        text(ax, x, 945, str(tick), 11, MUTED, ha="center")
    for y, (seed, stock, shaped) in zip([805, 875], results):
        text(ax, 90, y - 10, f"Seed {seed}", 15, INK, weight="bold")
        text(ax, 90, y + 17, "paired decoder comparison", 11, MUTED)
        ax.plot([position(stock), position(shaped)], [y, y], color=GRAY, linewidth=3, zorder=3)
        for number, c in [(stock, TEAL), (shaped, PURPLE)]:
            ax.plot(position(number), y, "o", color=c, markersize=9, zorder=5)
        # Small differences are visible in the labels without exaggerating the axis.
        if shaped - stock < 4:
            text(ax, position(stock) - 15, y - 24, str(stock), 15, TEAL, ha="right", weight="bold")
            text(ax, position(shaped) + 15, y - 24, str(shaped), 15, PURPLE, ha="left", weight="bold")
        else:
            text(ax, position(stock), y - 24, str(stock), 15, TEAL, ha="center", weight="bold")
            text(ax, position(shaped), y - 24, str(shaped), 15, PURPLE, ha="center", weight="bold")
        text(ax, 1410, y, f"+{shaped - stock}", 21, PURPLE, ha="center", weight="bold")
    text(ax, 60, 1020, "Replay recomputes counts from 600 curated gate-decision rows; it does not regenerate or rescore molecules.",
         12, MUTED)
    footer(ax, 1120, "Exploratory recorded results. This is not a decoupled-versus-coupled ablation, deployment evidence, or a biological efficacy claim.")
    save(fig, "03-pfm-example")


def lifecycle():
    fig, ax = canvas(1600, 1010, 4, "Independent lifecycles, explicit adoption",
                     "Task execution uses a selected version. Evolution maintains a separate invocation and state.")
    arrow(ax, [(300, 175), (1490, 175)], GRAY)
    label(ax, 890, 175, "conceptual ordering  /  not measured wall-clock time", GRAY)
    panel(ax, 60, 220, 1480, 235, "TASK EXECUTION", TEAL, TEAL_BG)
    card(ax, 300, 290, 260, 90, "Tasks at v0", ["Current selected version"], TEAL,
         title_size=18, body_size=11.5)
    card(ax, 680, 290, 260, 90, "More tasks at v0", ["No automatic candidate activation"], TEAL,
         title_size=17, body_size=10.5)
    card(ax, 1160, 290, 300, 90, "Subsequent tasks at v1", ["After explicit adoption"], TEAL,
         title_size=16, body_size=12)
    arrow(ax, [(560, 335), (680, 335)], TEAL)
    arrow(ax, [(940, 335), (1160, 335)], TEAL)
    panel(ax, 60, 515, 1480, 340, "SELF-EVOLUTION", PURPLE, PURPLE_BG)
    items = [
        (300, 190, "Read traces", "Select a problem"),
        (540, 190, "Build candidate", "Allowed changes"),
        (780, 200, "Evaluate", "Independent rules"),
        (1030, 200, "Adopt", "Explicit decision"),
    ]
    for x, w, title, line in items:
        card(ax, x, 595, w, 100, title, [line], PURPLE,
             title_size=14, body_size=11)
    for start, end in [(490, 540), (730, 780), (980, 1030)]:
        arrow(ax, [(start, 645), (end, 645)], PURPLE)
    arrow(ax, [(430, 380), (430, 480), (395, 480), (395, 595)], AMBER)
    label(ax, 555, 480, "persisted execution traces", AMBER)
    arrow(ax, [(1130, 595), (1130, 335), (1160, 335)], TEAL)
    label(ax, 1250, 480, "activate selected version", TEAL)
    card(ax, 540, 750, 690, 70, "Evolution Memory",
         ["Attempts, lineage, evaluation outcomes, decisions, and cost"],
         AMBER, AMBER_BG, title_size=16, body_size=11)
    arrow(ax, [(880, 695), (880, 750)], AMBER)
    arrow(ax, [(540, 785), (270, 785), (270, 645), (300, 645)], AMBER)
    label(ax, 400, 785, "reuse in a later cycle", AMBER)
    text(ax, 60, 905, "The shared state is durable evidence and selected versions; an in-progress candidate does not replace the task policy.",
         12, MUTED)
    footer(ax, 1010, "Lifecycle illustration only: no always-on scheduler, concurrent service, or hot-deployment mechanism is implied.")
    save(fig, "04-lifecycles")


if __name__ == "__main__":
    architecture()
    memories()
    pfm()
    lifecycle()

