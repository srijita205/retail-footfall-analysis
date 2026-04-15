"""
dashboard.py
============
Smart Store Footfall & Behavior Analytics Dashboard
Built with Streamlit | MERL Shopping Dataset
Author: Srijita Kayal
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from data_loader import get_data, ACTION_COLORS, ACTION_LABELS

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Store Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global Styles
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── App background ── */
.stApp { background: #0d1117; color: #e6edf3; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.8rem; font-weight: 700; }

/* ── Section headers ── */
h1 { color: #e6edf3 !important; font-weight: 700; font-size: 1.9rem !important; }
h2 { color: #c9d1d9 !important; font-weight: 500; font-size: 1.2rem !important; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
h3 { color: #8b949e !important; font-size: 0.85rem !important; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Divider ── */
hr { border-color: #21262d !important; margin: 1.5rem 0; }

/* ── Insight cards ── */
.insight-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 6px 0;
    line-height: 1.6;
}
.insight-card .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin-bottom: 6px;
}
.insight-card .value {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 6px;
}
.insight-card .reco {
    font-size: 0.85rem;
    color: #8b949e;
}
.pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.pill-green  { background: #0d4429; color: #3fb950; border: 1px solid #238636; }
.pill-blue   { background: #0c2d6b; color: #79c0ff; border: 1px solid #1f6feb; }
.pill-orange { background: #3d1f00; color: #ffa657; border: 1px solid #9e4200; }
.pill-purple { background: #2d1f6e; color: #d2a8ff; border: 1px solid #6e40c9; }

/* ── Badge ── */
.badge {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
    padding: 3px 9px; border-radius: 4px;
    background: #1f6feb22; color: #58a6ff;
    border: 1px solid #1f6feb66;
    display: inline-block; margin-bottom: 6px;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #21262d !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important; border-radius: 6px !important;
}
.stDownloadButton > button:hover { background: #30363d !important; }

/* ── Expander ── */
[data-testid="stExpander"] { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib dark theme
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "#161b22",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#8b949e",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        120,
})

PALETTE = list(ACTION_COLORS.values())


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛍️ Smart Store\n### Analytics Dashboard")
    st.markdown("---")

    mode = st.radio(
        "**Data Source**",
        ["📡 Real MERL Data", "🧪 Simulated Demo"],
        index=1,
        help="Use real data if .mat files are accessible. Demo mode uses generated data.",
    )
    force_sim = (mode == "🧪 Simulated Demo")

    st.markdown("---")
    st.markdown("**Filters**")

    score_threshold = st.slider("Min Confidence Score", 0.0, 1.0, 0.5, 0.05)

    action_options = list(ACTION_LABELS.values())
    selected_actions = st.multiselect(
        "Actions to Include",
        options=action_options,
        default=action_options,
    )

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "Built on the [MERL Shopping Dataset](https://www.merl.com/demos/merl-shopping-dataset) — "
        "106 videos, 41 subjects, 5 in-store shopping actions."
    )
    st.markdown("---")
    st.caption("Srijita Kayal · Data Analytics Portfolio")


# ─────────────────────────────────────────────────────────────────────────────
# Load & Filter Data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_data(force_sim: bool):
    return get_data(force_simulate=force_sim)

with st.spinner("Loading data…"):
    df_raw, source = cached_data(force_sim)

df = df_raw[
    (df_raw["action"].isin(selected_actions)) &
    (df_raw["score"] >= score_threshold)
].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

source_labels = {
    "real_detected": ("LIVE · DETECTED ACTIONS", "pill-green"),
    "real_labels":   ("LIVE · GROUND TRUTH",     "pill-green"),
    "simulated":     ("DEMO · SIMULATED DATA",   "pill-blue"),
}
pill_text, pill_cls = source_labels.get(source, ("UNKNOWN", "pill-orange"))

st.markdown(f'<span class="pill {pill_cls}">{pill_text}</span>', unsafe_allow_html=True)
st.title("Smart Store Footfall & Behavior Analytics")
st.caption("Tracking what customers actually do at the shelf — and turning it into business decisions.")
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# KPI Row
# ─────────────────────────────────────────────────────────────────────────────

if df.empty:
    st.warning("No data matches current filters. Adjust the sidebar.")
    st.stop()

total_actions = len(df)
avg_duration  = df["duration_sec"].mean()
top_action    = df["action"].value_counts().index[0]
avg_score     = df["score"].mean()
unique_subjects = df["subject_id"].nunique() if "subject_id" in df.columns else "N/A"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Actions",       f"{total_actions:,}")
k2.metric("Avg Duration",        f"{avg_duration:.1f}s")
k3.metric("Top Action",          top_action.split()[0])
k4.metric("Avg Confidence",      f"{avg_score:.2%}")
k5.metric("Subjects",            str(unique_subjects))

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Row 1 — Action Frequency & Duration
# ─────────────────────────────────────────────────────────────────────────────

col_a, col_b = st.columns(2)

# — Chart 1: Action Frequency Horizontal Bar —
with col_a:
    st.markdown("### Action Frequency")
    counts = df["action"].value_counts().reset_index()
    counts.columns = ["action", "count"]
    counts = counts.sort_values("count")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [ACTION_COLORS[a] for a in counts["action"]]
    bars = ax.barh(counts["action"], counts["count"], color=colors, height=0.55, edgecolor="none")

    for bar, val in zip(bars, counts["count"]):
        ax.text(bar.get_width() + max(counts["count"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9, color="#c9d1d9")

    ax.set_xlabel("Number of Instances")
    ax.set_title("Which actions happen most?", fontsize=11, pad=10, color="#c9d1d9")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# — Chart 2: Duration Box Plot —
with col_b:
    st.markdown("### Duration Distribution by Action")
    fig, ax = plt.subplots(figsize=(7, 4))

    action_order = df.groupby("action")["duration_sec"].median().sort_values(ascending=False).index.tolist()
    duration_data = [df[df["action"] == a]["duration_sec"].values for a in action_order]
    box_colors = [ACTION_COLORS[a] for a in action_order]

    bp = ax.boxplot(
        duration_data,
        labels=[a.replace(" ", "\n") for a in action_order],
        patch_artist=True,
        medianprops=dict(color="#e6edf3", linewidth=2),
        whiskerprops=dict(color="#8b949e"),
        capprops=dict(color="#8b949e"),
        flierprops=dict(marker=".", color="#8b949e", markersize=4, alpha=0.5),
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel("Duration (seconds)")
    ax.set_title("How long does each action last?", fontsize=11, pad=10, color="#c9d1d9")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Row 2 — Time Heatmap & Score Distribution
# ─────────────────────────────────────────────────────────────────────────────

col_c, col_d = st.columns(2)

# — Chart 3: Activity Heatmap over Time —
with col_c:
    st.markdown("### Activity Heatmap Over Time")
    n_bins = 12
    df["time_bin"] = pd.cut(df["start_sec"], bins=n_bins, labels=False)
    heatmap_df = df.groupby(["action", "time_bin"]).size().unstack(fill_value=0)
    heatmap_df = heatmap_df.reindex(action_order)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        heatmap_df,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="#0d1117",
        cbar_kws={"shrink": 0.7, "label": "Action count"},
        annot=False,
    )
    ax.set_xlabel("Time Segment →  (start of video to end)")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_title("When do actions cluster?", fontsize=11, pad=10, color="#c9d1d9")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# — Chart 4: Confidence Score per Action —
with col_d:
    st.markdown("### Confidence Score by Action")
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, action in enumerate(action_order):
        subset = df[df["action"] == action]["score"]
        ax.hist(
            subset, bins=25, alpha=0.7,
            label=action, color=ACTION_COLORS[action], edgecolor="none"
        )

    ax.set_xlabel("Detection Confidence Score")
    ax.set_ylabel("Frequency")
    ax.set_title("How reliable are the detections?", fontsize=11, pad=10, color="#c9d1d9")
    ax.legend(fontsize=7, framealpha=0.1, edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.yaxis.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Row 3 — Action Timeline (if subject_id available)
# ─────────────────────────────────────────────────────────────────────────────

if "subject_id" in df.columns:
    st.markdown("---")
    st.markdown("### Subject Activity Timeline")
    st.caption("One subject's complete session — showing which actions occurred and when.")

    subject_ids = sorted(df["subject_id"].unique())
    selected_subject = st.select_slider("Select Subject", options=subject_ids, value=subject_ids[0])

    subject_df = df[df["subject_id"] == selected_subject].sort_values("start_sec")

    if not subject_df.empty:
        sessions = sorted(subject_df["session_id"].unique()) if "session_id" in subject_df.columns else [1]

        fig, axes = plt.subplots(len(sessions), 1, figsize=(12, 2.5 * len(sessions)), squeeze=False)

        for i, sess in enumerate(sessions):
            ax = axes[i][0]
            sess_df = subject_df[subject_df.get("session_id", pd.Series([1]*len(subject_df))) == sess] \
                if "session_id" in subject_df.columns else subject_df

            y_map = {a: j for j, a in enumerate(action_order)}
            for _, row in sess_df.iterrows():
                y = y_map.get(row["action"], 0)
                ax.barh(
                    y, row["duration_sec"],
                    left=row["start_sec"],
                    height=0.5,
                    color=ACTION_COLORS[row["action"]],
                    alpha=0.85,
                    edgecolor="none",
                )

            ax.set_yticks(range(len(action_order)))
            ax.set_yticklabels(action_order, fontsize=8)
            ax.set_xlabel("Time (seconds)")
            ax.set_title(f"Session {sess}", fontsize=10, color="#c9d1d9")
            ax.xaxis.grid(True)
            ax.yaxis.grid(False)

        plt.suptitle(f"Subject {selected_subject} — Full Activity Timeline", fontsize=12, color="#e6edf3", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Business Insights
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 💡 Business Insights & Recommendations")

action_stats = df.groupby("action").agg(
    count=("action", "count"),
    avg_duration=("duration_sec", "mean"),
    avg_score=("score", "mean"),
).reset_index().sort_values("count", ascending=False)

high_action = action_stats.iloc[0]
long_action = action_stats.sort_values("avg_duration", ascending=False).iloc[0]
low_action  = action_stats.iloc[-1]
insp = action_stats[action_stats["action"].str.startswith("Inspect")]
insp_product_count = insp[insp["action"] == "Inspect Product"]["count"].values
insp_shelf_count   = insp[insp["action"] == "Inspect Shelf"]["count"].values

i1, i2, i3 = st.columns(3)

with i1:
    st.markdown(f"""
    <div class="insight-card">
        <div class="label">🏆 Highest Traffic Action</div>
        <div class="value">{high_action['action']}</div>
        <div class="reco">
            Occurs <strong>{int(high_action['count']):,}×</strong> — 
            place high-margin impulse products at this touchpoint. 
            Consider combo offers adjacent to this zone.
        </div>
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown(f"""
    <div class="insight-card">
        <div class="label">⏳ Longest Engagement</div>
        <div class="value">{long_action['action']}</div>
        <div class="reco">
            Avg <strong>{long_action['avg_duration']:.1f}s</strong> per instance — 
            customers linger here. Prioritise shelf visibility, 
            better lighting, and premium product placement.
        </div>
    </div>
    """, unsafe_allow_html=True)

with i3:
    st.markdown(f"""
    <div class="insight-card">
        <div class="label">📉 Needs Attention</div>
        <div class="value">{low_action['action']}</div>
        <div class="reco">
            Only <strong>{int(low_action['count']):,}×</strong> — 
            lowest engagement action. Review product placement, 
            signage, and shelf-eye-level alignment to boost interaction.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Inspect product vs inspect shelf ratio
if len(insp_product_count) and len(insp_shelf_count):
    ratio = insp_product_count[0] / max(insp_shelf_count[0], 1)
    st.markdown(f"""
    <div class="insight-card" style="margin-top: 10px;">
        <div class="label">🔎 Product Pick-Up Ratio</div>
        <div class="value">
            {"High" if ratio > 1 else "Low"} Product Inspection vs Shelf Browse
        </div>
        <div class="reco">
            <em>Inspect Product</em> to <em>Inspect Shelf</em> ratio = 
            <strong>{ratio:.2f}</strong>.
            {"Customers who browse are converting to pick-up — good shelf design signals." if ratio > 1
             else "Many customers browse but don't pick up — review product appeal, pricing labels, or facings."}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Raw Data & Export
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
with st.expander("📋 Raw Data Preview & Export"):
    st.dataframe(df.head(200), use_container_width=True)
    st.caption(f"Showing first 200 of {len(df):,} rows. Download for full dataset.")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️  Download Processed Data (.csv)",
        data=csv,
        file_name="smart_store_processed.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#484f58; font-size:0.8rem; padding: 12px 0;">'
    "Smart Store Analytics · Built by <strong style='color:#8b949e'>Srijita Kayal</strong> · "
    "MERL Shopping Dataset (Mitsubishi Electric Research Labs, 2016)"
    "</div>",
    unsafe_allow_html=True,
)


