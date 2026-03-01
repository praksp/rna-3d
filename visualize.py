#!/usr/bin/env python3
"""
Visualize RNA 3D structure predictions from the submission CSV.

Generates interactive 3D plots for each target using Plotly.
Each of the 5 predictions is shown as a separate colored trace so
you can compare diversity across predictions.

Usage:
    python visualize.py                                  # all targets
    python visualize.py --target 8ZNQ                    # single target
    python visualize.py --target 8ZNQ 9IWF               # multiple targets
    python visualize.py --prediction 1                   # only prediction 1
    python visualize.py --output-dir viz                 # save HTML files
    python visualize.py --side-by-side --target 8ZNQ     # 5 predictions side-by-side
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


NUCLEOTIDE_COLORS = {
    "A": "#e74c3c",  # red
    "C": "#3498db",  # blue
    "G": "#2ecc71",  # green
    "U": "#f39c12",  # orange
}

PREDICTION_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
]


def load_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["target_id"] = df["ID"].apply(lambda x: x.rsplit("_", 1)[0])
    return df


def plot_target_overlay(df_target: pd.DataFrame, target_id: str,
                        predictions: list = None) -> go.Figure:
    """Plot all 5 predictions overlaid in one 3D scene.

    Each prediction is a separate trace with its own color.
    Nucleotide types are shown as hover labels.
    """
    if predictions is None:
        predictions = [1, 2, 3, 4, 5]

    fig = go.Figure()

    for pred_idx in predictions:
        x_col = f"x_{pred_idx}"
        y_col = f"y_{pred_idx}"
        z_col = f"z_{pred_idx}"

        x = df_target[x_col].values
        y = df_target[y_col].values
        z = df_target[z_col].values

        hover = [
            f"Residue {row['resid']}: {row['resname']}<br>"
            f"({row[x_col]:.1f}, {row[y_col]:.1f}, {row[z_col]:.1f})"
            for _, row in df_target.iterrows()
        ]

        # Backbone trace (line connecting consecutive residues)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=PREDICTION_COLORS[pred_idx - 1], width=3),
            name=f"Prediction {pred_idx} backbone",
            legendgroup=f"pred_{pred_idx}",
            showlegend=True,
            hoverinfo="skip",
        ))

        # Nucleotide markers
        marker_colors = [NUCLEOTIDE_COLORS.get(r, "#999999")
                         for r in df_target["resname"]]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=4,
                color=marker_colors,
                opacity=0.8,
                line=dict(width=1, color=PREDICTION_COLORS[pred_idx - 1]),
            ),
            name=f"Prediction {pred_idx} residues",
            legendgroup=f"pred_{pred_idx}",
            showlegend=False,
            hovertext=hover,
            hoverinfo="text",
        ))

    n_res = len(df_target)
    fig.update_layout(
        title=dict(
            text=f"RNA 3D Structure: {target_id} ({n_res} residues)",
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
            aspectmode="data",
        ),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        width=1000,
        height=700,
    )

    return fig


def plot_target_side_by_side(df_target: pd.DataFrame,
                              target_id: str) -> go.Figure:
    """Plot all 5 predictions in a 1x5 side-by-side layout."""
    fig = make_subplots(
        rows=1, cols=5,
        specs=[[{"type": "scatter3d"}] * 5],
        subplot_titles=[f"Prediction {i}" for i in range(1, 6)],
        horizontal_spacing=0.02,
    )

    for pred_idx in range(1, 6):
        x = df_target[f"x_{pred_idx}"].values
        y = df_target[f"y_{pred_idx}"].values
        z = df_target[f"z_{pred_idx}"].values

        marker_colors = [NUCLEOTIDE_COLORS.get(r, "#999999")
                         for r in df_target["resname"]]

        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines+markers",
                line=dict(color=PREDICTION_COLORS[pred_idx - 1], width=3),
                marker=dict(size=3, color=marker_colors, opacity=0.8),
                name=f"Pred {pred_idx}",
                showlegend=False,
            ),
            row=1, col=pred_idx,
        )

    n_res = len(df_target)
    fig.update_layout(
        title=dict(
            text=f"RNA 3D Structure: {target_id} ({n_res} residues) — Side-by-Side",
            font=dict(size=16),
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        width=2000,
        height=500,
    )

    for i in range(1, 6):
        scene_key = f"scene{i}" if i > 1 else "scene"
        fig.update_layout(**{
            scene_key: dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode="data",
            )
        })

    return fig


def plot_residue_colored(df_target: pd.DataFrame, target_id: str,
                          pred_idx: int = 1) -> go.Figure:
    """Single prediction colored by nucleotide type with residue labels."""
    x = df_target[f"x_{pred_idx}"].values
    y = df_target[f"y_{pred_idx}"].values
    z = df_target[f"z_{pred_idx}"].values

    marker_colors = [NUCLEOTIDE_COLORS.get(r, "#999999")
                     for r in df_target["resname"]]

    hover = [
        f"<b>{row['resname']}{row['resid']}</b><br>"
        f"({row[f'x_{pred_idx}']:.1f}, {row[f'y_{pred_idx}']:.1f}, {row[f'z_{pred_idx}']:.1f})"
        for _, row in df_target.iterrows()
    ]

    fig = go.Figure()

    # Backbone colored by position (rainbow gradient)
    n = len(x)
    colors_gradient = np.linspace(0, 1, n)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(
            color=colors_gradient,
            colorscale="Rainbow",
            width=5,
        ),
        name="Backbone (5'→3')",
        hoverinfo="skip",
    ))

    # Nucleotide-colored markers
    for nt, color in NUCLEOTIDE_COLORS.items():
        mask = df_target["resname"] == nt
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=x[mask], y=y[mask], z=z[mask],
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.9),
            name=f"{nt} ({mask.sum()})",
            hovertext=[h for h, m in zip(hover, mask) if m],
            hoverinfo="text",
        ))

    # Mark 5' and 3' ends
    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode="markers+text",
        marker=dict(size=10, color="black", symbol="diamond"),
        text=["5'"], textposition="top center",
        name="5' end",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode="markers+text",
        marker=dict(size=10, color="black", symbol="square"),
        text=["3'"], textposition="top center",
        name="3' end",
        hoverinfo="skip",
    ))

    n_res = len(df_target)
    fig.update_layout(
        title=dict(
            text=f"RNA Structure: {target_id} — Prediction {pred_idx} "
                 f"({n_res} residues, 5'→3' rainbow)",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="X (Å)", yaxis_title="Y (Å)", zaxis_title="Z (Å)",
            aspectmode="data",
        ),
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        width=1000,
        height=700,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize RNA 3D predictions")
    parser.add_argument("--submission", type=str, default="output/submission.csv",
                        help="Path to submission CSV")
    parser.add_argument("--target", nargs="*", default=None,
                        help="Target IDs to visualize (default: all)")
    parser.add_argument("--prediction", type=int, default=None,
                        help="Show only this prediction number (1-5)")
    parser.add_argument("--output-dir", type=str, default="viz",
                        help="Directory to save HTML files")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Show 5 predictions side by side")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser, only save files")
    args = parser.parse_args()

    df = load_submission(args.submission)
    targets = sorted(df["target_id"].unique())

    if args.target:
        targets = [t for t in args.target if t in targets]
        if not targets:
            print(f"No matching targets found. Available: {sorted(df['target_id'].unique())}")
            return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("  RNA 3D — Visualization")
    print("  " + "—" * 50)
    print(f"  Submission: {args.submission}")
    print(f"  Targets:   {len(targets)}")
    print(f"  Output:    {output_dir}/")
    print()

    for target_id in targets:
        df_target = df[df["target_id"] == target_id].copy()
        n_res = len(df_target)
        print(f"  {target_id}: {n_res} residues")

        # Main overlay plot
        if args.prediction:
            preds = [args.prediction]
        else:
            preds = [1, 2, 3, 4, 5]

        fig_overlay = plot_target_overlay(df_target, target_id, preds)
        overlay_path = output_dir / f"{target_id}_overlay.html"
        fig_overlay.write_html(str(overlay_path))
        print(f"    -> {overlay_path}")

        # Residue-colored detail plot (best prediction)
        pred_num = args.prediction or 1
        fig_detail = plot_residue_colored(df_target, target_id, pred_num)
        detail_path = output_dir / f"{target_id}_detail.html"
        fig_detail.write_html(str(detail_path))
        print(f"    -> {detail_path}")

        # Side-by-side (optional)
        if args.side_by_side:
            fig_sbs = plot_target_side_by_side(df_target, target_id)
            sbs_path = output_dir / f"{target_id}_sidebyside.html"
            fig_sbs.write_html(str(sbs_path))
            print(f"    -> {sbs_path}")

    # Generate index page linking all visualizations
    _write_index(output_dir, targets, args.side_by_side, df=df)

    if not args.no_browser and len(targets) <= 5:
        import webbrowser
        webbrowser.open(str((output_dir / "index.html").resolve()))


def _write_index(output_dir: Path, targets: list, has_sbs: bool, df=None):
    """Generate a modern HTML dashboard linking all target visualizations."""
    n_total = len(targets)
    # Optionally load run_log for best scores
    run_log = {}
    log_path = output_dir.parent / "output" / "run_log.json"
    if not log_path.exists():
        log_path = Path("output") / "run_log.json"
    if log_path.exists():
        try:
            import json
            with open(log_path) as f:
                run_log = json.load(f)
        except Exception:
            pass

    rows = []
    for tid in targets:
        n_res = ""
        if df is not None:
            n_res = len(df[df["target_id"] == tid])
        score = run_log.get(tid, {}).get("best_score")
        score_str = f"{score:.3f}" if score is not None else "—"
        rows.append((tid, n_res, score_str))

    table_rows = "".join(
        f"""
        <tr>
          <td><a href="{tid}_overlay.html" class="target-link">{tid}</a></td>
          <td>{n_res}</td>
          <td class="score">{score_str}</td>
          <td>
            <a href="{tid}_overlay.html" class="btn btn-overlay">Overlay</a>
            <a href="{tid}_detail.html" class="btn btn-detail">Detail</a>
            {f'<a href="{tid}_sidebyside.html" class="btn btn-sbs">Side-by-side</a>' if has_sbs else ''}
          </td>
        </tr>"""
        for tid, n_res, score_str in rows
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RNA 3D Folding — Visualization Dashboard</title>
  <style>
    :root {{
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --accent-hover: #79b8ff;
      --green: #3fb950;
      --border: #30363d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
      line-height: 1.5;
      min-height: 100vh;
    }}
    .container {{ max-width: 1000px; margin: 0 auto; }}
    header {{
      margin-bottom: 32px;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{
      font-size: 1.75rem;
      font-weight: 600;
      margin: 0 0 8px 0;
      color: var(--text);
    }}
    .subtitle {{ color: var(--muted); font-size: 0.95rem; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{
      font-weight: 600;
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(88, 166, 255, 0.06); }}
    .target-link {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 500;
    }}
    .target-link:hover {{ color: var(--accent-hover); text-decoration: underline; }}
    .btn {{
      display: inline-block;
      padding: 6px 12px;
      margin-right: 8px;
      margin-bottom: 4px;
      border-radius: 6px;
      font-size: 0.85rem;
      text-decoration: none;
      background: var(--border);
      color: var(--text);
      transition: background 0.15s;
    }}
    .btn:hover {{ background: var(--accent); color: #fff; }}
    .btn-overlay {{ }}
    .btn-detail {{ }}
    .btn-sbs {{ }}
    .score {{ font-variant-numeric: tabular-nums; color: var(--green); }}
    footer {{ margin-top: 32px; color: var(--muted); font-size: 0.85rem; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>RNA 3D Structure Predictions</h1>
      <p class="subtitle">{n_total} targets · Interactive Plotly 3D · Overlay, detail, and side-by-side views</p>
    </header>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Target</th>
            <th>Residues</th>
            <th>Best score</th>
            <th>Views</th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </div>
    <footer>
      Generated by visualize.py · Open any link to explore 3D structures. Rotate, zoom, and toggle traces in the legend.
    </footer>
  </div>
</body>
</html>"""

    index_path = output_dir / "index.html"
    index_path.write_text(html)
    print(f"\n  Index: {index_path}")


if __name__ == "__main__":
    main()
