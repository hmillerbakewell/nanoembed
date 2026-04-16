"""Structured logging and plotting for experiment runs.

Each Run creates a timestamped directory containing:
  - log.txt     — tee'd console output
  - results.json — structured metrics + metadata
  - results.png  — Pareto plot (size vs primary metric)
"""


import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RunResult:
    name: str
    size_mb: float
    metrics: dict[str, float]
    provenance: list[str] = field(default_factory=list)


class Run:
    """A timestamped experiment run directory that captures logs, JSON, and plots."""

    def __init__(self, label: str, root: str | Path = "runs",
                 metadata: dict | None = None) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / f"{ts}_{label}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "log.txt"
        self.json_path = self.dir / "results.json"
        self.plot_path = self.dir / "results.png"
        self.results: list[RunResult] = []
        self.metadata: dict = {
            "label": label,
            "timestamp": ts,
            **(metadata or {}),
        }
        # Truncate log file on creation
        self.log_path.write_text("")

    def log(self, msg: str = "") -> None:
        """Print to stdout and append to log.txt."""
        print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def add_result(self, result: RunResult) -> None:
        self.results.append(result)

    def save(self) -> None:
        """Write results.json."""
        data = {
            "metadata": self.metadata,
            "results": [asdict(r) for r in self.results],
        }
        self.json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def plot_pareto(
        self,
        primary_metric: str = "ndcg@10",
        title: str | None = None,
        color_by: str | None = None,
    ) -> Path | None:
        """Scatter plot of size_mb vs primary_metric. Returns the plot path or None."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.log("matplotlib not installed; skipping plot")
            return None

        if not self.results:
            return None

        # Filter to results that have the primary metric
        pts = [r for r in self.results if primary_metric in r.metrics]
        if not pts:
            self.log(f"no results with metric '{primary_metric}'; skipping plot")
            return None

        xs = [r.size_mb for r in pts]
        ys = [r.metrics[primary_metric] for r in pts]
        labels = [r.name for r in pts]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Optional colour by a provenance keyword (e.g. "pca" extracts pca-128d step)
        if color_by is not None:
            groups: dict[str, list[int]] = {}
            for i, r in enumerate(pts):
                key = next(
                    (step for step in r.provenance if step.startswith(color_by)),
                    "other",
                )
                groups.setdefault(key, []).append(i)
            cmap = plt.get_cmap("tab10")
            for ci, (key, idxs) in enumerate(sorted(groups.items())):
                ax.scatter(
                    [xs[i] for i in idxs],
                    [ys[i] for i in idxs],
                    label=key,
                    color=cmap(ci),
                    s=60,
                    alpha=0.85,
                )
            ax.legend(title=color_by, loc="best", fontsize=8)
        else:
            ax.scatter(xs, ys, s=60, alpha=0.85)

        # Label points with short names
        for x, y, label in zip(xs, ys, labels):
            short = label
            ax.annotate(short, (x, y), fontsize=6, xytext=(3, 3),
                        textcoords="offset points", alpha=0.7)

        ax.set_xscale("log")
        ax.set_xlabel("size (MB, log scale)")
        ax.set_ylabel(primary_metric)
        ax.set_title(title or f"Pareto: size vs {primary_metric}")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)
        return self.plot_path

    def finalise(self, primary_metric: str = "ndcg@10",
                 color_by: str | None = "pca") -> None:
        """Save JSON and plot. Call at the end of a run."""
        self.save()
        plot = self.plot_pareto(primary_metric=primary_metric, color_by=color_by)
        self.log(f"\nSaved results to {self.json_path}")
        if plot:
            self.log(f"Saved plot to {plot}")
