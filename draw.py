import csv
import json
import os
from typing import Iterable, List


def _as_list(x: Iterable) -> List:
    if x is None:
        return []
    return list(x)


def visualize(
    acc_list,
    error_list,
    f1_list,
    epoch_list,
    time_list,
    output_dir,
    entropy_list=None,
    kl2u_list=None,
    confidence_list=None,
):
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "epoch": _as_list(epoch_list),
        "acc": _as_list(acc_list),
        "error": _as_list(error_list),
        "f1": _as_list(f1_list),
        "time_sec": _as_list(time_list),
        "entropy": _as_list(entropy_list),
        "kl2u": _as_list(kl2u_list),
        "confidence": _as_list(confidence_list),
    }

    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "acc",
                "error",
                "f1",
                "time_sec",
                "entropy",
                "kl2u",
                "confidence",
            ]
        )
        n = len(payload["epoch"])
        for i in range(n):
            writer.writerow(
                [
                    payload["epoch"][i],
                    payload["acc"][i] if i < len(payload["acc"]) else "",
                    payload["error"][i] if i < len(payload["error"]) else "",
                    payload["f1"][i] if i < len(payload["f1"]) else "",
                    payload["time_sec"][i] if i < len(payload["time_sec"]) else "",
                    payload["entropy"][i] if i < len(payload["entropy"]) else "",
                    payload["kl2u"][i] if i < len(payload["kl2u"]) else "",
                    payload["confidence"][i] if i < len(payload["confidence"]) else "",
                ]
            )

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    def _plot(y, title, fname):
        if not y:
            return
        plt.figure()
        plt.plot(payload["epoch"][: len(y)], y)
        plt.title(title)
        plt.xlabel("epoch")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()

    _plot(payload["acc"], "Accuracy", "acc.png")
    _plot(payload["error"], "Error", "error.png")
    _plot(payload["f1"], "F1", "f1.png")
    _plot(payload["entropy"], "Predictive Entropy", "entropy.png")
    _plot(payload["kl2u"], "KL to Uniform", "kl2u.png")
    _plot(payload["confidence"], "Confidence", "confidence.png")
