import json
import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

# Download the dataset
# url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
# downloaded_file = "challenge.mat"
# gdown.download(url, downloaded_file, quiet=False, fuzzy=True)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx):
    fit_tx = helpers["fit_tx_prediction"]
    sf = helpers["score_filter"]
    n_ch = rx.shape[1]

    tx_pred = fit_tx(rx)
    resid = rx - tx_pred

    resid_band = np.column_stack([sf(resid[:, c]) for c in range(n_ch)])
    cov = resid_band.conj().T @ resid_band / resid_band.shape[0]
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1]
    shared = resid_band @ v
    shared_norm_sq = np.vdot(shared, shared).real
    w = (shared.conj() @ resid_band) / shared_norm_sq
    shared_full = resid @ v

    A_c = np.mean(np.abs(resid_band) ** 2, axis=0)
    B_c = np.abs(w) ** 2 * (shared_norm_sq / resid_band.shape[0])

    beta = np.ones(n_ch)
    for _ in range(2):
        rx_test = rx - tx_pred - shared_full[:, None] * (beta * w)[None, :]
        rem = rx - rx_test
        rem_band = np.column_stack([sf(rem[:, c]) for c in range(n_ch)])
        tx_part = fit_tx(rem)
        rv = rem_band - tx_part
        cv = rv.conj().T @ rv / rv.shape[0]
        _, vv = np.linalg.eigh(cv)
        uv = rv @ vv[:, -1]
        d_uv = np.vdot(uv, uv).real
        rank1_v = uv[:, None] * (uv.conj() @ rv / d_uv)[None, :]
        err_c = np.mean(np.abs(rv - rank1_v) ** 2, axis=0).real

        ratio = np.clip((A_c - err_c / 0.795) / B_c, 0.0, 1.0)
        beta = 1.0 - np.sqrt(1.0 - ratio)

    return rx - tx_pred - shared_full[:, None] * (beta * w)[None, :]


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
