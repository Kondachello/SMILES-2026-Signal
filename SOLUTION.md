**Result: 9.79 dB average** (per-channel: 11.01 / 8.30 / 12.41 / 7.44 dB)
on the provided `challenge.mat`. Baseline: 4.02 dB.

The whole canceller is in `applicant_solution.py`, ~25 lines. The script is deterministic.
```
V = (10 fixed IM3 cross-products × 13 lags ∈ [-6, +6]) ⊕ (one rank-1 spatial mode)
```

with two hard gates: ≥ 95 % of the removed energy must lie in V
(`explain_ratio`), and per channel `err_c ≤ 0.80 · residual_c`. Anything
outside breaks validity → score forced to 0 dB.

Both gates are computed *after* applying the band-pass filter `h` to prediction. **The filter is applied twice** — 
when `rank1` is built from `score_filter(rx − tx_pred)`and on `rx_after`
when the score function measures residual power.

### 1. Wideband rank-1 = exact deconvolution (free of charge)

The "naive" rank-1 step is

```python
target  = rank1_from_band_matrix(score_filter(rx - tx_pred))   # band-filtered, length N × 4
rx_hat  = rx - tx_pred - target
```

Score: ~7.0 dB. The Blackman-windowed FIR `h` is *not* idempotent on its
own output — passing `target` (already once-filtered) through `h` again
attenuates band edges by ~21 % in power.

The closed-form is one identity. Let `v ∈ ℂ⁴` be the top eigenvector of
the band-filtered residual's spatial covariance, `shared = (h*resid) @ v`
the band-filtered shared waveform, `w_c` the per-channel LS coefficients.
By **linearity of `h`**:

```
h * ((rx - tx_pred) @ v)  =  (h * (rx - tx_pred)) @ v  =  shared
```

So the *wideband* projection `shared_full := (rx - tx_pred) @ v` is
**exactly** `h⁻¹ shared`. Subtracting `w_c · shared_full` from `rx`
makes the scorer's second `h` pass produce `w_c · shared = target_c` on
each channel.

### 2. Per-channel auto-damped β (the validator binds on one channel only)

With β = 1 (the LS optimum) the wideband cancellation overshoots into
the per-channel guard. Diagnostically (I did some experiments), at β = 1 the four ratios are
`[0.06, 0.29, 0.90, 0.09]` — only **ch2 binds** (0.90 > 0.80); the other
three have 3–18× margin. So I kept β = 1 on slack channels and damped only the binding ones.

The TX-mismatch err per channel is essentially independent of β,
so the optimal `β_c` is the closed-form root of a quadratic:

```
residual_c(β) = A_c − (2β − β²) · B_c
A_c = ‖h*resid[:, c]‖² / N        (per-ch rx_band power)
B_c = |w_c|² · ‖shared‖² / N       (per-ch maximum cancellation)
β_c = 1 − sqrt(1 − clip((A_c − err_c / 0.795) / B_c, 0, 1))
```

`err_c` is read off from one validator decomposition; the formula is
iterated 2 times to chase the small fixed-point (err_c does *weakly*
depend on β through the validator's rank-1 step, but it's nearly fixed).
The 0.795 safety margin sits just inside the 0.80 hard guard.

Output for this capture: `β = [1.000, 1.000, 0.870, 1.000]`. ch2 damps
to 0.87, the other three stay at the LS optimum.

---
## Algorithm in 25 lines

```python
def your_canceller(tx_n, rx):
    fit_tx = helpers["fit_tx_prediction"]
    sf     = helpers["score_filter"]
    n_ch   = rx.shape[1]

    tx_pred = fit_tx(rx)
    resid   = rx - tx_pred

    resid_band = np.column_stack([sf(resid[:, c]) for c in range(n_ch)])
    cov = resid_band.conj().T @ resid_band / resid_band.shape[0]
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1]
    shared         = resid_band @ v
    shared_norm_sq = np.vdot(shared, shared).real
    w              = (shared.conj() @ resid_band) / shared_norm_sq
    shared_full    = resid @ v

    A_c = np.mean(np.abs(resid_band) ** 2, axis=0)
    B_c = np.abs(w) ** 2 * (shared_norm_sq / resid_band.shape[0])

    beta = np.ones(n_ch)
    for _ in range(2):
        rx_test = rx - tx_pred - shared_full[:, None] * (beta * w)[None, :]
        rem      = rx - rx_test
        rem_band = np.column_stack([sf(rem[:, c]) for c in range(n_ch)])
        tx_part  = fit_tx(rem)
        rv       = rem_band - tx_part
        cv       = rv.conj().T @ rv / rv.shape[0]
        _, vv    = np.linalg.eigh(cv)
        uv       = rv @ vv[:, -1]
        d_uv     = np.vdot(uv, uv).real
        rank1_v  = uv[:, None] * (uv.conj() @ rv / d_uv)[None, :]
        err_c    = np.mean(np.abs(rv - rank1_v) ** 2, axis=0).real

        ratio = np.clip((A_c - err_c / 0.795) / B_c, 0.0, 1.0)
        beta  = 1.0 - np.sqrt(1.0 - ratio)

    return rx - tx_pred - shared_full[:, None] * (beta * w)[None, :]
```

That's it. The first half is the LS rank-1 setup. The second half is two fixed-point iterations of
the analytical β.

---

## Experiments and failed attempts

I worked from the provided baseline (~4 dB) up to 9.79 dB through a lot of dead ends. All numbers are from the official scorer on `challenge.mat`.

### How the score evolved

| What I was trying | Avg dB |
|---|---|
| Baseline (TX only, no rank-1) | 4.02 |
| Naive band-filtered rank-1 on top of TX | 7.01 |
| Rank-1 first, then TX | 8.12 |
| Wideband rank-1, β = 1 (no damping) | **0 dB** (INVALID — ch2 hit the per-channel guard) |
| Wideband rank-1, uniform β = 0.86 | 9.63 |
| Wideband rank-1, per-channel β = [1, 1, 0.87, 1] | 9.79 |
| **Auto-tuned analytical β (final)** | **9.79** |

The 7.01 → 9.63 jump is the wideband identity (Idea 1). The 9.63 → 9.79 is per-channel β instead of a uniform one — at β = 1 the four ratios are `[0.06, 0.29, 0.90, 0.09]`, so only one channel actually binds.

### What didn't make it in

**ALS (alternate TX-fit and rank-1).** Score bounces: `7.01 → 5.27 → 5.04 → 6.27 → 8.45 → 4.76`. TX fit lives on the locked `[20_000, 220_000]` window, rank-1 lives on full N — the two domains aren't a contraction. Even with TX rebuilt on full N (chunked Gram), it still oscillated. Single-pass wins.

**Wider TX-lag window.** ±6 is the peak (8.12 dB, explain 0.973); ±10 → 8.02, ±20 → 7.72. Extra lags add flexibility but the non-redundant parts slide into `err`.

**Wider TX-fit subset.** Non-monotone: `[0, 500K]`, `[0, 1M]`, `[0, 2M]` are all INVALID (explain ≈ 0.92), full N comes back valid but at 7.35 dB. The validator's window isn't a free parameter — using a different one disagrees with the `α` it re-derives.

**Richer TX basis (5th-order IM5).** Adding `tx_i² · tx_j* · |tx_k|²` and envelope variants only fattens the validator's `err` term — the column space is welded at 10 terms. Score → 0 dB.

**FFT Wiener deconvolution of `h`.** Linear vs circular convolution mismatch at the 2047-tap edges blows up to ~1000× amplification. Best result 3.5 dB, below baseline. The wideband identity sidesteps this entirely.

**Richardson iteration on `tx_pred`.** Anything other than `fit_tx_prediction(rx)` lands in `err` because the validator re-runs that exact call. Score 4–7 dB, mostly invalid.

**Free 4-complex rank-1 weights via Nelder-Mead.** Converges back to `[1, 1, 0.87, 1] · w_LS`. The LS weights are already optimal on their direction.

**Higher-rank spatial.** Validator only takes the top eigenvector. Rank > 1 → `err`.

**Time-varying / block-wise rank-1.** Globally rank > 1 → only the dominant block survives → INVALID.

**ICA on the band-filtered RX.** Higher-order statistics give a different direction than the validator's covariance eigendecomp; off-axis content goes to `err`.