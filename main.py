#!/usr/bin/env python3
# pumpfun-top5-forecast-01: Monte Carlo order + growth-projected Gaussian sizes.
# Python 3.13, stdlib only, deterministic (fixed seed, no clock, no network).
# Reads /input (task.json, history.ndjson), writes /output/forecast.json.

import json, os, math, random
from datetime import datetime

INPUT = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# ---- order model ----
SIMS = 1200          # sims per answer read
DRIFT_SHRINK = 0.50  # keep half of recent momentum
SIGMA_BASE = 0.22    # log10 sd per sqrt(10 min)
SIGMA_FLOOR = 0.22
GHOST_RATE = 1.20    # newcomer coefficient
GHOST_POW = 0.75
GHOST_CAP = 4.0
GHOST_MU = 3.40      # mean log10 cap of a serious newcomer
GHOST_SD = 0.55
MAX_CAND = 160

# ---- size model (cap at rank k projected with its own growth) ----
GROWTH_SHRINK = 0.50  # project half the observed per-rank growth
SIZE_SD = 0.38        # base sd of log10 cap
SIZE_SD_MAX = 0.85


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() / 60.0


def poisson(lam, rng):
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def erfc(x):
    z = abs(x)
    t = 1 / (1 + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2 - r


def above(x, mean, sd):
    """P(X >= x) for X ~ Normal(mean, sd)."""
    if x == -math.inf:
        return 1.0
    if x == math.inf:
        return 0.0
    return 0.5 * erfc((x - mean) / (sd * math.sqrt(2)))


def main():
    task = json.load(open(os.path.join(INPUT, "task.json")))
    lines = [json.loads(l) for l in open(os.path.join(INPUT, "history.ndjson")) if l.strip()]
    marks = [m for m in lines if m.get("ok") and m.get("rows")]
    if not marks:
        raise SystemExit("no ok mark")
    places = int(task["places"])
    edges = [float(x) for x in task["size"]["edges"]]
    nbin = len(edges) + 1
    reads = [ts(r) for r in task["reads"]]

    # per-mint time series of log10(cap in SOL)
    series = {}
    for m in marks:
        t = ts(m["at"])
        for r in m["rows"]:
            c = math.log10(int(r["market_cap_lamports"])) - 9.0
            series.setdefault(r["mint"], []).append((t, c))
    for v in series.values():
        v.sort()
    last_t = max(v[-1][0] for v in series.values())

    # ---- order: Monte Carlo over candidate coins ----
    cand = []
    for mint, v in series.items():
        t_last, c_last = v[-1]
        drift = 0.0
        if len(v) >= 2:
            j = max(0, len(v) - 4)
            dt = t_last - v[j][0]
            if dt > 0:
                drift = (c_last - v[j][1]) / dt
        cand.append((c_last, drift, mint))
    cand.sort(reverse=True)
    cand = cand[:MAX_CAND]

    rng = random.Random(20261112)
    place_cnt = {}
    total_sims = 0
    for T in reads:
        dt_min = max(0.0, T - last_t)
        sig = SIGMA_BASE * math.sqrt(dt_min / 10.0)
        lam = min(GHOST_CAP, GHOST_RATE * (max(dt_min, 1.0) / 10.0) ** GHOST_POW)
        for _ in range(SIMS):
            caps = []
            for c_last, drift, mint in cand:
                adv = drift * dt_min * DRIFT_SHRINK
                c = c_last + adv + (rng.gauss(0.0, sig) if sig > 0 else 0.0)
                caps.append((c, mint))
            for _g in range(poisson(lam, rng)):
                caps.append((rng.gauss(GHOST_MU, GHOST_SD), None))
            caps.sort(reverse=True)
            for pi, (c, mint) in enumerate(caps[:places]):
                if mint is not None:
                    place_cnt[(mint, pi + 1)] = place_cnt.get((mint, pi + 1), 0) + 1
            total_sims += 1

    probabilities = []
    for place in range(1, places + 1):
        rows = sorted(((cnt, m) for (m, pl), cnt in place_cnt.items() if pl == place), reverse=True)[:100]
        rounded = [(round(cnt / total_sims, 6), m) for cnt, m in rows if cnt / total_sims >= 1e-4]
        excess = sum(p for p, _ in rounded) - 0.999999
        if excess > 0:
            rounded.sort(reverse=True)
            rounded[0] = (round(rounded[0][0] - excess, 6), rounded[0][1])
        for p, m in rounded:
            if p > 0:
                probabilities.append({"mint": m, "place": place, "p": p})

    # ---- sizes: cap at rank k, projected with its own growth, Gaussian over bins ----
    rank_series = {}
    for m in marks:
        t = ts(m["at"])
        for r in sorted(m["rows"], key=lambda r: r["rank"])[:100]:
            cap = math.log10(int(r["market_cap_lamports"])) - 9.0
            rank_series.setdefault(r["rank"], []).append((t, cap))

    read_lo, read_hi = min(reads), max(reads)
    dt_mid = max(1.0, (read_lo + read_hi) / 2.0 - last_t)
    read_span = max(0.0, read_hi - read_lo)

    sizes = []
    for place in range(1, places + 1):
        pts = rank_series.get(place, [])
        if not pts:
            continue
        c_last = pts[-1][1]
        slope = 0.0
        if len(pts) >= 2:
            p = pts[-3:]
            n = len(p)
            sx = sum(x for x, _ in p)
            sy = sum(y for _, y in p)
            sxx = sum(x * x for x, _ in p)
            sxy = sum(x * y for x, y in p)
            den = n * sxx - sx * sx
            if den > 1e-9:
                slope = (n * sxy - sx * sy) / den
        proj = c_last + slope * dt_mid * GROWTH_SHRINK
        sd = math.sqrt(SIZE_SD ** 2 + (slope * read_span * 0.5) ** 2)
        sd = min(max(sd, SIZE_SD), SIZE_SD_MAX)
        rows = []
        for b in range(nbin):
            lo = -math.inf if b == 0 else edges[b - 1]
            hi = math.inf if b == len(edges) else edges[b]
            pr = above(lo, proj, sd) - above(hi, proj, sd)
            if pr > 1e-5:
                rows.append((pr, b))
        rounded = [(round(pr, 6), b) for pr, b in rows]
        excess = sum(pr for pr, _ in rounded) - 0.999999
        if excess > 0:
            rounded.sort(reverse=True)
            rounded[0] = (round(rounded[0][0] - excess, 6), rounded[0][1])
        for pr, b in rounded:
            if pr > 0:
                sizes.append({"place": place, "bin": b, "p": pr})

    # top5: expected place from the order probabilities
    score = {}
    for r in probabilities:
        score[r["mint"]] = score.get(r["mint"], 0.0) + r["p"] * (6 - r["place"])
    top5 = [m for m, _ in sorted(score.items(), key=lambda kv: -kv[1])[:5]]
    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    if len(top5) < 5:
        for r in last_rows:
            if r["mint"] not in top5 and r["mint"] in series:
                top5.append(r["mint"])
            if len(top5) >= 5:
                break
    while len(top5) < 5:
        for m in series:
            if m not in top5:
                top5.append(m)
                break

    out = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT, exist_ok=True)
    with open(os.path.join(OUTPUT, task.get("output", "forecast.json")), "w") as f:
        json.dump(out, f)
    print(f"cand={len(cand)} sims={total_sims} prob_rows={len(probabilities)} size_rows={len(sizes)}", flush=True)


if __name__ == "__main__":
    main()
