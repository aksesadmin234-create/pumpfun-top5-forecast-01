#!/usr/bin/env python3
# pumpfun-top5-forecast-01: Monte Carlo forecaster (Python 3.13, stdlib only).
# Reads /input (task.json, history.ndjson), writes /output/forecast.json.
# Deterministic: fixed seed, no clock, no network.

import json, os, math, random
from datetime import datetime

INPUT = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# ---- tunable parameters (tuned on the practice history's marks) ----
SIMS = 1500          # sims per answer read
DRIFT_SHRINK = 0.30  # shrink on recent momentum
SIGMA_BASE = 0.45    # log10 sd per sqrt(10 min)
SIGMA_FLOOR = 0.45
GHOST_RATE = 1.20    # newcomer coefficient
GHOST_POW = 0.75     # newcomer growth with horizon
GHOST_CAP = 4.0      # max expected newcomer count
GHOST_MU = 3.30      # mean log10 cap of a serious newcomer
GHOST_SD = 0.50
SIZE_TEMP = 1.60     # sharpen (>1) the size histogram
MAX_CAND = 160


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

    # volatility estimate: robust sd of per-sqrt-mark changes
    deltas = []
    for v in series.values():
        for i in range(1, len(v)):
            dt = v[i][0] - v[i - 1][0]
            if dt > 0:
                deltas.append((v[i][1] - v[i - 1][1]) / math.sqrt(dt / 10.0))
    if deltas:
        deltas.sort()
        med = deltas[len(deltas) // 2]
        mad = sorted(abs(d - med) for d in deltas)[len(deltas) // 2]
        sigma_mark = max(SIGMA_FLOOR, 1.4826 * mad)
    else:
        sigma_mark = SIGMA_BASE

    # candidate coins: largest at the last mark + strong movers
    cand = []
    for mint, v in series.items():
        t_last, c_last = v[-1]
        drift = 0.0
        if len(v) >= 2:
            j = max(0, len(v) - 4)
            dt = t_last - v[j][0]
            if dt > 0:
                drift = (c_last - v[j][1]) / dt  # per minute
        cand.append((c_last, drift, mint))
    cand.sort(reverse=True)
    cand = cand[:MAX_CAND]

    # last-mark ranks for top5 guess
    last_rows = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    hist_mints = set(series.keys())

    rng = random.Random(20261112)
    # accumulators over reads
    place_cnt = {}          # (mint, place) -> count
    size_cnt = [[0] * nbin for _ in range(places + 1)]
    total_sims = 0

    for T in reads:
        dt_min = T - last_t
        if dt_min < 0:
            dt_min = 0.0
        scale = math.sqrt(dt_min / 10.0)
        sig = sigma_mark * scale
        lam = min(GHOST_CAP, GHOST_RATE * (max(dt_min, 1.0) / 10.0) ** GHOST_POW)
        for _ in range(SIMS):
            caps = []
            for c_last, drift, mint in cand:
                adv = drift * dt_min * DRIFT_SHRINK
                c = c_last + adv + (rng.gauss(0.0, sig) if sig > 0 else 0.0)
                caps.append((c, mint))
            k = poisson(lam, rng)
            for _g in range(k):
                caps.append((rng.gauss(GHOST_MU, GHOST_SD), None))
            caps.sort(reverse=True)
            top = caps[:places]
            for pi, (c, mint) in enumerate(top):
                if mint is not None:
                    key = (mint, pi + 1)
                    place_cnt[key] = place_cnt.get(key, 0) + 1
                # size bin
                if c < edges[0]:
                    b = 0
                elif c >= edges[-1]:
                    b = nbin - 1
                else:
                    lo, hi = 0, len(edges) - 1
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if edges[mid - 1] <= c:
                            lo = mid
                        else:
                            hi = mid - 1
                    b = lo
                size_cnt[pi + 1][b] += 1
            total_sims += 1

    # probabilities: frequency / total_sims, only history mints, keep top rows per place
    probabilities = []
    for place in range(1, places + 1):
        rows = []
        for (mint, pl), cnt in place_cnt.items():
            if pl == place:
                rows.append((cnt, mint))
        rows.sort(reverse=True)
        rows = rows[:100]
        # renormalise: leave some unnamed mass for newcomers (already implicit via ghosts)
        for cnt, mint in rows:
            p = cnt / total_sims
            if p >= 1e-4:
                probabilities.append({"mint": mint, "place": place, "p": round(p, 6)})

    # sizes: bin frequencies per place, sharpen with SIZE_TEMP, keep meaningful ones
    sizes = []
    for place in range(1, places + 1):
        tot = sum(size_cnt[place])
        if tot == 0:
            continue
        raw = {b: (size_cnt[place][b] / tot) ** SIZE_TEMP
               for b in range(nbin) if size_cnt[place][b] > 0}
        ssum = sum(raw.values())
        rows = sorted(((v / ssum, b) for b, v in raw.items()), reverse=True)
        rows = [(p, b) for p, b in rows if p >= 1e-4][:40]
        s = sum(p for p, _ in rows)
        for p, b in rows:
            sizes.append({"place": place, "bin": b, "p": round(p / s, 6)})

    # top5: by expected rank (highest mean place probability)
    score = {}
    for r in probabilities:
        score[r["mint"]] = score.get(r["mint"], 0.0) + r["p"] * (6 - r["place"])
    top5 = [m for m, _ in sorted(score.items(), key=lambda kv: -kv[1])[:5]]
    if len(top5) < 5:
        for r in last_rows:
            if r["mint"] not in top5 and r["mint"] in hist_mints:
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
    print(f"sigma_mark={sigma_mark:.3f} cand={len(cand)} sims={total_sims} "
          f"prob_rows={len(probabilities)} size_rows={len(sizes)}", flush=True)


if __name__ == "__main__":
    main()
