#!/usr/bin/env python3
"""Local backtest for the pumpfun top-5 forecaster.

Uses the practice history marks as a stand-in for answer reads: cut at mark k,
forecast the top 5 at marks k+1..k+n, score with the competition's log score.
Tune parameters here before submitting.
"""
import json, math, random
from datetime import datetime

KIT = "/home/ubuntu/.hermes/scripts/pumpfun-kit/pumpfun-top5-forecast-01-kit"
CHART = 100


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() / 60.0


def load():
    task = json.load(open(KIT + "/input/task.json"))
    lines = [json.loads(l) for l in open(KIT + "/input/history.ndjson") if l.strip()]
    marks = [m for m in lines if m.get("ok") and m.get("rows")]
    return task, marks


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


def build_series(marks):
    series = {}
    for m in marks:
        t = ts(m["at"])
        for r in m["rows"]:
            c = math.log10(int(r["market_cap_lamports"])) - 9.0
            series.setdefault(r["mint"], []).append((t, c))
    for v in series.values():
        v.sort()
    return series


def forecast(marks, reads, places, edges, P):
    series = build_series(marks)
    last_t = max(v[-1][0] for v in series.values())
    nbin = len(edges) + 1

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
        sigma_mark = max(P["SIGMA_FLOOR"], 1.4826 * mad)
    else:
        sigma_mark = P["SIGMA_BASE"]

    cand = []
    for mint, v in series.items():
        t_last, c_last = v[-1]
        drift = 0.0
        if len(v) >= 2:
            j = max(0, len(v) - P["DRIFT_WIN"])
            dt = t_last - v[j][0]
            if dt > 0:
                drift = (c_last - v[j][1]) / dt
        cand.append((c_last, drift, mint))
    cand.sort(reverse=True)
    cand = cand[: P["MAX_CAND"]]

    rng = random.Random(P["SEED"])
    place_cnt = {}
    size_cnt = [[0] * nbin for _ in range(places + 1)]
    total = 0
    for T in reads:
        dt_min = max(0.0, T - last_t)
        sig = sigma_mark * math.sqrt(dt_min / 10.0)
        lam = min(P["GHOST_CAP"], P["GHOST_RATE"] * (max(dt_min, 1.0) / 10.0) ** P["GHOST_POW"])
        for _ in range(P["SIMS"]):
            caps = []
            for c_last, drift, mint in cand:
                adv = drift * dt_min * P["DRIFT_SHRINK"]
                c = c_last + adv + (rng.gauss(0.0, sig) if sig > 0 else 0.0)
                caps.append((c, mint))
            for _g in range(poisson(lam, rng)):
                caps.append((rng.gauss(P["GHOST_MU"], P["GHOST_SD"]), None))
            caps.sort(reverse=True)
            for pi, (c, mint) in enumerate(caps[:places]):
                if mint is not None:
                    place_cnt[(mint, pi + 1)] = place_cnt.get((mint, pi + 1), 0) + 1
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
            total += 1

    probs = []
    PROB_MIN = P.get("PROB_MIN", 0.0)
    MAX_ROWS = P.get("MAX_PROB_ROWS", 100)
    for place in range(1, places + 1):
        rows = sorted(((cnt, m) for (m, pl), cnt in place_cnt.items() if pl == place), reverse=True)[:MAX_ROWS]
        for cnt, m in rows:
            p = cnt / total
            if p >= max(1e-4, PROB_MIN):
                probs.append({"mint": m, "place": place, "p": round(p, 6)})
    sizes = []
    T = P.get("SIZE_TEMP", 1.0)
    for place in range(1, places + 1):
        tot = sum(size_cnt[place])
        if not tot:
            continue
        raw = {b: (size_cnt[place][b] / tot) ** T for b in range(nbin) if size_cnt[place][b] > 0}
        ssum = sum(raw.values())
        rows = sorted(((v / ssum, b) for b, v in raw.items()), reverse=True)
        rows = [(p, b) for p, b in rows if p >= 1e-4][:40]
        s = sum(p for p, _ in rows)
        for p, b in rows:
            sizes.append({"place": place, "bin": b, "p": round(p / s, 6)})
    sc = {}
    for r in probs:
        sc[r["mint"]] = sc.get(r["mint"], 0.0) + r["p"] * (6 - r["place"])
    top5 = [m for m, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:5]]
    return {"probabilities": probs, "top5": top5, "sizes": sizes}


def bin_of(c, edges):
    nbin = len(edges) + 1
    if c < edges[0]:
        return 0
    if c >= edges[-1]:
        return nbin - 1
    lo, hi = 0, len(edges) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if edges[mid - 1] <= c:
            lo = mid
        else:
            hi = mid - 1
    return lo


def score(fc, ans_rows, places, edges):
    """fc: forecast dict; ans_rows: answer mark's rows (rank order)."""
    chart = ans_rows[:CHART]
    chart_mints = [r["mint"] for r in chart]
    chart_set = set(chart_mints)
    order = 0.0
    sizes = 0.0
    agree = 0
    named = {}
    for r in fc["probabilities"]:
        named.setdefault(r["place"], {})[r["mint"]] = r["p"]
    named_size = {}
    for r in fc["sizes"]:
        named_size.setdefault(r["place"], {})[r["bin"]] = r["p"]
    nbin = len(edges) + 1
    for k in range(1, places + 1):
        if k - 1 >= len(chart):
            continue
        actual = chart[k - 1]
        nm = named.get(k, {})
        s_named = sum(nm.values())
        cnt_unnamed = max(1, CHART - len(nm))
        p = nm.get(actual["mint"])
        if p is None:
            p = max(0.0, 1.0 - s_named) / cnt_unnamed
        order += math.log(max(p, 0.0001))
        cap = math.log10(int(actual["market_cap_lamports"])) - 9.0
        b = bin_of(cap, edges)
        ns = named_size.get(k, {})
        s_ns = sum(ns.values())
        cnt_nb = max(1, nbin - len(ns))
        ps = ns.get(b)
        if ps is None:
            ps = max(0.0, 1.0 - s_ns) / cnt_nb
        sizes += math.log(max(ps, 0.0001))
        # agreement
        if actual["mint"] in fc["top5"]:
            j = fc["top5"].index(actual["mint"])
            agree += max(0, 5 - abs(j - (k - 1)))
    return order, sizes, agree


def run(params, verbose=False):
    task, marks = load()
    places = int(task["places"])
    edges = [float(x) for x in task["size"]["edges"]]
    tot_o = tot_s = tot_a = 0.0
    n = 0
    details = []
    for cut in range(1, len(marks) - 1):
        hist = marks[: cut + 1]
        ans = marks[cut + 1 :]
        if not ans:
            continue
        reads = [ts(m["at"]) for m in ans]
        fc = forecast(hist, reads, places, edges, params)
        for a in ans:
            o, s, ag = score(fc, a["rows"], places, edges)
            tot_o += o
            tot_s += s
            tot_a += ag
            n += 1
            details.append((cut, a["at"], o + s, o, s))
    if verbose:
        for d in details:
            print(f"  cut={d[0]} at={d[1][11:16]} total={d[2]:.3f} order={d[3]:.3f} size={d[4]:.3f}")
    return tot_o / n, tot_s / n, tot_a, n


BASE = dict(SIMS=400, DRIFT_WIN=4, DRIFT_SHRINK=0.30, SIGMA_BASE=0.13, SIGMA_FLOOR=0.10,
            GHOST_RATE=0.55, GHOST_POW=0.75, GHOST_CAP=3.2, GHOST_MU=3.15, GHOST_SD=0.55,
            MAX_CAND=160, SEED=20261112)

if __name__ == "__main__":
    o, s, a, n = run(BASE, verbose=True)
    print(f"\nMEAN over {n} read(s): order={o:.4f} size={s:.4f} total={o+s:.4f} agree={a}")
