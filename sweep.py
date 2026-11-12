#!/usr/bin/env python3
"""Parameter sweep vs the house baseline on the same backtest cuts."""
import backtest as bt

def baseline_forecast(hist, reads, places, edges, chart_last):
    # persistence: P_STAY 0.3 on #k, 0.1 on neighbours, rest unnamed; sizes normal sd 0.35
    import math
    rows = sorted(hist[-1]["rows"], key=lambda r: r["rank"])
    probs, sizes = [], []
    def size_of(r): return math.log10(int(r["market_cap_lamports"])) - 9.0
    def erfc(x):
        z = abs(x); t = 1/(1+0.5*z)
        r = t*math.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))))
        return r if x >= 0 else 2-r
    def above(x, mean, sd):
        if x == -math.inf: return 1.0
        if x == math.inf: return 0.0
        return 0.5*erfc((x-mean)/(sd*math.sqrt(2)))
    for place in range(1, places+1):
        probs.append({"mint": rows[place-1]["mint"], "place": place, "p": 0.30})
        if place > 1: probs.append({"mint": rows[place-2]["mint"], "place": place, "p": 0.10})
        if place < len(rows): probs.append({"mint": rows[place]["mint"], "place": place, "p": 0.10})
        mean = size_of(rows[place-1])
        for b in range(len(edges)+1):
            lo = -math.inf if b == 0 else edges[b-1]
            hi = math.inf if b == len(edges) else edges[b]
            u = math.floor((above(lo, mean, 0.35) - above(hi, mean, 0.35))*1e6)
            if u > 0: sizes.append({"place": place, "bin": b, "p": u/1e6})
    return {"probabilities": probs, "top5": [rows[i]["mint"] for i in range(min(5, len(rows)))], "sizes": sizes}


def run_baseline():
    task, marks = bt.load()
    places = int(task["places"]); edges = [float(x) for x in task["size"]["edges"]]
    to = ts_ = ta = 0.0; n = 0
    for cut in range(1, len(marks)-1):
        hist = marks[:cut+1]; ans = marks[cut+1:]
        if not ans: continue
        fc = baseline_forecast(hist, [], places, edges, None)
        for a in ans:
            o, s, ag = bt.score(fc, a["rows"], places, edges)
            to += o; ts_ += s; ta += ag; n += 1
    return to/n, ts_/n, ta, n


if __name__ == "__main__":
    o, s, a, n = run_baseline()
    print(f"BASELINE mean over {n}: order={o:.4f} size={s:.4f} total={o+s:.4f} agree={a}")
    # sweep sigma and drift shrink
    for sig in [0.0, 0.05, 0.08, 0.10, 0.13, 0.18, 0.25]:
        for ds in [0.0, 0.15, 0.30, 0.50]:
            p = dict(bt.BASE); p["SIGMA_BASE"] = sig; p["SIGMA_FLOOR"] = sig; p["DRIFT_SHRINK"] = ds
            o, s, a, n = bt.run(p)
            print(f"sigma={sig:.2f} shrink={ds:.2f} -> order={o:.3f} size={s:.3f} total={o+s:.3f} agree={a:.0f}")
