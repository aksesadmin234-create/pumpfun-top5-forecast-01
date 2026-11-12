#!/usr/bin/env python3
import backtest as bt

def run(p):
    return bt.run(p)

if __name__ == "__main__":
    print("=== sigma sweep (ghost fixed) ===")
    for sig in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.80]:
        p = dict(bt.BASE); p["SIGMA_BASE"]=sig; p["SIGMA_FLOOR"]=sig; p["DRIFT_SHRINK"]=0.30
        o,s,a,n = run(p)
        print(f"sigma={sig:.2f} -> order={o:.3f} size={s:.3f} total={o+s:.3f} agree={a:.0f}")

    print("\n=== ghost sweep at sigma=0.35 ===")
    for gr in [0.0, 0.3, 0.55, 1.0, 1.5]:
        for gc in [1.0, 2.0, 3.2, 5.0]:
            p = dict(bt.BASE); p["SIGMA_BASE"]=0.35; p["SIGMA_FLOOR"]=0.35
            p["GHOST_RATE"]=gr; p["GHOST_CAP"]=gc
            o,s,a,n = run(p)
            print(f"ghost_rate={gr:.2f} cap={gc:.1f} -> total={o+s:.3f} order={o:.3f} size={s:.3f}")

    print("\n=== ghost mu/sd at sigma=0.35 rate=0.55 ===")
    for mu in [2.8, 3.0, 3.15, 3.3, 3.5]:
        for sd in [0.4, 0.55, 0.8]:
            p = dict(bt.BASE); p["SIGMA_BASE"]=0.35; p["SIGMA_FLOOR"]=0.35
            p["GHOST_MU"]=mu; p["GHOST_SD"]=sd
            o,s,a,n = run(p)
            print(f"mu={mu:.2f} sd={sd:.2f} -> total={o+s:.3f} order={o:.3f} size={s:.3f}")
