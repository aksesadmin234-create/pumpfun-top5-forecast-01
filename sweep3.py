#!/usr/bin/env python3
import backtest as bt

def run(p):
    return bt.run(p)

if __name__ == "__main__":
    print("=== fine sigma x ghost_rate (ghost_mu=3.30 sd=0.45) ===")
    best=None
    for sig in [0.38, 0.42, 0.46, 0.50, 0.55]:
        for gr in [0.8, 1.1, 1.4, 1.8]:
            p = dict(bt.BASE); p["SIGMA_BASE"]=sig; p["SIGMA_FLOOR"]=sig
            p["GHOST_RATE"]=gr; p["GHOST_MU"]=3.30; p["GHOST_SD"]=0.45; p["GHOST_CAP"]=4.0
            p["SIMS"]=800
            o,s,a,n = run(p)
            t=o+s
            tag=""
            if best is None or t>best[0]: best=(t,sig,gr,o,s); tag=" <-- best"
            print(f"sigma={sig:.2f} gr={gr:.1f} -> total={t:.3f} order={o:.3f} size={s:.3f}{tag}")
    print("\nbest:",best)
