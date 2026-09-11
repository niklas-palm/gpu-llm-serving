#!/usr/bin/env python3
"""Turn one engine's measured capacity into a fleet size and a cost per million tokens.

    python3 scripts/size_fleet.py --engine-rps 12.8 --demand-rps 70 --price-per-hour 5.85
    python3 scripts/size_fleet.py --engine-rps 12.8 --demand-rps 70 --price-per-hour 2.02 \
        --input-tokens 1000 --output-tokens 190 --headroom 0.2 --gpus-per-instance 1

`--engine-rps` is the requests per second ONE engine sustained inside your latency budget, from
`scripts/benchmark.py` on one instance with your own prompt shape (unique prompts unless your traffic
really shares a prefix). Everything else is arithmetic: instances = demand / (per-instance rate x
(1 - headroom)), rounded up; cost per million tokens = hourly fleet price / tokens served per hour.

The numbers move with the day's price and with the prompt shape; run this with yours, do not copy the
example. Measured capacities to start from are in docs/tuning.md ("Choosing a model to host") and in
measurements/.
"""

from __future__ import annotations

import argparse
import math
import sys


def plan(engine_rps: float, demand_rps: float, price_per_hour: float, gpus_per_instance: int,
         input_tokens: int, output_tokens: int, headroom: float) -> dict:
    per_instance = engine_rps * gpus_per_instance
    usable = per_instance * (1 - headroom)
    instances = max(1, math.ceil(demand_rps / usable - 1e-9))   # 2.1 / 0.7 is 3.0000000000000004 in floats
    fleet_rps = instances * per_instance
    fleet_price = instances * price_per_hour
    tokens_per_hour = demand_rps * 3600 * (input_tokens + output_tokens)
    return {
        "instances": instances,
        "fleet_capacity_rps": fleet_rps,
        "utilisation_at_demand": demand_rps / fleet_rps,
        "fleet_price_per_hour": fleet_price,
        "price_per_million_tokens_at_demand": fleet_price / (tokens_per_hour / 1e6) if tokens_per_hour else 0.0,
        "price_per_million_tokens_saturated": fleet_price / (fleet_rps * 3600 * (input_tokens + output_tokens) / 1e6),
        "price_per_1000_requests_at_demand": fleet_price / (demand_rps * 3.6) if demand_rps else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-rps", type=float, required=True,
                    help="requests/s one engine held inside the latency budget (scripts/benchmark.py)")
    ap.add_argument("--demand-rps", type=float, required=True, help="steady-state requests/s to serve")
    ap.add_argument("--price-per-hour", type=float, required=True, help="price of one instance per hour")
    ap.add_argument("--gpus-per-instance", type=int, default=1, help="engines per instance (one per GPU)")
    ap.add_argument("--input-tokens", type=int, default=1000, help="average input tokens per request")
    ap.add_argument("--output-tokens", type=int, default=190, help="average output tokens per request")
    ap.add_argument("--headroom", type=float, default=0.15,
                    help="fraction of capacity kept free for bursts and a lost instance (0.15 = 15%%)")
    a = ap.parse_args()
    if (min(a.engine_rps, a.demand_rps, a.price_per_hour) <= 0 or a.gpus_per_instance < 1 or not 0 <= a.headroom < 1
            or min(a.input_tokens, a.output_tokens) < 0 or a.input_tokens + a.output_tokens < 1):
        sys.exit("rates and price above 0, gpus-per-instance at least 1, headroom in [0, 1), tokens at least 1 in total")
    p = plan(a.engine_rps, a.demand_rps, a.price_per_hour, a.gpus_per_instance, a.input_tokens, a.output_tokens, a.headroom)
    print(f"one engine {a.engine_rps:g} req/s x {a.gpus_per_instance} per instance, {a.headroom:.0%} headroom, "
          f"demand {a.demand_rps:g} req/s at {a.input_tokens}+{a.output_tokens} tokens\n")
    print(f"  instances needed             {p['instances']}")
    print(f"  fleet capacity               {p['fleet_capacity_rps']:.1f} req/s ({p['utilisation_at_demand']:.0%} used at demand)")
    print(f"  fleet price                  ${p['fleet_price_per_hour']:,.2f} per hour")
    print(f"  per 1,000 requests           ${p['price_per_1000_requests_at_demand']:.3f} at demand")
    print(f"  per million tokens           ${p['price_per_million_tokens_at_demand']:.3f} at demand, "
          f"${p['price_per_million_tokens_saturated']:.3f} if saturated")
    print("\nA fixed fleet at the minimum for steady state; autoscaling is burst insurance on top "
          "(docs/tuning.md, 'Sizing a fleet, and when to autoscale').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
