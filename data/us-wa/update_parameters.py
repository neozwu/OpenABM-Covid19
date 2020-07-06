#!/usr/bin/env python3
"""
Merge baseline, state, and county parameters into county parameter files.
"""

import argparse
import csv
import os
import sys

parser = argparse.ArgumentParser(description="Create County parameter files.")
parser.add_argument("--baseline_parameters", type=str, default="./tests/data/baseline_parameters.csv", help="Baseline parameters file.")
parser.add_argument("--statewide_parameters", type=str, default="./data/us-wa/wa_state_parameters.csv", help="State-specific parameters file. Will overwite baseline values, but can be overwritten by individual county-level parameters.")
parser.add_argument("--household_demographics", type=str, default="./data/us-wa/household_demographics.csv")
parser.add_argument("--output_dir", type=str, default="./data/us-wa/")

HOUSEHOLD_SIZES=[f"household_size_{i}" for i in  range(1,7)]
AGE_BUCKETS=[f"{l}_{h}" for l, h in zip(range(0, 80, 10), range(9, 80, 10))] + ["80"]

def set_unset(d, k, v):
  if k not in d:
    d[k] = v

def county_params_from_state(state_params, county_params):
  tot = sum((county["n_total"] for county in county_params.values()))
  for county in county_params.values():
    scale_factor = tot / county["n_total"]
    set_unset(county, "manual_trace_n_workers", round(state_params["manual_trace_n_workers"] * scale_factor))
    for size in HOUSEHOLD_SIZES:
      set_unset(county, size, round(state_params[size] * scale_factor))
    for bucket in AGE_BUCKETS:
      set_unset(county, size, round(state_params[f"population_{bucket}"] * scale_factor))
    for k in state_params:
      set_unset(county, k, state_params[k])

def main(args):
  with open(args.baseline_parameters, newline="") as f:
    dr = csv.DictReader(f, skipinitialspace=True)
    for row in dr:
      fieldnames = df.fieldnames
      base_params = row
      break

  with open(args.statewide_parameters, newline="") as f:
    if "transpose" in args.statewide_parameters:
      for row in csv.reader(f):
        base_params[row[0]] = row[1]
    else:
      for row in csv.DictReader(f, skipinitialspace=True):
        base_params.update(row)

  with open(os.path.join(args.output_dir, "baseline_parameters.csv"), mode="w", newline="") as of:
    writer = csv.DictWriter(of, fieldnames=fieldnames)
    writer.writeheader()
    for idx, county in enumerate(counties):
      writer.writerow(county_params[county])
      with open(os.path.join(args.output_dir, f"{idx+1}_household_parameters.csv"), mode="w", newline="") as ohf:
        hwriter = csv.DictWriter(ohf, fieldnames=household_fields)
        hwriter.writeheader()
        hwriter.writerows(household_params[county])

if __name__ == "__main__":
  main(parser.parse_args())
