# Lint as: python3
"""Write county-specific parameters to county param file.
"""

import argparse
import os

import itertools
import pandas as pd

def relative_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), filename)

parser = argparse.ArgumentParser(description="Generate county-specific parameters.")
parser.add_argument("--households_file", type=str, default=relative_path("../data/us-wa/wa_county_household_demographics.csv"))
parser.add_argument("--output_file", type=str, default=relative_path("../data/us-wa/wa_county_parameters.csv"))

mobility_glm = [
    0.6695635, 0.46162257, 0.532143, 0.6348413, 0.51558036, 0.48932594,
    0.516369, 0.6522389, 0.61336815, 0.5563802, 0.59022725, 0.5765775,
    0.3916168, 0.35938537, 0.48781577, 0.52317595, 0.4550434, 0.5110669,
    0.48891503, 0.4618209, 0.48036945, 0.33749515, 0.24225044, 0.20447512,
    0.20293882, 0.16157931, 0.14307933, 0.13991378, 0.16423374, 0.15298668,
    0.17763957, 0.21125548, 0.1803969, 0.16467823, 0.17971474, 0.17753959,
    0.23082298, 0.24066994, 0.27494055, 0.28916162, 0.2248088, 0.20942202,
    0.1871627, 0.23500676, 0.25223547, 0.269152, 0.30410707, 0.2500342,
    0.20626648, 0.26878333, 0.29226655, 0.23681639, 0.18180531, 0.22707182,
    0.22352692, 0.22715293, 0.28314465, 0.24651927, 0.29080024, 0.28779477,
    0.27488065, 0.30428088, 0.21379879, 0.2934495, 0.28768358, 0.46316183,
    0.3261379, 0.48078573, 0.50866747, 0.68582726, 0.6859936, 0.32802516,
    0.29141086, 0.35144496, 0.3450396, 0.44209433, 0.31643462, 0.43847868,
    0.3764527, 0.38416898, 0.33781767, 0.3604901, 0.41484457, 0.6701375,
    0.811031, 0.3250221, 0.5161743, 0.6503538, 0.6024895, 0.59661555,
    0.30728325, 0.5039729, 0.5274018, 0.48070994, 0.5910089, 0.5676799,
    0.5196237, 0.5648359, 0.57741904, 0.4517982, 0.35300913, 0.5808511,
    0.58153296, 0.43211246, 0.51997614, 0.75426483, 0.38321602, 0.55460995,
    0.7359332, 0.8361097, 0.8358396, 0.7294188, 0.93402165, 0.7875712, 0.803308,
    0.74414486, 0.8540179, 0.77662456, 0.70934975, 0.77900165, 0.68099344,
    0.5967497, 0.6122558, 0.71269685, 0.81963265, 0.79037124, 0.98989326,
    0.71127105, 0.6018918, 0.7285367, 0.674034, 0.7988311
]

changepoint_rate = [
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.9999999, 0.99999917,
    0.9999918, 0.99993706, 0.99956065, 0.9967578, 0.98093027, 0.9192135,
    0.8291116, 0.7511561, 0.7000672, 0.6582161, 0.62701786, 0.60616887,
    0.58921355, 0.5760716, 0.566327, 0.5590012, 0.5562643, 0.553969, 0.55310875,
    0.55243456, 0.5520327, 0.55177236, 0.55177236, 0.5517309, 0.551372,
    0.55112934, 0.55112934, 0.55112934, 0.5510583, 0.5510425, 0.5509847,
    0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201,
    0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201,
    0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201,
    0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201,
    0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201, 0.5509201,
    0.5509201, 0.5509201, 0.5509201, 0.550797, 0.550797, 0.550797, 0.550797,
    0.550797, 0.550797, 0.550797, 0.55070627, 0.55063486, 0.55063486, 0.5505831,
    0.5505831
]

all_params = [{
    "county_fips": 53033, # King
    "n_seed_infection": 30,
    "infectious_rate": 4.7,
    "seeding_date_delta": 32,
    "lockdown_scalars": mobility_glm,
    "changepoint_scalars": changepoint_rate,
},{
    "county_fips": 53061, # Snohomish
    "n_seed_infection": 30,
    "infectious_rate": 4.8,
    "seeding_date_delta": 25,
    "lockdown_scalars": mobility_glm,
    "changepoint_scalars": changepoint_rate,
},{
    "county_fips": 53053, # Pierce
    "n_seed_infection": 30,
    "infectious_rate": 5.0,
    "seeding_date_delta": 14,
    "lockdown_scalars": mobility_glm,
    "changepoint_scalars": changepoint_rate,
},
]

HOUSEHOLD_SIZES=[f"household_size_{i}" for i in  range(1,7)]
AGE_BUCKETS=[f"{l}_{h}" for l, h in zip(range(0, 80, 10), range(9, 80, 10))] + ["80"]

def stringify(l):
  return ",".join((f"{x}" for x in l))

def process_scalars(scalars):
  l_rescaled = [s / scalars[0] for s in scalars]
  l_rescaled.append(pd.Series(scalars).rolling(7,1).mean().to_list()[-1])
  return stringify(l_rescaled)

def household_sizes(households_file):
  households = pd.read_csv(households_file, skipinitialspace=True, comment="#")
  pops = households.groupby("county_fips").sum()
  pops = pops.rename(columns=lambda l: l.replace("a_","population_"))
  pops["n_total"] = pops.sum(axis=1)

  house_sizes = households.set_index("county_fips").sum(axis=1).apply(lambda x: x if x <= 6 else 6)
  houses_df = pd.DataFrame(house_sizes, columns=["sizes"])
  houses_df["ones"] = 1
  house_counts = houses_df.pivot_table(index=houses_df.index,columns="sizes",aggfunc='count')['ones'].rename(columns=lambda l:f"household_size_{l}")
  house_counts.columns.name = None
  house_counts.drop('household_size_0', axis=1, inplace=True)
  return pops.join(house_counts)

def main(args):
  for p in all_params:
    p["lockdown_scalars"] = process_scalars(p["lockdown_scalars"])
    p["changepoint_scalars"] = process_scalars(p["changepoint_scalars"])
  pd.DataFrame(all_params).set_index("county_fips").join(household_sizes(args.households_file), how="right").to_csv(args.output_file)

if __name__ == '__main__':
  main(parser.parse_args())
