"""
Utility functions created for running multi-county simulations.

Created: 25 June 2020
Author: mattea
"""
from matplotlib import pyplot as plt
import argparse
import collections
import csv
import glob
import example_utils as utils
import itertools
import math
import pandas as pd
import numpy as np
import os
import sys

def relative_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), filename)

parser = argparse.ArgumentParser(description="Run County Simluations.")
parser.add_argument("--statewide_parameters", type=str, default=relative_path("../data/us-wa/wa_state_parameters_transpose.csv"), help="State-specific parameters file. Will overwite baseline values, but can be overwritten by individual county-level parameters. Reads as transpose file if 'transpose' is present in file name.")
parser.add_argument("--county_parameters", type=str, default=relative_path("../data/us-wa/wa_county_parameters.csv"), help="County-specific parameters file(s). Will overwite baseline and state values. Expects an extra column of county_fips to designate which county this refers to.")
parser.add_argument("--household_demographics", type=str, default=relative_path("../data/us-wa/wa_county_household_demographics.csv"), help="County-specific household demographics file(s). Expects an extra column of county_fips_code to designate which county this refers to.")
parser.add_argument("--occupations", type=str, default=relative_path("../data/us-wa/wa_county_occupation_networks.csv"), help="County-specific household demographics file(s). Expects an extra column of county_fips_code to designate which county this refers to.")
parser.add_argument("--study_params", type=str, default=None, help="Optional. Parameter file with one set of overrides per line. If an extra column of \'study_name\" is used, will be used for writing results, otherwise the line number will be used.")
parser.add_argument("--output_dir", type=str, default="results/us-wa/")
parser.add_argument("--counties", type=str, default=None, help="Optional. If specified, only specified counties will be processed (comma-separated list).")

DEFAULT_ARGS = parser.parse_args([])

LOCAL_DEFAULT_PARAMS = {
    "county_fips": "00000",
    "lockdown_days": 35,
    "app_users_fraction": 0.8,
    "custom_occupation_network": 1,
    "study_name": "0",
    "Index": 0,
}
HOUSEHOLD_SIZES=[f"household_size_{i}" for i in  range(1,7)]
AGE_BUCKETS=[f"{l}_{h}" for l, h in zip(range(0, 80, 10), range(9, 80, 10))] + ["80"]

def bucket_to_age(b):
  return int(b.split("_")[1]) // 10

def set_app_user_fraction(params, frac):
    name = "app_users_fraction"
    for b in age_buckets:
      bucket_name = f"{name}_{b}"
      params.set_param(bucket_name, params.get_param(bucket_name) * frac)

def read_param_file(file_name):
  params = {}
  pf = pd.read_csv(file_name,skipinitialspace=True, comment="#")
  if "transpose" in file_name:
    for row in pf.itertuples():
      params[row.parameter_name] = row.parameter_value
  else:
    for row in pf.itertuples():
      params.update(row.as_dict())
  return params

def read_param_files(file_names):
  params = {}
  for f in file_names:
    params.update(read_param_file(f))
  return params

def set_unset(d, k, v):
  if k not in d:
    d[k] = v

def county_params_from_state(state_params, county_params):
  """
    Sets params from state data if not set in county data.
  """
  scale_factor = county_params["n_total"] / county_params.n_total.sum()
  if "manual_trace_n_workers" not in county_params.columns:
    county_params["manual_trace_n_workers"] = (scale_factor * state_params["manual_trace_n_workers"]).round()
  for hs in HOUSEHOLD_SIZES:
    if hs in state_params and hs not in county_params.columns:
      county_params[hs] = (scale_factor * state_params[hs]).round()
  for bucket in AGE_BUCKETS:
    ps = f"population_{bucket}"
    if ps in state_params and ps not in county_params.columns:
      county_params[ps] = (scale_factor * state_params[ps]).round()
  for k, v in state_params.items():
    if k not in county_params.columns:
      county_params[k] = v

def build_population(params_dict, houses):
  n_total = params_dict["n_total"]
  IDs = np.arange( n_total, dtype='int32')
  house_no = np.zeros( IDs.shape, dtype='int32' )
  ages = np.zeros( IDs.shape, dtype='int32' )
  idx = 0
  house_idx = 0
  for house in houses.itertuples():
    house_pop = 0
    for col in houses.columns:
      if col.startswith("a_"):
        age = bucket_to_age(col)
        cnt = int(getattr(house,col))
        if cnt == 0:
          continue
        house_pop += cnt
        house_no[idx:idx+cnt] = house_idx
        ages[idx:idx+cnt] = age
        idx += cnt
    if house_pop != 0:
      house_idx += 1
    if house_pop >= n_total:
      break
  return pd.DataFrame({'ID':IDs, 'age_group':ages, 'house_no':house_no})

def sim_plot(axs, df, label, n_total, time_offset):
  """Plot the simulation result to the given axes"""
  diff = df.diff()
  df['new_infected'] = diff["total_infected"]
  df['new_death'] = diff["n_death"]
  df["total_infected_rate"] = df["total_infected"] / n_total
  df["quarantine_rate"] = df["n_quarantine"] / n_total
  df["time (days))"] = df["time"] - time_offset
  df.plot( ax=axs[0], x = "time (days))", y = "new_infected", label=label, legend=False)
  df.plot( ax=axs[1], x = "time (days))", y = "total_infected_rate", label=label, legend=False )
  df.plot( ax=axs[2], x = "time (days))", y = "n_hospital", label=label, legend=False)
  df.plot( ax=axs[3], x = "time (days))", y = "n_tests", label=label, legend=False)
  df.plot( ax=axs[4], x = "time (days))", y = "quarantine_rate", label=label, legend=False)

def common_prefix(labels):
  psubstr = ""
  while True:
    idx = labels[0].find(" ", len(psubstr)) + 1
    if idx == 0:
      return psubstr
    substr = labels[0][:idx]
    for label in labels[1:]:
      if substr != label[:idx]:
        return psubstr
    psubstr = substr

def sim_display(results, labels):
  """Display the simulation results."""
  fig, axs = plt.subplots(2,3,figsize=(12,7.2),constrained_layout=True)
  axs = axs.flatten()
  axs[5].remove()
  axs[0].set_title('new infected')
  axs[1].set_title('total infected percentage')
  axs[2].set_title('total in hospital')
  axs[3].set_title('tests per day')
  axs[4].set_title('percent in quarantine')
  label_prefix = common_prefix(labels)
  if label_prefix:
    labels = [label[len(label_prefix):] for label in labels]

  for result, label in zip(results,labels):
      param_model, result = result
      sim_plot(axs, result, label, param_model['n_total'], param_model['time_offset'])
  axs[2].legend(bbox_to_anchor=(0, -0.2),loc='upper left',
                  title=label_prefix, ncol=math.ceil(len(results) / 12)).set_in_layout(False)

def read_county_occupation_network( county_fips, all_occupations ):
  sector_df = all_occupations[all_occupations.area_fips == county_fips]
  sector_name = sector_df['industry_code'].values
  sector_size = sector_df['qrtly_emplvl'].values
  sector_pdf = sector_size / np.sum( sector_size )
  return list( sector_name ), list( sector_pdf )

def get_mean_work_interaction( params, age_type ):
  if age_type == 0:
    return params.get_param( 'mean_work_interactions_child' )
  elif age_type == 1:
    return params.get_param( 'mean_work_interactions_adult' )
  elif age_type == 2:
    return params.get_param( 'mean_work_interactions_elderly' )
  else:
    raise ValueError( 'not supported age type' )

def get_lockdown_multipliers( params, n_networks ):
  # For now, just use default lockdown_multipliers.
  lockdown_multiplier = np.ones( n_networks ) * params.get_param('lockdown_occupation_multiplier_working_network')
  lockdown_multiplier[0] = params.get_param('lockdown_occupation_multiplier_primary_network')
  lockdown_multiplier[1] = params.get_param('lockdown_occupation_multiplier_secondary_network')
  lockdown_multiplier[-2] = params.get_param('lockdown_occupation_multiplier_retired_network')
  lockdown_multiplier[-1] = params.get_param('lockdown_occupation_multiplier_elderly_network')
  return lockdown_multiplier

def build_occupation_networks( params, sector_names, n_child_network=2, n_elderly_network=2 ):
  n_networks = n_child_network + len( sector_names ) + n_elderly_network
  network_no = np.arange( n_networks, dtype='int32' )
  network_names = ['primary', 'secondary'] + sector_names + ['retired', 'elderly']
  age_types = np.ones( n_networks )
  age_types[:2] = 0 # child
  age_types[-2:] = 2 # elderly
  mean_work_interactions = map( lambda x: get_mean_work_interaction(params, x), 
                                age_types )
                                              
  return pd.DataFrame({
      'network_no': network_no,
      'age_type': age_types,
      'mean_work_interaction': mean_work_interactions,
      'lockdown_multiplier': get_lockdown_multipliers( params, n_networks ),
      'network_id' : network_no,
      'network_name': network_names
  })

def build_occupation_assignment(household_df, network_df, network_pdf):
  IDs = household_df.ID.values
  age_groups = household_df.age_group.values

  # assign adult workplace networks
  sector_ids = network_df[network_df.age_type == 1].network_no.values
  assignment = np.random.choice(sector_ids, len(IDs), p=network_pdf)

  # assign child and elderly networks
  assignment[age_groups == 0] = 0 # primary_network
  assignment[age_groups == 1] = 1 # secondary_network
  assignment[age_groups == 7] = np.max(sector_ids) + 1 # retired_network
  assignment[age_groups == 8] = np.max(sector_ids) + 2 # elderly_network

  return pd.DataFrame({'ID': IDs, 'network_no': assignment})

def run_model(params_dict, houses, sector_names, sector_pdf):
  pt = LOCAL_DEFAULT_PARAMS.copy()
  pt.update(params_dict)
  params_dict = pt

  total_days_left = int(params_dict['end_time'])
  app_turned_on = "app_turned_on" in params_dict and params_dict["app_turned_on"]

  params  = utils.get_baseline_parameters()
  for p, v in params_dict.items():
    if p in LOCAL_DEFAULT_PARAMS:
      continue
    if isinstance(v, np.int64) or (hasattr(v, "is_integer") and v.is_integer()):
      params.set_param( p, int(v) )
    else:
      params.set_param( p, v )

  if "app_user_fraction" in params_dict:
    set_app_user_fraction(params, params_dict["app_user_fraction"])
  
  hh_df = build_population( params_dict, houses )
  params.set_demographic_household_table( hh_df )
  if params_dict["custom_occupation_network"]:
    occupation_network_df = build_occupation_networks( params, 
                                                       sector_names )
  
    occupation_assignment = build_occupation_assignment( hh_df, 
                                                         occupation_network_df, 
                                                         sector_pdf )
    params.set_occupation_network_table( occupation_assignment, 
                                         occupation_network_df )

  model = utils.get_simulation( params ).env.model

  m_out = []
  model.one_time_step()
  m_out.append(model.one_time_step_results())
  total_days_left -= 1
  while m_out[-1]["total_infected"] < params_dict["n_total"] * 0.01:
      model.one_time_step()
      m_out.append(model.one_time_step_results())
      total_days_left -= 1

  params_dict["time_offset"] = params_dict['end_time'] - total_days_left

  model.update_running_params("lockdown_on", 1)
          
  for step in range(total_days_left):
      if step == params_dict['lockdown_days']:
          model.update_running_params("lockdown_on", 0)
          if app_turned_on == 1:
              model.update_running_params("app_turned_on", 1)
          
      model.one_time_step()
      m_out.append(model.one_time_step_results())

  df = pd.DataFrame( m_out )
  m_out = []
  del model
  return params_dict, df

def run_counties(county_params, all_households, all_occupations, params_files=[], params_overrides={}, counties=None):
  if counties is None:
    counties = county_params.county_fips.unique()
  base_params = read_param_files(params_files)
  outputs = []
  for county in counties[:1]:
    params = base_params.copy()
    params.update(county_params[county_params["county_fips"] == county].iloc[0])
    params.update(params_overrides)
    households = all_households[all_households["county_fips"] == county]
    sector_names, sector_pdf = read_county_occupation_network(county, all_occupations)
    outputs.append(run_model(params, households, sector_names, sector_pdf))
  return outputs

def remove_nans(d):
  d = d.copy()
  for k in list(d.keys()):
    try:
      if np.isnan(d[k]):
        del d[k]
    except TypeError:
      pass
  return d

class Network(object):
  def __init__(self, households, sector_names, sector_pdf):
    self.households = households
    self.sector_names = sector_names
    self.sector_pdf = sector_pdf

class AggregateModel(object):
  def __init__(self,
               param_files=[DEFAULT_ARGS.statewide_parameters],
               households_file=DEFAULT_ARGS.household_demographics,
               occupations_file=DEFAULT_ARGS.occupations,
               county_params_file=DEFAULT_ARGS.county_parameters,
               params_overrides={},
               run_parallel=True):
    self.params = read_param_files(param_files)
    self.params.update(params_overrides)
    self.households = pd.read_csv(households_file, skipinitialspace=True, comment="#")
    self.county_params = pd.read_csv(county_params_file, index_col="county_fips", skipinitialspace=True, comment="#")
    self.occupations = pd.read_csv(occupations_file, skipinitialspace=True, comment="#")
    self.counties = self.county_params.index.unique()
    county_params_from_state(self.params, self.county_params)
    self.run_parallel = run_parallel
    self.results = {}
    self.merged_results = None

  def run_county(self, county_fips, params_overrides={}):
    sector_names, sector_pdf = read_county_occupation_network(county_fips, self.occupations)
    households = self.households[self.households["county_fips"] == county_fips]
    params = self.county_params.loc[county_fips].to_dict()
    params.update(params_overrides)
    return run_model(params, households, sector_names, sector_pdf)

  def get_county_params(self, county_fips, params_overrides={}):
    sector_names, sector_pdf = read_county_occupation_network(county_fips, self.occupations)
    households = self.households[self.households["county_fips"] == county_fips]

    params_dict = LOCAL_DEFAULT_PARAMS.copy()
    params_dict.update(self.county_params.loc[county_fips].to_dict())
    params_dict.update(params_overrides)

    params  = utils.get_baseline_parameters()
    for p, v in params_dict.items():
      if p in LOCAL_DEFAULT_PARAMS:
        continue
      if isinstance(v, np.int64) or (hasattr(v, "is_integer") and v.is_integer()):
        params.set_param( p, int(v) )
      else:
        params.set_param( p, v )

    if "app_user_fraction" in params_dict:
      set_app_user_fraction(params, params_dict["app_user_fraction"])
    
    hh_df = build_population( params_dict, households )
    params.set_demographic_household_table( hh_df )
    if params_dict["custom_occupation_network"]:
      occupation_network_df = build_occupation_networks( params, sector_names )
    
      occupation_assignment = build_occupation_assignment( hh_df, occupation_network_df, sector_pdf )
      params.set_occupation_network_table( occupation_assignment, occupation_network_df )
    return params

  def run_counties(self, counties_fips, params_overrides={}):
    params_overrides = remove_nans(params_overrides)
    if self.run_parallel:
      from tqdm import tqdm, trange
      from concurrent.futures import ProcessPoolExecutor

      with ProcessPoolExecutor() as ex:
        outputs = list(
          tqdm(
            ex.map(self.run_county, counties_fips, itertools.repeat(params_overrides, len(counties_fips))),
            total=len(counties_fips),
            desc="Batch progress"
          )
        )
        self.results = dict(zip(counties_fips, outputs))
    else:
      for county in counties_fips:
        self.results[county] = self.run_county(county)
    return self.results

  def run_all(self, params_overrides={}):
    self.run_counties(self.counties, params_overrides)
    return self.results

  def merge_results(self):
    merged_results = pd.concat([result[1] for result in self.results.values()]).groupby(["time"]).sum().reset_index()
    merged_params = next(iter(self.results.values()))[0].copy()
    merged_params["n_total"] = sum((result[0]["n_total"] for result in self.results.values()))
    merged_params["time_offset"] = 0
    self.merged_results = [merged_params, merged_results]
    return self.merged_results

  def plot_results(self):
    if not self.results:
      return
    if not self.merged_results:
      self.merge_results()
    sim_display([self.merged_results], ["AggregateModel"])

  def write_results(self, output_dir, write_merged=True):
    if not self.results:
      return
    try:
      os.makedirs(output_dir)
    except FileExistsError:
      pass

    for county, output in self.results.items():
      params, result = output
      pd.DataFrame(params, index=[0]).to_csv(os.path.join(output_dir, f"{county}_params.csv"), index=False)
      result.to_csv(os.path.join(output_dir, f"{county}_results.csv"), index=False)

    if not write_merged:
      return
    if not self.merged_results:
      self.merge_results()
    params, result = self.merged_results
    pd.DataFrame(params, index=[0]).to_csv(os.path.join(output_dir, f"merged_params.csv"), index=False)
    result.to_csv(os.path.join(output_dir, f"merged_results.csv"), index=False)


def read_results(base_dir, sim_names=None):
  if sim_names is None:
    paths = glob.glob(os.path.join(base_dir, "*/merged_results.csv"))
    sim_names = [os.path.basename(os.path.dirname(path)) for path in paths]

  results = []
  for sim in sim_names:
    params = pd.read_csv(os.path.join(base_dir, f"{sim}/merged_params.csv")).loc[0].to_dict()
    result = pd.read_csv(os.path.join(base_dir, f"{sim}/merged_results.csv"))
    results.append([params, result])

  return results, sim_names


def main(args):
  output_dir = args.output_dir
  if args.counties:
    counties = [int(c.strip()) for c in args.counties.split(',')]

  if args.study_params:
    overrides = pd.read_csv(args.study_params, comment="#")
    if "study_name" not in overrides:
      overrides["study_name"] = [f"{i}" for i in range(len(overrides))]
  else:
    overrides = pd.DataFrame({"study_name": ["0"]})
  state_param_file = args.statewide_parameters
  households_file = args.household_demographics
  occupations_file = args.occupations
  county_params_file = args.county_parameters

  model = AggregateModel([state_param_file], households_file, occupations_file, county_params_file)
  for override in overrides.itertuples():
    if counties:
      model.run_counties(counties, override._asdict())
    else:
      model.run_all(override._asdict())
    if output_dir:
      model.write_results(os.path.join(output_dir, override.study_name))

if __name__ == "__main__":
  main(parser.parse_args())
