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
import itertools
import math
import pandas as pd
import numpy as np
import os
import sys
import covid19
from tqdm import tqdm, trange
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from google.cloud import storage

from COVID19.model import Model, Parameters, ModelParameterException
import COVID19.simulation as simulation



def relative_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), filename)

def remove_nans(d):
  d = d.copy()
  for k in list(d.keys()):
    try:
      if np.isnan(d[k]):
        del d[k]
    except (TypeError, ValueError):
      pass
  return d

parser = argparse.ArgumentParser(description="Run County Simluations.")
parser.add_argument("--statewide_parameters", type=str, default=relative_path("../data/us-wa/wa_state_parameters_transpose.csv"), help="State-specific parameters file. Will overwite baseline values, but can be overwritten by individual county-level parameters. Reads as transpose file if 'transpose' is present in file name.")
parser.add_argument("--county_parameters", type=str, default=relative_path("../data/us-wa/wa_county_parameters.csv"), help="County-specific parameters file(s). Will overwite baseline and state values. Expects an extra column of county_fips to designate which county this refers to.")
parser.add_argument("--household_demographics", type=str, default=relative_path("../data/us-wa/wa_county_household_demographics.csv"), help="County-specific household demographics file(s). Expects an extra column of county_fips_code to designate which county this refers to.")
parser.add_argument("--occupations", type=str, default=relative_path("../data/us-wa/wa_county_occupation_networks.csv"), help="County-specific household demographics file(s). Expects an extra column of county_fips_code to designate which county this refers to.")
parser.add_argument("--output_dir", type=str, default="results/us-wa/")
parser.add_argument("--counties", type=str, default=None, help="Optional. If specified, only specified counties will be processed (comma-separated list).")
parser.add_argument("--gcs_path", type=str, default=None)
parser.add_argument("--study_params", type=str, default=relative_path("../data/us-wa/simulations.csv"), help="Optional. Parameter file with one set of overrides per line. If an extra column of \'study_name\" is used, will be used for writing results, otherwise the line number will be used.")
parser.add_argument("--study_line", type=int, default=None, help="Optional. The line (0-indexed) of the study_params to run. Requires study_params to be specified. If omitted, all studies are run.")

DEFAULT_ARGS = parser.parse_args([])

LOCAL_DEFAULT_PARAMS = {
    "county_fips": "00000",
    "lockdown_days": 35,
    "app_turned_on": 0,
    "manual_trace_on": 0,
    "manual_trace_n_workers_per_100k": None,
    "app_users_fraction": 0.8,
    "custom_occupation_network": 1,
    "use_default_lockdown_multiplier": True,
    "use_default_work_interaction": False,
    "study_name": "0",
    "Index": 0,
    "lockdown_scalars": "",
    "changepoint_scalars": "",
    "seeding_date_delta": 0,
    "mobility_on": 0, # No effect yet. TODO(mattea): Implement once we have values for all counties
    "changepoint_on": 0, # No effect yet. TODO(mattea): Implement once we have values for all counties
    "mobility_scale_all": 0,
    "static_mobility_scalar": 0,
    "iteration": 0,
    "mobility_start_date": "2020-03-01",
}
HOUSEHOLD_SIZES=[f"household_size_{i}" for i in  range(1,7)]
AGE_BUCKETS=[f"{l}_{h}" for l, h in zip(range(0, 80, 10), range(9, 80, 10))] + ["80"]

# adjust lockdown_multiplier based on:
# https://bfi.uchicago.edu/key-economic-facts-about-covid-19/#shutdown-sectors
HIGH_IMPACT_LOCKDOWN_SECTORS = [
  '72', # Accommodation and food services
  '48-49', # Transportation and warehousing
  '62', # Healthcare and social assistance
  '71', # Arts, entertainment and recreation
  '44-45', # Retail trade
  '53', # Real estate and rental and leasing
  '42', # Wholesale trade
  '31-33', # Manufacturing
]

# adjust mean_work_interactions based on:
# https://www.doh.wa.gov/Portals/1/Documents/1600/coronavirus/covid_occupation_industry_summary_2020-06-12.pdf
WORK_INTERACTION_ADJUST_RATIO = {
  '11':  0.06 / 0.03, # Agriculture, forestry, fishing and hunting
  '21':  1.0, # Mining, quarrying, and oil and gas extraction
  '22':  0.01 / 0.01, # Utilities
  '23':  0.06 / 0.06,  # Construction
  '31-33': 0.09 / 0.09, # Manufacturing
  '42':  0.01 / 0.04, # Wholesale trade
  '44-45':  0.08 / 0.12, # Retail trade
  '48-49':  0.05 / 0.04, # Transportation and warehousing
  '51':  0.01 / 0.04, # Information
  '52':  0.02 / 0.03, # Finance and insurance
  '53':  0.01 / 0.02, # Real estate and rental and leasing
  '54':  0.03 / 0.06, #  Professional and technical services
  '55':  0.01 / 0.02, # Management of companies and enterprises
  '56':  0.04 / 0.05, # Administrative and waste services
  '61':  0.03 / 0.09, # Educational services
  '62':  0.37 / 0.13, # Health care and social assistance
  '71':  0.01 / 0.02,  # Arts, entertainment, and recreation
  '72':  0.07  / 0.09, # Accommodation and food services
  '81':  0.02 / 0.03, # Other services, except public administration
  '99':  1.0, # Unclassified
}

input_parameter_file = relative_path("../tests/data/baseline_parameters.csv")
parameter_line_number = 1
output_dir = "."
household_demographics_file = relative_path("../tests/data/baseline_household_demographics.csv")
hospital_file = relative_path("../tests/data/hospital_baseline_parameters.csv")

BASE_PARAMS = collections.defaultdict(str)
BASE_PARAMS.update(remove_nans(pd.read_csv(input_parameter_file).loc[0].to_dict()))

def get_baseline_parameters():
    params = Parameters(input_parameter_file, parameter_line_number, output_dir, household_demographics_file, hospital_file)
    return params

def get_simulation( params ):
    model = simulation.COVID19IBM(model = Model(params))
    sim = simulation.Simulation(env = model, end_time = params.get_param( "end_time" ) )
    return sim

def cycle(l, n):
  return itertools.chain.from_iterable(itertools.repeat(l, n))

def stringify_dict(p):
  for k in p.keys():
    if isinstance(p[k], list):
      p[k] = ",".join([f"{v}" for v in p[k]])
  return p

def bucket_to_age(b):
  return int(b.split("_")[1]) // 10

def set_app_users_fraction(params, frac):
    name = "app_users_fraction"
    for b in AGE_BUCKETS:
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

def print_params(params):
  for k in dir(params.c_params):
    if k.startswith("_"):
      continue
    print(f"{k}: {getattr(test_params.c_params, k)}")

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

def setup_params(network, params_overrides={}):
  params_dict = LOCAL_DEFAULT_PARAMS.copy()
  params_dict.update(params_overrides)

  params = get_baseline_parameters()
  for p, v in params_dict.items():
    if p in LOCAL_DEFAULT_PARAMS or p.startswith("predicted_"):
      continue
    if isinstance(v, np.int64) or (hasattr(v, "is_integer") and v.is_integer()):
      params.set_param( p, int(v) )
    else:
      params.set_param( p, v )

  if "app_users_fraction" in params_dict:
    set_app_users_fraction(params, params_dict["app_users_fraction"])

  if params_dict["manual_trace_n_workers_per_100k"] is not None:
    params.set_param(
        "manual_trace_n_workers",
        int(params_dict["manual_trace_n_workers_per_100k"] / 100000 *
            params_dict["n_total"]))
  
  if "rng_seed" in params_dict:
    np.random.seed(int(params_dict["rng_seed"]))
  hh_df = build_population( params_dict, network.households )
  params.set_demographic_household_table( hh_df )
  occupation_network_df = None
  if params_dict["custom_occupation_network"]:
    occupation_network_df = build_occupation_networks( params, network.sector_names )
  
    occupation_assignment = build_occupation_assignment( hh_df, occupation_network_df, network.sector_pdf )
    params.set_occupation_network_table( occupation_assignment, occupation_network_df )
  return params, occupation_network_df

def build_population(params_dict, houses):
  n_total = params_dict["n_total"]
  IDs = np.arange( n_total, dtype='int32')
  house_no = np.zeros( IDs.shape, dtype='int32' )
  ages = np.zeros( IDs.shape, dtype='int32' )
  idx = 0
  house_idx = 0
  for house in houses.sample(frac=1).itertuples():
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
    if idx >= n_total:
      break
  return pd.DataFrame({'ID':IDs, 'age_group':ages, 'house_no':house_no})


def diff_params(local_params, global_params={}):          
  if isinstance(local_params, pd.DataFrame):                                                
    local_params = local_params.loc[0].to_dict()
  if isinstance(global_params, pd.DataFrame):
    global_params = global_params.loc[0].to_dict()
  for col in sorted(set(local_params.keys()).union(global_params.keys())):
    g = global_params[col] if col in global_params else BASE_PARAMS[col]
    l = local_params[col] if col in local_params else BASE_PARAMS[col]
    if g != l:
      print(f"{col} {g} {l}")


def sim_plot(axs, df, label, n_total, time_offset):
  """Plot the simulation result to the given axes"""
  if "iteration" in df.columns:
    diff = df.groupby(["iteration"]).transform(lambda x: x.diff())
  else:
    diff = df.diff()
  df['new_infected'] = diff["total_infected"]
  df['new_death'] = diff["n_death"]
  df["total_infected_rate"] = df["total_infected"] / n_total
  df["quarantine_rate"] = df["n_quarantine"] / n_total
  if isinstance(time_offset, pd.Timestamp):
    df["date"] = time_offset + df.time.apply(lambda t: pd.Timedelta(t - 1, unit="d"))
  else:
    df["date"] = df["time"] - time_offset

  if "iteration" in df.columns and df.iteration.max() > 0:
    dgroup = df.groupby(["date"])
    dlow = dgroup.quantile([0.05])
    dmed = dgroup.mean()
    dhigh = dgroup.quantile([0.95])

    for ax, y in zip(axs, ["new_infected", "total_infected", "total_infected_rate", "n_death", "n_hospital", "n_tests", "quarantine_rate"]):
      dmed.plot( ax=ax, y = y, label=label, legend=False)
      ax.fill_between(dmed.index, dlow[y], dhigh[y], alpha=0.25)

  else:
    for ax, y in zip(axs, ["new_infected", "total_infected", "total_infected_rate", "n_death", "n_hospital", "n_tests", "quarantine_rate"]):
      df.plot( ax=ax, x = "date", y = y, label=label, legend=False)

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
  fig, axs = plt.subplots(2,4,figsize=(16,7.2),constrained_layout=True)
  axs = axs.flatten()
  axs[7].remove()
  axs[0].set_title('new infected')
  axs[1].set_title('total infected')
  axs[2].set_title('total infected percentage')
  axs[3].set_title('total deaths')
  axs[4].set_title('total in hospital')
  axs[5].set_title('tests per day')
  axs[6].set_title('percent in quarantine')
  label_prefix = common_prefix(labels)
  if label_prefix:
    labels = [label[len(label_prefix):] for label in labels]

  for result, label in zip(results,labels):
      param_model, df = result
      if "mobility_start_date" in param_model and "seeding_date_delta" in param_model:
        time_offset = pd.Timestamp(param_model['mobility_start_date']) - pd.Timedelta(param_model['seeding_date_delta'], unit='d')
      else:
        time_offset = param_model['time_offset']
      sim_plot(axs, df, label, param_model['n_total'], time_offset)
  axs[3].legend(bbox_to_anchor=(0, -0.2),loc='upper left',
                  title=label_prefix, ncol=math.ceil(len(results) / 12)).set_in_layout(False)
  return fig

def read_county_occupation_network( county_fips, all_occupations ):
  sector_df = all_occupations[all_occupations.area_fips == county_fips]
  sector_name = sector_df['industry_code'].values
  sector_size = sector_df['qrtly_emplvl'].values
  sector_pdf = sector_size / np.sum( sector_size )
  return list( sector_name ), list( sector_pdf )

def get_mean_work_interaction( params, sector_names, age_type, use_default):
  if use_default:
    return get_default_mean_work_interaction( params, age_type )
  else:
    return get_custom_mean_work_interaction( params, sector_names, age_type )

def get_custom_mean_work_interaction( params, sector_names, age_types ):
  default_value = get_default_mean_work_interaction( params, age_types )
  adjustment = [1.0, 1.0] + list(map(lambda x: WORK_INTERACTION_ADJUST_RATIO[x], sector_names)) + [1.0, 1.0]
  return default_value * np.array(adjustment)

def get_default_mean_work_interaction( params, age_types ):
  def get_by_age_type( age_type ):
    if age_type == 0:
      return params.get_param( 'mean_work_interactions_child' )
    elif age_type == 1:
      return params.get_param( 'mean_work_interactions_adult' )
    elif age_type == 2:
      return params.get_param( 'mean_work_interactions_elderly' )
    else:
      raise ValueError( 'not supported age type' )

  return np.array(list(map( lambda x: get_by_age_type( x ), age_types )))

def get_lockdown_multipliers( params, sector_names, use_default ):
  if use_default:
    return get_default_lockdown_multipliers( params, sector_names )
  else:
    return get_custom_lockdown_multipliers( params, sector_names )

def get_custom_lockdown_multipliers( params, sector_names ):
  m = get_default_lockdown_multipliers( params, sector_names )
  for i, sector_name in enumerate(sector_names):
    if sector_name in HIGH_IMPACT_LOCKDOWN_SECTORS:
      m[i+2] = m[i+2] * 0.5
  return m

def get_default_lockdown_multipliers( params, sector_names ):
  lockdown_multiplier = np.ones(
    len( sector_names ) + 4 ) * params.get_param('lockdown_occupation_multiplier_working_network')
  lockdown_multiplier[0] = params.get_param('lockdown_occupation_multiplier_primary_network')
  lockdown_multiplier[1] = params.get_param('lockdown_occupation_multiplier_secondary_network')
  lockdown_multiplier[-2] = params.get_param('lockdown_occupation_multiplier_retired_network')
  lockdown_multiplier[-1] = params.get_param('lockdown_occupation_multiplier_elderly_network')
  return lockdown_multiplier

def build_occupation_networks( params, sector_names,
                               use_default_work_interaction=False,
                               use_default_lockdown_multiplier=True,
                               n_child_network=2,
                               n_elderly_network=2):
  n_networks = n_child_network + len( sector_names ) + n_elderly_network
  network_no = np.arange( n_networks, dtype='int32' )
  network_names = ['primary', 'secondary'] + sector_names + ['retired', 'elderly']
  age_types = np.ones( n_networks )
  age_types[:2] = 0 # child
  age_types[-2:] = 2 # elderly
  mean_work_interactions = get_mean_work_interaction( params, sector_names, age_types, use_default_work_interaction )
  lockdown_multipliers = get_lockdown_multipliers( params, sector_names, use_default_lockdown_multiplier )

  return pd.DataFrame({
    'network_no': network_no,
    'age_type': age_types,
    'mean_work_interaction': mean_work_interactions,
    'lockdown_multiplier': lockdown_multipliers,
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

def run_lockdown(network, params_dict):
  params, occupation_network = setup_params(network, params_dict)
  model = get_simulation( params ).env.model

  total_days_left = int(params_dict['end_time'])
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
          
  for step in trange(total_days_left):
    if step == params_dict['lockdown_days']:
      model.update_running_params("lockdown_on", 0)
      if params_dict["app_turned_on"]:
        model.update_running_params("app_turned_on", 1)
      if params_dict["manual_trace_on"]:
        model.update_running_params("manual_trace_on", 1)
        
    model.one_time_step()
    m_out.append(model.one_time_step_results())
  del model
  return m_out

def scale_lockdown(model, scalar, base_multipliers, scale_all=False):
  if scale_all:
    for idx, base in base_multipliers.iteritems():
      covid19.set_model_param_lockdown_occupation_multiplier(model.c_model, scalar * base, idx)
  else:
    # Ignore school and elderly networks.
    for idx, base in base_multipliers[2:-2].iteritems():
      covid19.set_model_param_lockdown_occupation_multiplier(model.c_model, scalar * base, idx)

def run_baseline_forecast(network, params_dict):
  params, occupation_network = setup_params(network, params_dict)
  model = get_simulation( params ).env.model

  if isinstance(params_dict["lockdown_scalars"], list):
    scalars = params_dict["lockdown_scalars"]
  else:
    scalars = [float(x) for x in params_dict["lockdown_scalars"].split(",")]

  if isinstance(params_dict["changepoint_scalars"], list):
    changepoints = params_dict["changepoint_scalars"]
  else:
    changepoints = [float(x) for x in params_dict["changepoint_scalars"].split(",")]

  if len(scalars) != len(changepoints):
    raise ValueError(f"Scalar length mismatch {len(scalars)} vs {len(changepoints)}")

  if occupation_network is not None:
    base_multipliers = occupation_network.lockdown_multiplier.copy()
  else:
    base_multipliers = pd.Series([
      params.get_param('lockdown_occupation_multiplier_primary_network'),
      params.get_param('lockdown_occupation_multiplier_secondary_network'),
      params.get_param('lockdown_occupation_multiplier_working_network'),
      params.get_param('lockdown_occupation_multiplier_retired_network'),
      params.get_param('lockdown_occupation_multiplier_elderly_network'),
    ])

  if params_dict["static_mobility_scalar"]:
    base_multipliers[:] = 1
  scale_all = params_dict["mobility_scale_all"]

  base_random = model.get_param("lockdown_random_network_multiplier")

  base_rel_trans_rand = params.get_param("relative_transmission_random")
  base_rel_trans_occ = params.get_param("relative_transmission_occupation")

  m_out = []
  seeding_date_delta = params_dict["seeding_date_delta"]
  baseline_days = len(scalars) + seeding_date_delta
  params_dict["time_offset"] = baseline_days
  end_time = max(params_dict["end_time"], baseline_days)
  with tqdm(total=end_time) as pbar:
    for step in range(seeding_date_delta):
      model.one_time_step()
      m_out.append(model.one_time_step_results())
      pbar.update()

    model.update_running_params("lockdown_on", 1)

    for step in range(len(scalars)):
      scale_lockdown(model, scalars[step], base_multipliers, scale_all)
      # Scale random network too.
      model.update_running_params('lockdown_random_network_multiplier', scalars[step] * base_random)

      model.update_running_params('relative_transmission_random', changepoints[step] * base_rel_trans_rand)
      model.update_running_params('relative_transmission_occupation', changepoints[step] * base_rel_trans_occ)

      model.one_time_step()
      m_out.append(model.one_time_step_results())
      pbar.update()

    if params_dict["app_turned_on"]:
      model.update_running_params("app_turned_on", 1)
    if params_dict["manual_trace_on"]:
      model.update_running_params("manual_trace_on", 1)
    for p, v in params_dict.items():
      if not p.startswith("predicted_"):
        continue
      model.update_running_params(p.replace("predicted_",""), v)
            
    for step in range(end_time - baseline_days):
      model.one_time_step()
      m_out.append(model.one_time_step_results())
      pbar.update()

  del model
  return m_out

def run_model(params_dict, houses, sector_names, sector_pdf):
  pt = LOCAL_DEFAULT_PARAMS.copy()
  pt.update(params_dict)
  params_dict = pt

  network = Network(houses, sector_names, sector_pdf)

  if params_dict["lockdown_scalars"]:
    m_out = run_baseline_forecast(network, params_dict)
  else:
    m_out = run_lockdown(network, params_dict)

  df = pd.DataFrame( m_out )
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
               run_parallel=True,
               n_iterations=1):
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
    self.n_iterations = n_iterations

  def run_county(self, county_fips, params_overrides={}):
    sector_names, sector_pdf = read_county_occupation_network(county_fips, self.occupations)
    households = self.households[self.households["county_fips"] == county_fips]
    params = self.county_params.loc[county_fips].to_dict()
    params.update(params_overrides)
    return run_model(params, households, sector_names, sector_pdf)

  def get_county_run(self, county_fips, params_overrides={}):
    sector_names, sector_pdf = read_county_occupation_network(county_fips, self.occupations)
    households = self.households[self.households["county_fips"] == county_fips]
    params = self.county_params.loc[county_fips].to_dict()
    params.update(params_overrides)
    return params, Network(households, sector_names, sector_pdf)

  def get_county_params(self, county_fips, params_overrides={}):
    params, network = self.get_county_run(county_fips, params_overrides)

    return setup_params(network, params)[0]

  def run_counties(self, counties_fips, params_overrides={}):
    params_overrides = remove_nans(params_overrides)
    n_iterations = self.n_iterations
    n_runs = n_iterations * len(counties_fips)
    all_params = []
    for idx in range(n_iterations):
      p = params_overrides.copy()
      if "rng_seed" in p:
        p["rng_seed"] += idx * 10
      all_params.append(itertools.repeat(p, len(counties_fips)))
    all_params = itertools.chain.from_iterable(all_params)

    if self.run_parallel:
      runner = ProcessPoolExecutor
    else:
      runner = lambda: ThreadPoolExecutor(max_workers=1)

    with runner() as ex:
      outputs = list(
        tqdm(
          ex.map(self.run_county,
                 cycle(counties_fips, n_iterations),
                 all_params),
          total=n_runs,
          desc="Batch progress"
        )
      )

    results = collections.defaultdict(list)
    for county_fips, output in zip(cycle(counties_fips, n_iterations), outputs):
      output[1]["iteration"] = (output[0]["iteration"]
                                if self.n_iterations == 1
                                else len(results[county_fips]))
      results[county_fips].append(output)
    for county_fips, all_results in results.items():
      df = pd.concat([result for _, result in all_results])
      self.results[county_fips] = [all_results[0][0], df]

    return self.results

  def run_all(self, params_overrides={}):
    self.run_counties(self.counties, params_overrides)
    return self.results

  def merge_results(self):
    merged_results = pd.concat([result[1] for result in self.results.values()]).groupby(["iteration","time"]).sum().reset_index()
    merged_params = next(iter(self.results.values()))[0].copy()
    merged_params["n_total"] = sum((result[0]["n_total"] for result in self.results.values()))
    if not merged_params["lockdown_scalars"]:
      merged_params["time_offset"] = 0
    self.merged_results = [merged_params, merged_results]
    return self.merged_results

  def plot_results(self, output_dir=None):
    if not self.results:
      return
    if not self.merged_results:
      self.merge_results()
    fig = sim_display([self.merged_results], ["AggregateModel"])
    if output_dir:
      fig.savefig(os.path.join(output_dir, f"merged_results.png"))

  def plot_county(self, county_fips, output_dir=None):
    if county_fips not in self.results:
      return
    fig = sim_display([self.results[county_fips]], [f"County {county_fips}"])
    if output_dir:
      fig.savefig(os.path.join(output_dir, f"{county_fips}_results.png"))

  def write_results(self, output_dir, write_merged=True):
    if not self.results:
      return
    try:
      os.makedirs(output_dir)
    except FileExistsError:
      pass

    for county, output in self.results.items():
      pd.DataFrame(stringify_dict(output[0]), index=[0]).to_csv(os.path.join(output_dir, f"{county}_params.csv"), index=False)
      output[1].to_csv(os.path.join(output_dir, f"{county}_results.csv"), index=False)

    if not write_merged:
      return
    if not self.merged_results:
      self.merge_results()
    params, result = self.merged_results
    pd.DataFrame(stringify_dict(params), index=[0]).to_csv(os.path.join(output_dir, f"merged_params.csv"), index=False)
    result.to_csv(os.path.join(output_dir, f"merged_results.csv"), index=False)


def read_results(base_dir, sim_names=None):
  study_params = None
  model = None
  if sim_names is None:
    paths = glob.glob(os.path.join(base_dir, "*_merged_results.csv"))
    sim_names = [os.path.basename(path).replace("_merged_results.csv","") for path in paths]

  results = []
  for sim in sim_names:
    try:
      params = pd.read_csv(os.path.join(base_dir, f"{sim}_merged_params.csv")).loc[0].to_dict()
    except:
      if study_params is None:
        study_params = pd.read_csv(DEFAULT_ARGS.study_params, comment="#", index_col="study_name")
        model = AggregateModel()
      params, _ = model.get_county_run(53033, study_params.loc[sim].to_dict())
    result = pd.read_csv(os.path.join(base_dir, f"{sim}_merged_results.csv"))
    results.append([params, result])

  return results, sim_names


def upload(d, gcs_path):
  no_prefix = gcs_path[len('gs://'):]
  splits = no_prefix.split('/')
  bucket = splits[0]
  save_path = '/'.join(splits[1:])

  prefix = os.path.basename(d)
  storage_client = storage.Client()
  for f in os.listdir(d):
    blob = storage.Blob(os.path.join(save_path, f"{prefix}_{f}"),
                        storage_client.get_bucket(bucket))
    with open(os.path.join(d, f), 'rb') as fh:
      blob.upload_from_file(fh)


def main(args):
  output_dir = args.output_dir

  if args.study_params:
    overrides = pd.read_csv(args.study_params, comment="#")
    if "study_name" not in overrides:
      overrides["study_name"] = [f"{i}" for i in range(len(overrides))]
  else:
    overrides = pd.DataFrame({"study_name": ["0"]})

  if args.study_line is not None:
    overrides = overrides.iloc[args.study_line:args.study_line+1]

  state_param_file = args.statewide_parameters
  households_file = args.household_demographics
  occupations_file = args.occupations
  county_params_file = args.county_parameters

  model = AggregateModel([state_param_file], households_file, occupations_file, county_params_file)
  if args.counties:
    counties = [int(c.strip()) for c in args.counties.split(',')]
  else:
    counties = model.counties
  for override in overrides.itertuples():
    model.run_counties(counties, override._asdict())
    if output_dir:
      study_path = os.path.join(output_dir, override.study_name)
      model.write_results(study_path)
      if args.gcs_path:
        upload(study_path, args.gcs_path)

if __name__ == "__main__":
  main(parser.parse_args())
