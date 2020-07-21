# Lint as: python3
import itertools
import pandas as pd

adoption_sweep = [(f"en_{rate:0.1f}", {
    "app_turned_on": 1,
    "app_users_fraction": rate,
}) for rate in [0,0.2,0.4,0.6,0.8,1]]

social_distancing = [(f"en_{rate:0.1f}_social_dist_{dist:0.1f}", {
    "app_turned_on": 1,
    "app_users_fraction": rate,
    "predicted_relative_transmission_occupation": dist,
    "predicted_relative_transmission_random": dist,
}) for rate, dist in itertools.product([0,0.2,0.4,0.6,0.8,1], [0.4,0.6,0.8,1])]

test_wait = [(f"en_{rate:0.1f}_test_delay_{delay}", {
    "app_turned_on": 1,
    "app_users_fraction": rate,
    "predicted_test_result_wait": delay,
}) for rate, delay in itertools.product([0,0.2,0.4,0.6,0.8,1], [0,1,2,4,6,8,10])]

manual_tracing_delay = [(f"en_{rate:0.1f}_man_trace_delay_{delay}", {
    "app_turned_on": 1,
    "manual_trace_on": 1,
    "app_users_fraction": rate,
    "manual_trace_delay": delay,
}) for rate, delay in itertools.product([0,0.2,0.4,0.6,0.8,1], [0,1,2,4])]

manual_tracing_nwork = [(f"en_{rate:0.1f}_man_trace_numwork_{nwork}", {
    "app_turned_on": 1,
    "manual_trace_on": 1,
    "app_users_fraction": rate,
   # "manual_trace_n_workers": nwork,
    "manual_trace_n_workers_per_100k": nwork,
}) for rate, nwork in itertools.product([0,0.2,0.4,0.6,0.8,1], [0, 4.7, 8.3, 15, 30, 100])]
# }) for rate, nwork in itertools.product([0,0.2,0.4,0.6,0.8,1], [0, 22, 49, 405, 810])]
#Totals for WA State are 1371,1500,2285,1000000

school_manual_tracing_nwork = [(f"en_{rate:0.1f}_school_man_trace_numwork_{nwork}", {
    "app_turned_on": 1,
    "manual_trace_on": 1,
    "app_users_fraction": rate,
    "predicted_lockdown_occupation_multiplier_primary_network": 0.8,
    "predicted_lockdown_occupation_multiplier_secondary_network": 0.8,
    "manual_trace_n_workers_per_100k": nwork,
}) for rate, nwork in itertools.product([0, 0.4], [0, 4.7, 8.3, 15, 30, 100])]

exclude_app_manual_tracing_nwork = [(f"en_{rate:0.1f}_exclude_app_man_trace_numwork_{nwork}", {
    "app_turned_on": 1,
    "manual_trace_on": 1,
    "app_users_fraction": rate,
    "manual_trace_exclude_app_users": 1,
    "manual_trace_n_workers_per_100k": nwork,
}) for rate, nwork in itertools.product([0, 0.4, 0.8], [0, 4.7, 8.3, 15, 30, 100])]

def main():
  all_sweeps = list(itertools.chain(adoption_sweep, social_distancing, test_wait,
                                    manual_tracing_delay, manual_tracing_nwork,
                                    school_manual_tracing_nwork,
                                    exclude_app_manual_tracing_nwork))
  sims = []
  for sim_name, params in all_sweeps:
    for idx in range(10):
      p = params.copy()
      p["rng_seed"] = idx * 10
      p["study_name"] = f"{sim_name}_{idx}"
      p["iteration"] = idx
      sims.append(p)
  pd.DataFrame(sims).to_csv("../data/us-wa/simulations.csv",index=False)

if __name__ == "__main__":
  main()
