# Lint as: python3
import county_utils
import itertools
import pandas as pd

mobility_glm = [
    0.8402953, 0.66203356, 0.6886954, 0.7405823, 0.5981137, 0.50632834,
    0.6051636, 0.64862704, 0.5861519, 0.5325351, 0.54879653, 0.5175134,
    0.37579238, 0.50470424, 0.5443538, 0.44272733, 0.40119568, 0.44219303,
    0.40037775, 0.3844627, 0.37436888, 0.29775846, 0.24074203, 0.222709,
    0.21712849, 0.17861758, 0.17651758, 0.19569567, 0.22771499, 0.17034364,
    0.19542843, 0.22134075, 0.1913721, 0.18518353, 0.23677555, 0.21784332,
    0.2253096, 0.2340494, 0.25321692, 0.24587187, 0.22654305, 0.23475423,
    0.2149702, 0.21154118, 0.23558968, 0.23848315, 0.24233575, 0.21790002,
    0.22207996, 0.2653197, 0.23323302, 0.2031609, 0.17560856, 0.19093654,
    0.21117698, 0.25086582, 0.2563419, 0.21936506, 0.22368538, 0.23401839,
    0.22154263, 0.2489432, 0.22269249, 0.28478175, 0.23344915, 0.31041574,
    0.24227664, 0.3074823, 0.34701902, 0.42372382, 0.41289246, 0.23164195,
    0.24963501, 0.29317856, 0.2595467, 0.32093036, 0.25804445, 0.3545251,
    0.28350806, 0.29591748, 0.24741918, 0.23066965, 0.2666502, 0.37182048,
    0.4252485, 0.19042203, 0.32094365, 0.4092322, 0.35159194, 0.3726731,
    0.24597853, 0.32668212, 0.31032312, 0.2978509, 0.35226935, 0.33507806,
    0.3191262, 0.40766823, 0.39124697, 0.30448493, 0.25948995, 0.4151942,
    0.34768903, 0.32266316, 0.410343, 0.5093061, 0.27706283, 0.40219194,
    0.46280217, 0.46270996, 0.44115028, 0.45049942, 0.49620712, 0.42809692,
    0.47197294, 0.42999616, 0.46053863, 0.4188922, 0.36068365
]

mobility_glm_scalar = [i / mobility_glm[0] for i in mobility_glm]

mobility_glm_scalar.append(pd.Series(mobility_glm_scalar).rolling(7,1).mean().to_list()[-1])

changepoint_rate = [
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    0.9999999, 0.9999995, 0.9999976, 0.9999867, 0.9999467, 0.99974406,
    0.9986508, 0.9961408, 0.9910468, 0.98409015, 0.97409594, 0.9661807,
    0.95578575, 0.9453572, 0.9319753, 0.9198195, 0.90893304, 0.89379597,
    0.88160473, 0.86828125, 0.85234666, 0.83898604, 0.82659066, 0.81570756,
    0.80153155, 0.78918207, 0.7782723, 0.766945, 0.7561462, 0.7479454,
    0.73912144, 0.73034966, 0.72378343, 0.71540636, 0.70769167, 0.6987137,
    0.690784, 0.68500406, 0.6780368, 0.6704352, 0.6698507, 0.6695024, 0.6694232,
    0.6694232, 0.6693733, 0.6693729, 0.6693728, 0.6693728, 0.6693728, 0.6693728,
    0.6693728, 0.6693728, 0.6693728, 0.6693728, 0.6693728, 0.6693728, 0.6693728,
    0.6693728, 0.6693728, 0.6693728, 0.6693728, 0.6693728, 0.6693728, 0.6693728,
    0.6693728, 0.6693728, 0.6693728, 0.6693728, 0.6693728, 0.6693728, 0.6693728,
    0.6693728, 0.6693728, 0.6693728, 0.6693728, 0.6693728, 0.6693728, 0.6693728,
    0.6693728, 0.6693728, 0.6693728, 0.6693728
]

changepoint_rate_scalar = [i / changepoint_rate[0] for i in changepoint_rate]

changepoint_rate_scalar.append(changepoint_rate_scalar[-1])

params_simulation = {
    "end_time" : 400,
    "app_users_fraction": 0.6,
    "quarantine_on_traced": 1,
    "retrace_on_positive": 1,
    "trace_on_positive": 1,
    "trace_on_symptoms": 0,
}

def stringify(l):
  return ",".join((f"{x}" for x in l))

params_baseline = {
    "rng_seed" : 0,
    "app_turned_on" : 0,
    "n_seed_infection": 30,
    "custom_occupation_network": 1,
    "infectious_rate": 6.5,
    "lockdown_scalars": stringify(mobility_glm_scalar),
    "changepoint_scalars": stringify(changepoint_rate_scalar),
    "seeding_date_delta": 15,
    "mobility_scale_all": 0,
    "static_mobility_scalar": 0,
}

adoption_sweep = [(f"en_{rate:0.2f}", {
    "app_turned_on": 1,
    "app_users_fraction": rate,
}) for rate in [0,0.2,0.4,0.6,0.8,1]]

social_distancing = [(f"en_{rate:0.2f}_social_dist_{dist:0.2f}", {
    "app_turned_on": 1,
    "app_users_fraction": rate,
    "relative_transmission_occupation": dist,
    "relative_transmission_random": dist,
}) for rate, dist in itertools.product([0,0.2,0.4,0.6,0.8,1], [0.4,0.6,0.8,1])]

test_wait = [(f"en_{rate:0.2f}_test_delay_{delay}", {
    "app_turned_on": 1,
    "app_users_fraction": rate,
    "test_result_wait": delay,
}) for rate, delay in itertools.product([0,0.2,0.4,0.6,0.8,1], [0,1,2,4,6,8,10])]

manual_tracing_delay = [(f"en_{rate:0.2f}_man_trace_delay_{delay}", {
    "app_turned_on": 1,
    "manual_trace_on": 1,
    "app_users_fraction": rate,
    "manual_trace_delay": delay,
}) for rate, delay in itertools.product([0,0.2,0.4,0.6,0.8,1], [0,1,2,4])]

manual_tracing_nwork = [(f"en_{rate:0.2f}_man_trace_numwork_{nwork}", {
    "app_turned_on": 1,
    "manual_trace_on": 1,
    "app_users_fraction": rate,
    "manual_trace_n_workers": nwork,
}) for rate, nwork in itertools.product([0,0.2,0.4,0.6,0.8,1], [1371,1500,2285,1000000])]

def main():
  all_sweeps = list(itertools.chain(adoption_sweep, social_distancing, test_wait, manual_tracing_delay, manual_tracing_nwork))
  bp = params_simulation
  bp.update(params_baseline)
  
  sims = []
  for sim_name, params in all_sweeps:
    for idx in range(10):
      p = bp.copy()
      p.update(params)
      if "rng_seed" in p:
        p["rng_seed"] += idx * 10
      p["study_name"] = f"{sim_name}_{idx}"
      p["iteration"] = idx
      sims.append(p)
  pd.DataFrame(sims).to_csv("../data/us-wa/simulations.csv",index=False)

if __name__ == "__main__":
  main()
