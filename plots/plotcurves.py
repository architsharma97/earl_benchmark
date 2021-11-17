mport matplotlib  # pylint: disable=g-import-not-at-top, unused-import
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
# plt.rcParams["text.usetex"] = True

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": ["Palatino"]})

import matplotlib.cm as cm  # pylint: disable=g-import-not-at-top, unused-import
import matplotlib.ticker as ticker

import numpy as np  # pylint: disable=g-import-not-at-top, reimported
import seaborn as sns  # pylint: disable=g-import-not-at-top, unused-import
import re  # pylint: disable=g-import-not-at-top, unused-import

import pickle as pkl  # pylint: disable=g-import-not-at-top, unused-import
import os  # pylint: disable=g-import-not-at-top, reimported
import csv

max_index = int(1e7)
custom = False

# alpha = 0 -> no smoothing, alpha=1 -> perfectly smoothed to initial value
def smooth(x, alpha):
  if isinstance(x, list):
    size = len(x)
  else:
    size = x.shape[0]
  for idx in range(1, size):
    x[idx] = (1 - alpha) * x[idx] + alpha * x[idx - 1]
  return x

def make_graph_with_variance(vals, x_interval, use_standard_error=True):
  data_x = []
  data_y = []
  global max_index
  num_seeds = 0
  for y_coords, eval_interval in zip(vals, x_interval):
    num_seeds += 1
    data_y.append(smooth(y_coords, 0.95))
    x_coords = [eval_interval * idx for idx in range(len(y_coords))]
    data_x.append(x_coords)

  plot_dict = {}
  cur_max_index = max_index
  for cur_x, cur_y in zip(data_x, data_y):
    cur_max_index = min(cur_max_index, cur_x[-1])
    # print(cur_x[-1])
  print(cur_max_index)

  for cur_x, cur_y in zip(data_x, data_y):
    for x, y in zip(cur_x, cur_y):
      if x <= cur_max_index:
        if x in plot_dict.keys():
          plot_dict[x].append(y)
        else:
          plot_dict[x] = [y]

  print('output at step:', cur_max_index)
  print(np.mean(plot_dict[cur_max_index]), np.std(plot_dict[cur_max_index]) / np.sqrt(5))

  index, means, stds = [], [], []
  for key in sorted(plot_dict.keys()):  # pylint: disable=g-builtin-op
    index.append(key)
    means.append(np.mean(plot_dict[key]))
    if use_standard_error:
      stds.append(np.std(plot_dict[key]) / np.sqrt(num_seeds))
    else:
      stds.append(np.std(plot_dict[key]))

  means = np.array(smooth(means, 0.8))
  stds = np.array(smooth(stds, 0.8))

  return index, means, stds

def np_custom_load(fname):
  return np.load(fname).astype(np.float32)

color_map = ['#73BA68', 'r', 'c', 'm', 'y', '#9A9C99', 'b']
style_map = []
for line_style in ['-', '--', '-.', ':']:
  style_map += [(color, line_style) for color in color_map]

def plot_call(experiment_paths,
              legend_label,
              plot_style,
              y_plot='return',
              process_softlearning_data=False,
              softlearning_idx=None,
              linewidth=1):
  """Outermost function for plotting graphs with variance."""
  if y_plot == 'return':
    if process_softlearning_data:
      if softlearning_idx is None:
        print('please write the code for getting the return index')
      base_path = os.path.dirname(experiment_paths)
      exp_name = experiment_paths.split('/')[-1]
      all_progress_files = []
      for path in os.listdir(base_path):
        if path.endswith(exp_name):
          all_progress_files.append(os.path.join(base_path, path, next(os.walk(os.path.join(base_path, path)))[1][0], 'progress.csv'))

      y_coords = []
      eval_interval = []
      for f in all_progress_files:
        data = open(f).read().splitlines()
        y_coords.append(np.array([float(entry[softlearning_idx]) for entry in list(csv.reader(data))[1:]]))
        eval_interval.append(2000)
    else:
      y_coords = [
          np_custom_load(os.path.join(experiment_path, 'eval', 'average_eval_return.npy'))
          for experiment_path in experiment_paths
      ]
      eval_interval = [
              np_custom_load(os.path.join(experiment_path, 'eval', 'eval_interval.npy'))
              for experiment_path in experiment_paths
      ]

  elif y_plot == 'success':
    if process_softlearning_data:
      # base_path = os.path.dirname(experiment_paths)
      # exp_name = experiment_paths.split('/')[-1]
      base_path = experiment_paths
      all_progress_files = []
      for path in os.listdir(base_path):
        if path.startswith('id='):
          # print(os.path.join(base_path, path, 'progress.csv'))
          # all_progress_files.append(os.path.join(base_path, path, next(os.walk(os.path.join(base_path, path)))[1][0], 'progress.csv'))
          all_progress_files.append(os.path.join(base_path, path, 'progress.csv'))

      if softlearning_idx is None:
        for idx, col_header in enumerate(open(all_progress_files[0], 'r').readline().split(',')):
          if col_header == 'evaluation_0/episode-reward-mean':
            softlearning_idx = idx
            break
        # print('softlearning eval idx:', softlearning_idx)

        for idx, col_header in enumerate(open(all_progress_files[0], 'r').readline().split(',')):
          if col_header == 'timestep':
            eval_interval_idx = idx
            break
        # print('eval interval idx:', eval_interval_idx)

      y_coords = []
      eval_interval = []
      for f in all_progress_files:
        data = open(f).read().splitlines()
        y_coords.append(10 * np.array([float(entry[softlearning_idx]) for entry in list(csv.reader(data))[1:]]))
        eval_interval.append(int(list(csv.reader(data))[1][eval_interval_idx]))

    else:
      y_coords = [
          np_custom_load(os.path.join(experiment_path, 'eval', 'average_eval_success.npy'))
          for experiment_path in experiment_paths
      ]
      # print(np.mean([np.where(val == 10.)[0][0] for val in y_coords]) * 10000, np.std([np.where(val == 10.)[0][0] for val in y_coords]) * 10000)
      eval_interval = [
            np_custom_load(os.path.join(experiment_path, 'eval', 'eval_interval.npy'))
            for experiment_path in experiment_paths
      ]

    # temporary fix
    y_coords = [y_entries / 10 for y_entries in y_coords]

  index, means, stds = make_graph_with_variance(y_coords, eval_interval)
  if plot_style[0] == '#9A9C99':
    plt.plot(index, means, color=plot_style[0], linestyle=plot_style[1], label=legend_label, linewidth=linewidth, 
      dashes=[6, 6])
  else:
    plt.plot(index, means, color=plot_style[0], linestyle=plot_style[1], label=legend_label, linewidth=linewidth)
  cur_color = plot_style[0]
  plt.fill_between(
      index, means - stds, means + stds, color=cur_color, alpha=0.2)

def compute_lifelong_return(experiments, 
                            legend_label,
                            plot_style,
                            process_softlearning_data=False,
                            linewidth=1):
  if process_softlearning_data:
      # base_path = os.path.dirname(experiment_paths)
      # exp_name = experiment_paths.split('/')[-1]
      base_path = experiments
      all_progress_files = []
      for path in os.listdir(base_path):
        if path.startswith('id='):
          # all_progress_files.append(os.path.join(base_path, path, next(os.walk(os.path.join(base_path, path)))[1][0], 'progress.csv'))
          all_progress_files.append(os.path.join(base_path, path, 'progress.csv'))

      for idx, col_header in enumerate(open(all_progress_files[0], 'r').readline().split(',')):
        if col_header == 'training_0/lifelong_return':
          softlearning_idx = idx
          break
      print('lifelong return idx:', softlearning_idx)

      for idx, col_header in enumerate(open(all_progress_files[0], 'r').readline().split(',')):
        if col_header == 'timestep':
          eval_interval_idx = idx
          break
      print('eval interval idx:', eval_interval_idx)

      ll_return_by_iter = []
      eval_interval = []
      for f in all_progress_files:
        data = open(f).read().splitlines()
        ll_return_by_iter.append(np.array([float(entry[softlearning_idx]) for entry in list(csv.reader(data))[1:]])*2)
        eval_interval.append(int(list(csv.reader(data))[1][eval_interval_idx])*2)

  else:    
    ll_return = []
    ll_return_by_iter = []
    for experiment in experiments:
      try:
        ll_return.append(np.load(os.path.join(experiment, 'eval/final_ll_return.npy')))
      except:
        print('final lifelong return not available')
      ll_return_by_iter.append(np.load(os.path.join(experiment, 'eval/lifelong_return_by_iter.npy')).astype(np.float32))
    
    eval_interval = [
        np_custom_load(os.path.join(experiment_path, 'eval', 'eval_interval.npy'))
        for experiment_path in experiments
    ]

  ll_return_by_iter = np.array(ll_return_by_iter)

  index, means, stds = make_graph_with_variance(ll_return_by_iter, eval_interval)
  if plot_style[0] == '#9A9C99':
    plt.plot(index, means, color=plot_style[0], linestyle=plot_style[1], label=legend_label, linewidth=linewidth, 
      dashes=[6, 6])
  else:
    plt.plot(index, means, color=plot_style[0], linestyle=plot_style[1], label=legend_label, linewidth=linewidth)
  cur_color = plot_style[0]
  plt.fill_between(
      index, means - stds, means + stds, color=cur_color, alpha=0.2)

  # print(ll_return)
  # print('lifelong return: %f (%f)' %(np.mean(ll_return), np.std(ll_return)))

if __name__ == '__main__':
  # basic configurations
  base_path = 'experiments'
  # base_path = '../benchmark_results'
  style_idx = -1

  plot_type = 'bulb'
  mode = 'll'

  # DHand Pickup Task ----------------------------
  if plot_type == 'dhand':
    max_index = int(7e6)
    plot_name = 'dhand_success.png'
    title = 'hand manipulation'
    yaxis_type = 'success'

    style_idx += 1
    legend_label = 'VaPRL'
    print(legend_label)
    experiment_base = 'sawyer_dhand/vaprl/hparam_sweep/off10_th-300'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [10, 11, 12, 13, 14, 15]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=3)

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = 'sawyer_dhand/fbrl_hand_relocation'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type)

    # style_idx += 1
    # legend_label = 'VaPRLs-200'
    # print(legend_label)
    # experiment_base = 'sawyer_dhand/vaprl/off10_th-200'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type)

    # style_idx += 1
    # legend_label = 'VaPRLg-200'
    # print(legend_label)
    # experiment_base = 'sawyer_dhand/vaprl/hparam_sweep/off10_th-200'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [10, 11, 12, 13, 14, 15]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type)

    # style_idx += 1
    # legend_label = 'VaPRLg-400'
    # print(legend_label)
    # experiment_base = 'sawyer_dhand/vaprl/hparam_sweep/off10_th-400'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [10, 11, 12, 13, 14, 15]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type)

    style_idx += 1
    legend_label = 'naive RL'
    print(legend_label)
    experiment_base = 'sawyer_dhand/naive'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type)

    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, 'r3l_baseline_results/gym/Sawyer/DhandPickupRandom-v0/sawyer_dhand_rf')
    plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type,
              process_softlearning_data=True,
              softlearning_idx=494 if yaxis_type == 'return' else 498)

    style_idx = 19
    legend_label = 'oracle RL'
    print(legend_label)
    experiment_base = 'sawyer_dhand/oracle'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [2, 3, 4, 6, 7]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=0.8)

  elif plot_type == 'tabletop':
    max_index = int(2.5e6)
    plot_name = 'tabletop'
    if mode == 'transfer':
      plot_name += '_transfer.png'
    elif mode == 'll':
      plot_name += '_ll.png'

    title = 'table-top reorganiztion'

    style_idx += 1
    legend_label = 'VaPRL'
    print(legend_label)
    experiment_base = 'tabletop_manipulation/vaprl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = 'tabletop_manipulation/fbrl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'naive'
    print(legend_label)
    experiment_base = 'tabletop_manipulation/naive_ep_terminate'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, '../../benchmark_results/r3l/r3l_tabletop')
    if mode == 'transfer':
      plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success',
              process_softlearning_data=True, linewidth=1.5)
    if mode == 'll':
      compute_lifelong_return(experiment_base, legend_label, plot_style=style_map[style_idx],
              process_softlearning_data=True, linewidth=1.5)

    style_idx = 19
    legend_label = 'oracle'
    print(legend_label)
    experiment_base = 'playpen_tree/rc_o-rc_k-rc_p-rc_b/oracle_reset'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1)

  elif plot_type == 'peg':
    max_index = int(2e6)
    plot_name = 'sawyer_peg'
    if mode == 'transfer':
      plot_name += '_transfer.png'
    elif mode == 'll':
      plot_name += '_ll.png'

    title = 'sawyer peg insertion'

    style_idx += 1
    legend_label = 'VaPRL'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_peg/vaprl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_peg/fbrl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'naive'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_peg/naive'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)
    
    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, '../../benchmark_results/r3l/peg_longer')
    if mode == 'transfer':
      plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success',
              process_softlearning_data=True, linewidth=1.5)
    if mode == 'll':
      compute_lifelong_return(experiment_base, legend_label, plot_style=style_map[style_idx],
              process_softlearning_data=True, linewidth=1.5)

    style_idx = 19
    legend_label = 'oracle'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_peg/oracle'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1)
  
  # PLAYPEN PLOTS --------------------------------
  elif plot_type == 'playpen':
    max_index = int(2.5e6)
    plot_name = 'playpen_ablation.png'
    title = 'table-top rearrangement'

    style_idx += 1
    legend_label = 'VaPRL (200k)'
    print(legend_label)
    experiment_base = 'playpen_tree/all/curriculum_offline_relabel/test'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=3)

    # style_idx += 1
    # legend_label = 'VaPRL + reset'
    # print(legend_label)
    # experiment_base = 'playpen_tree/rc_o-rc_k-rc_p-rc_b/curriculum_reset/th1.0'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=3)
    
    # style_idx += 1
    # legend_label = 'random'
    # print(legend_label)
    # experiment_base = 'playpen_tree/all/random_curriculum'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'naive RL'
    # print(legend_label)
    # experiment_base = 'playpen_tree/all/naive'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'FBRL'
    # print(legend_label)
    # experiment_base = 'playpen_tree/all/rf_fixed'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'uniform / R3L + reset'
    # print(legend_label)
    # experiment_base = 'playpen_tree/rc_o-rc_k-rc_p-rc_b/uniform_reset'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'uniform / R3L + reset (big)'
    # print(legend_label)
    # experiment_base = 'playpen_tree/rc_o-rc_k-rc_p-rc_b/uniform_reset/bigger_square'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'R3L'
    # print(legend_label)
    # experiment_base = os.path.join(base_path, 'r3l_baseline_results/gym/PlayPen/OneObject4Goals-v0/playpen_tree_rf')
    # plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success', process_softlearning_data=True)

    style_idx += 1
    # style_idx = 19
    legend_label = 'FBRL (200)'
    print(legend_label)
    experiment_base = 'playpen_tree/rc_o-rc_k-rc_p-rc_b/oracle_reset'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1)


    style_idx += 1
    legend_label = 'FBRL (2k)'
    print(legend_label)
    experiment_base = 'playpen_tree/all/reset_freq_exp/20k'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1)

    style_idx += 1
    legend_label = 'FBRL (20k)'
    print(legend_label)
    experiment_base = 'playpen_tree/all/reset_freq_exp/2k'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1)

    style_idx += 1
    legend_label = 'FBRL (200k)'
    print(legend_label)
    experiment_base = 'experiments/tests/playpen_reduced'
    experiments = [os.path.join('/iris/u/nsardana/reset_free_rl/', experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

  elif plot_type == 'playpen_no_data':
    max_index = int(2.5e6)
    plot_name = 'playpen_no_data.png'
    title = 'table-top rearrangement'

    style_idx += 1
    legend_label = 'VaPRL'
    print(legend_label)
    experiment_base = 'playpen_tree/all/dense/vaprl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='return', linewidth=3)

    style_idx += 1
    legend_label = 'VaPRL + noise'
    print(legend_label)
    experiment_base = 'playpen_tree/all/dense/vaprl_noise_factor'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='return', linewidth=3)

    # style_idx += 1
    # legend_label = 'FBRL'
    # print(legend_label)
    # experiment_base = 'playpen_tree/all/dense/fbrl'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='return')

  # SAWYER DOOR PLOTS ----------------------------
  elif plot_type == 'door':
    max_index = int(1.5e6)
    plot_name = 'sawyer_door'
    if mode == 'transfer':
      plot_name += '_transfer.png'
    elif mode == 'll':
      plot_name += '_ll.png'

    title = 'sawyer door closing'

    style_idx += 1
    legend_label = 'VaPRL'
    print(legend_label)
    experiment_base = 'sawyer_door_v2/vaprl_init_state/reduced_door_range'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    # style_idx += 1
    # legend_label = 'random'
    # print(legend_label)
    # experiment_base = '../reset_free_rl/experiments/sawyer_door_v2/vaprl_init_state/random_goal'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # if mode == 'transfer':
    #   plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    # elif mode == 'll':
    #   compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_door_v2/fbrl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'naive'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_door_v2/naive'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)
    
    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, '../../benchmark_results/r3l/door_longer')
    if mode == 'transfer':
      plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success',
              process_softlearning_data=True, linewidth=1.5)
    if mode == 'll':
      compute_lifelong_return(experiment_base, legend_label, plot_style=style_map[style_idx],
              process_softlearning_data=True, linewidth=1.5)

    style_idx = 19
    legend_label = 'oracle'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_door_v2/oracle'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1)

  elif plot_type == 'door_no_data':
    max_index = int(1.5e6)
    plot_name = 'sawyer_door_no_data.png'
    title = 'sawyer door closing'

    style_idx += 1
    legend_label = 'VaPRL'
    print(legend_label)
    experiment_base = 'sawyer_door_v1/bigger_door_range/no_data/curr/opt_values/goal_start'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=3)

    # style_idx += 1
    # legend_label = 'FBRL'
    # print(legend_label)
    # experiment_base = 'sawyer_door_v1/bigger_door_range/no_data/fixed'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'naive'
    # print(legend_label)
    # experiment_base = 'sawyer_door_v1/bigger_door_range/no_data/random'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'R3L'
    # print(legend_label)
    # experiment_base = os.path.join(base_path, 'r3l_baseline_results/gym/Sawyer/DoorClose-v0/sawyer_door_rf_no_data')
    # plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success', process_softlearning_data=True)

    style_idx += 1
    legend_label = 'VaPRL (+ demo)'
    print(legend_label)
    experiment_base = 'sawyer_door_v1/bigger_door_range/rf/3traj/curr'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=3)

    # style_idx = 19
    # legend_label = 'oracle'
    # print(legend_label)
    # experiment_base = 'sawyer_door_v1/bigger_door_range/no_data/oracle'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=0.8)

    # style_idx += 1
    # legend_label = 'R3L'
    # print(legend_label)
    # experiment_base = os.path.join(base_path, 'r3l_baseline_results/gym/Sawyer/DoorClose-v0/sawyer_door_rf_no_data')
    # plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success', process_softlearning_data=True)

  elif plot_type == 'kitchen':
    max_index = int(4.8e6)
    plot_name = 'kitchen'
    yaxis_type = 'return'
    if mode == 'transfer':
      plot_name += '_transfer.png'
    elif mode == 'll':
      plot_name += '_ll.png'

    title = 'kitchen'

    # to account for VaPRL, whose presence will always be felt
    style_idx += 1

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = '../../benchmark_results/kitchen_smallreplay/fbrl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'naive'
    print(legend_label)
    experiment_base = '../../benchmark_results/kitchen_smallreplay/naive'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)
    
    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, '../../benchmark_results/r3l/kitchen_v2')
    if mode == 'transfer':
      plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success',
              process_softlearning_data=True, linewidth=1.5)
    if mode == 'll':
      compute_lifelong_return(experiment_base, legend_label, plot_style=style_map[style_idx],
              process_softlearning_data=True, linewidth=1.5)

    style_idx = 19
    legend_label = 'oracle'
    print(legend_label)
    experiment_base = '../../benchmark_results/kitchen_smallreplay/oracle'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1)

  elif plot_type == 'minitaur':
    max_index = int(3e6)
    plot_name = 'minitaur'
    yaxis_type = 'return'
    if mode == 'transfer':
      plot_name += '_transfer.png'
    elif mode == 'll':
      plot_name += '_ll.png'

    title = 'minitaur'

    # to account for VaPRL, whose presence will always be felt
    style_idx += 1

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = '../../benchmark_results/minitaur/fbrl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'naive'
    print(legend_label)
    experiment_base = '../../benchmark_results/minitaur/naive'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)
    
    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, '../../benchmark_results/r3l/minitaur')
    if mode == 'transfer':
      plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success',
              process_softlearning_data=True, linewidth=1.5)
    if mode == 'll':
      compute_lifelong_return(experiment_base, legend_label, plot_style=style_map[style_idx],
              process_softlearning_data=True, linewidth=1.5)

    style_idx = 19
    legend_label = 'oracle'
    print(legend_label)
    experiment_base = '../../benchmark_results/minitaur/oracle'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1)
  
  elif plot_type == 'bulb':
    max_index = int(3e6)
    plot_name = 'bulb'
    yaxis_type = 'return'
    if mode == 'transfer':
      plot_name += '_transfer.png'
    elif mode == 'll':
      plot_name += '_ll.png'

    title = 'bulb'

    # to account for VaPRL, whose presence will always be felt
    style_idx += 1

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_bulb/fbrl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'naive'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_bulb/naive'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)
    
    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, '../../benchmark_results/r3l/hand')
    if mode == 'transfer':
      plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success',
              process_softlearning_data=True, linewidth=1.5)
    if mode == 'll':
      compute_lifelong_return(experiment_base, legend_label, plot_style=style_map[style_idx],
              process_softlearning_data=True, linewidth=1.5)

    style_idx = 19
    legend_label = 'oracle'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_bulb/oracle'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1)   
  # final plot config
  plt.grid(False)
  # plt.legend(prop={'size': 12}, loc=2)

  # fig = legend.figure
  # fig.canvas.draw()
  # bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  # fig.savefig(os.path.join(base_path, 'plots', 'legend.png'), dpi=200, bbox_inches=bbox)
  # exit()

  ax = plt.gca()
  # plt.xlabel('Steps in Training Environment', fontsize=18)
  if mode == 'transfer':
    plt.ylabel('Deployed Policy Evaluation', fontsize=18)
  elif mode == 'll':
    plt.ylabel('Continuing Policy Evaluation', fontsize=18)

  # plt.title(title, fontsize=20)
  print(list(ax.get_xticks()))
  # ax.set_xticks(list(ax.get_xticks())[1:-1])
  ax.set_yticks(list(ax.get_yticks())[2:-1])
  @ticker.FuncFormatter
  def major_formatter(x, pos):
    return '{:.1e}'.format(x).replace('+0', '')

  ax.xaxis.set_major_formatter(major_formatter)
  if mode == 'll':
    ax.yaxis.set_major_formatter(major_formatter)
  plt.setp(ax.get_xticklabels(), fontsize=10)
  plt.setp(ax.get_yticklabels(), fontsize=10)
  plt.savefig(os.path.join('experiments', 'plots', plot_name), dpi=600, bbox_inches='tight')
import matplotlib  # pylint: disable=g-import-not-at-top, unused-import
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
# plt.rcParams["text.usetex"] = True

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": ["Palatino"]})

import matplotlib.cm as cm  # pylint: disable=g-import-not-at-top, unused-import
import matplotlib.ticker as ticker

import numpy as np  # pylint: disable=g-import-not-at-top, reimported
import seaborn as sns  # pylint: disable=g-import-not-at-top, unused-import
import re  # pylint: disable=g-import-not-at-top, unused-import

import pickle as pkl  # pylint: disable=g-import-not-at-top, unused-import
import os  # pylint: disable=g-import-not-at-top, reimported
import csv

max_index = int(1e7)
custom = False

# alpha = 0 -> no smoothing, alpha=1 -> perfectly smoothed to initial value
def smooth(x, alpha):
  if isinstance(x, list):
    size = len(x)
  else:
    size = x.shape[0]
  for idx in range(1, size):
    x[idx] = (1 - alpha) * x[idx] + alpha * x[idx - 1]
  return x

def make_graph_with_variance(vals, x_interval, use_standard_error=True):
  data_x = []
  data_y = []
  global max_index
  num_seeds = 0
  for y_coords, eval_interval in zip(vals, x_interval):
    num_seeds += 1
    data_y.append(smooth(y_coords, 0.95))
    x_coords = [eval_interval * idx for idx in range(len(y_coords))]
    data_x.append(x_coords)

  plot_dict = {}
  cur_max_index = max_index
  for cur_x, cur_y in zip(data_x, data_y):
    cur_max_index = min(cur_max_index, cur_x[-1])
    # print(cur_x[-1])
  print(cur_max_index)

  for cur_x, cur_y in zip(data_x, data_y):
    for x, y in zip(cur_x, cur_y):
      if x <= cur_max_index:
        if x in plot_dict.keys():
          plot_dict[x].append(y)
        else:
          plot_dict[x] = [y]

  print('output at step:', cur_max_index)
  print(np.mean(plot_dict[cur_max_index]), np.std(plot_dict[cur_max_index]) / np.sqrt(5))

  index, means, stds = [], [], []
  for key in sorted(plot_dict.keys()):  # pylint: disable=g-builtin-op
    index.append(key)
    means.append(np.mean(plot_dict[key]))
    if use_standard_error:
      stds.append(np.std(plot_dict[key]) / np.sqrt(num_seeds))
    else:
      stds.append(np.std(plot_dict[key]))

  means = np.array(smooth(means, 0.8))
  stds = np.array(smooth(stds, 0.8))

  return index, means, stds

def np_custom_load(fname):
  return np.load(fname).astype(np.float32)

color_map = ['#73BA68', 'r', 'c', 'm', 'y', '#9A9C99', 'b']
style_map = []
for line_style in ['-', '--', '-.', ':']:
  style_map += [(color, line_style) for color in color_map]

def plot_call(experiment_paths,
              legend_label,
              plot_style,
              y_plot='return',
              process_softlearning_data=False,
              softlearning_idx=None,
              linewidth=1):
  """Outermost function for plotting graphs with variance."""
  if y_plot == 'return':
    if process_softlearning_data:
      if softlearning_idx is None:
        print('please write the code for getting the return index')
      base_path = os.path.dirname(experiment_paths)
      exp_name = experiment_paths.split('/')[-1]
      all_progress_files = []
      for path in os.listdir(base_path):
        if path.endswith(exp_name):
          all_progress_files.append(os.path.join(base_path, path, next(os.walk(os.path.join(base_path, path)))[1][0], 'progress.csv'))

      y_coords = []
      eval_interval = []
      for f in all_progress_files:
        data = open(f).read().splitlines()
        y_coords.append(np.array([float(entry[softlearning_idx]) for entry in list(csv.reader(data))[1:]]))
        eval_interval.append(2000)
    else:
      y_coords = [
          np_custom_load(os.path.join(experiment_path, 'eval', 'average_eval_return.npy'))
          for experiment_path in experiment_paths
      ]
      eval_interval = [
              np_custom_load(os.path.join(experiment_path, 'eval', 'eval_interval.npy'))
              for experiment_path in experiment_paths
      ]

  elif y_plot == 'success':
    if process_softlearning_data:
      # base_path = os.path.dirname(experiment_paths)
      # exp_name = experiment_paths.split('/')[-1]
      base_path = experiment_paths
      all_progress_files = []
      for path in os.listdir(base_path):
        if path.startswith('id='):
          # print(os.path.join(base_path, path, 'progress.csv'))
          # all_progress_files.append(os.path.join(base_path, path, next(os.walk(os.path.join(base_path, path)))[1][0], 'progress.csv'))
          all_progress_files.append(os.path.join(base_path, path, 'progress.csv'))

      if softlearning_idx is None:
        for idx, col_header in enumerate(open(all_progress_files[0], 'r').readline().split(',')):
          if col_header == 'evaluation_0/episode-reward-mean':
            softlearning_idx = idx
            break
        # print('softlearning eval idx:', softlearning_idx)

        for idx, col_header in enumerate(open(all_progress_files[0], 'r').readline().split(',')):
          if col_header == 'timestep':
            eval_interval_idx = idx
            break
        # print('eval interval idx:', eval_interval_idx)

      y_coords = []
      eval_interval = []
      for f in all_progress_files:
        data = open(f).read().splitlines()
        y_coords.append(10 * np.array([float(entry[softlearning_idx]) for entry in list(csv.reader(data))[1:]]))
        eval_interval.append(int(list(csv.reader(data))[1][eval_interval_idx]))

    else:
      y_coords = [
          np_custom_load(os.path.join(experiment_path, 'eval', 'average_eval_success.npy'))
          for experiment_path in experiment_paths
      ]
      # print(np.mean([np.where(val == 10.)[0][0] for val in y_coords]) * 10000, np.std([np.where(val == 10.)[0][0] for val in y_coords]) * 10000)
      eval_interval = [
            np_custom_load(os.path.join(experiment_path, 'eval', 'eval_interval.npy'))
            for experiment_path in experiment_paths
      ]

    # temporary fix
    y_coords = [y_entries / 10 for y_entries in y_coords]

  index, means, stds = make_graph_with_variance(y_coords, eval_interval)
  if plot_style[0] == '#9A9C99':
    plt.plot(index, means, color=plot_style[0], linestyle=plot_style[1], label=legend_label, linewidth=linewidth, 
      dashes=[6, 6])
  else:
    plt.plot(index, means, color=plot_style[0], linestyle=plot_style[1], label=legend_label, linewidth=linewidth)
  cur_color = plot_style[0]
  plt.fill_between(
      index, means - stds, means + stds, color=cur_color, alpha=0.2)

def compute_lifelong_return(experiments, 
                            legend_label,
                            plot_style,
                            process_softlearning_data=False,
                            linewidth=1):
  if process_softlearning_data:
      # base_path = os.path.dirname(experiment_paths)
      # exp_name = experiment_paths.split('/')[-1]
      base_path = experiments
      all_progress_files = []
      for path in os.listdir(base_path):
        if path.startswith('id='):
          # all_progress_files.append(os.path.join(base_path, path, next(os.walk(os.path.join(base_path, path)))[1][0], 'progress.csv'))
          all_progress_files.append(os.path.join(base_path, path, 'progress.csv'))

      for idx, col_header in enumerate(open(all_progress_files[0], 'r').readline().split(',')):
        if col_header == 'training_0/lifelong_return':
          softlearning_idx = idx
          break
      print('lifelong return idx:', softlearning_idx)

      for idx, col_header in enumerate(open(all_progress_files[0], 'r').readline().split(',')):
        if col_header == 'timestep':
          eval_interval_idx = idx
          break
      print('eval interval idx:', eval_interval_idx)

      ll_return_by_iter = []
      eval_interval = []
      for f in all_progress_files:
        data = open(f).read().splitlines()
        ll_return_by_iter.append(np.array([float(entry[softlearning_idx]) for entry in list(csv.reader(data))[1:]])*2)
        eval_interval.append(int(list(csv.reader(data))[1][eval_interval_idx])*2)

  else:    
    ll_return = []
    ll_return_by_iter = []
    for experiment in experiments:
      try:
        ll_return.append(np.load(os.path.join(experiment, 'eval/final_ll_return.npy')))
      except:
        print('final lifelong return not available')
      ll_return_by_iter.append(np.load(os.path.join(experiment, 'eval/lifelong_return_by_iter.npy')).astype(np.float32))
    
    eval_interval = [
        np_custom_load(os.path.join(experiment_path, 'eval', 'eval_interval.npy'))
        for experiment_path in experiments
    ]

  ll_return_by_iter = np.array(ll_return_by_iter)

  index, means, stds = make_graph_with_variance(ll_return_by_iter, eval_interval)
  if plot_style[0] == '#9A9C99':
    plt.plot(index, means, color=plot_style[0], linestyle=plot_style[1], label=legend_label, linewidth=linewidth, 
      dashes=[6, 6])
  else:
    plt.plot(index, means, color=plot_style[0], linestyle=plot_style[1], label=legend_label, linewidth=linewidth)
  cur_color = plot_style[0]
  plt.fill_between(
      index, means - stds, means + stds, color=cur_color, alpha=0.2)

  # print(ll_return)
  # print('lifelong return: %f (%f)' %(np.mean(ll_return), np.std(ll_return)))

if __name__ == '__main__':
  # basic configurations
  base_path = 'experiments'
  # base_path = '../benchmark_results'
  style_idx = -1

  plot_type = 'bulb'
  mode = 'll'

  # DHand Pickup Task ----------------------------
  if plot_type == 'dhand':
    max_index = int(7e6)
    plot_name = 'dhand_success.png'
    title = 'hand manipulation'
    yaxis_type = 'success'

    style_idx += 1
    legend_label = 'VaPRL'
    print(legend_label)
    experiment_base = 'sawyer_dhand/vaprl/hparam_sweep/off10_th-300'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [10, 11, 12, 13, 14, 15]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=3)

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = 'sawyer_dhand/fbrl_hand_relocation'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type)

    # style_idx += 1
    # legend_label = 'VaPRLs-200'
    # print(legend_label)
    # experiment_base = 'sawyer_dhand/vaprl/off10_th-200'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type)

    # style_idx += 1
    # legend_label = 'VaPRLg-200'
    # print(legend_label)
    # experiment_base = 'sawyer_dhand/vaprl/hparam_sweep/off10_th-200'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [10, 11, 12, 13, 14, 15]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type)

    # style_idx += 1
    # legend_label = 'VaPRLg-400'
    # print(legend_label)
    # experiment_base = 'sawyer_dhand/vaprl/hparam_sweep/off10_th-400'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [10, 11, 12, 13, 14, 15]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type)

    style_idx += 1
    legend_label = 'naive RL'
    print(legend_label)
    experiment_base = 'sawyer_dhand/naive'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type)

    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, 'r3l_baseline_results/gym/Sawyer/DhandPickupRandom-v0/sawyer_dhand_rf')
    plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type,
              process_softlearning_data=True,
              softlearning_idx=494 if yaxis_type == 'return' else 498)

    style_idx = 19
    legend_label = 'oracle RL'
    print(legend_label)
    experiment_base = 'sawyer_dhand/oracle'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [2, 3, 4, 6, 7]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=0.8)

  elif plot_type == 'tabletop':
    max_index = int(2.5e6)
    plot_name = 'tabletop'
    if mode == 'transfer':
      plot_name += '_transfer.png'
    elif mode == 'll':
      plot_name += '_ll.png'

    title = 'table-top reorganiztion'

    style_idx += 1
    legend_label = 'VaPRL'
    print(legend_label)
    experiment_base = 'tabletop_manipulation/vaprl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = 'tabletop_manipulation/fbrl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'naive'
    print(legend_label)
    experiment_base = 'tabletop_manipulation/naive_ep_terminate'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, '../../benchmark_results/r3l/r3l_tabletop')
    if mode == 'transfer':
      plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success',
              process_softlearning_data=True, linewidth=1.5)
    if mode == 'll':
      compute_lifelong_return(experiment_base, legend_label, plot_style=style_map[style_idx],
              process_softlearning_data=True, linewidth=1.5)

    style_idx = 19
    legend_label = 'oracle'
    print(legend_label)
    experiment_base = 'playpen_tree/rc_o-rc_k-rc_p-rc_b/oracle_reset'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1)

  elif plot_type == 'peg':
    max_index = int(2e6)
    plot_name = 'sawyer_peg'
    if mode == 'transfer':
      plot_name += '_transfer.png'
    elif mode == 'll':
      plot_name += '_ll.png'

    title = 'sawyer peg insertion'

    style_idx += 1
    legend_label = 'VaPRL'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_peg/vaprl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_peg/fbrl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'naive'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_peg/naive'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)
    
    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, '../../benchmark_results/r3l/peg_longer')
    if mode == 'transfer':
      plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success',
              process_softlearning_data=True, linewidth=1.5)
    if mode == 'll':
      compute_lifelong_return(experiment_base, legend_label, plot_style=style_map[style_idx],
              process_softlearning_data=True, linewidth=1.5)

    style_idx = 19
    legend_label = 'oracle'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_peg/oracle'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1)
  
  # PLAYPEN PLOTS --------------------------------
  elif plot_type == 'playpen':
    max_index = int(2.5e6)
    plot_name = 'playpen_ablation.png'
    title = 'table-top rearrangement'

    style_idx += 1
    legend_label = 'VaPRL (200k)'
    print(legend_label)
    experiment_base = 'playpen_tree/all/curriculum_offline_relabel/test'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=3)

    # style_idx += 1
    # legend_label = 'VaPRL + reset'
    # print(legend_label)
    # experiment_base = 'playpen_tree/rc_o-rc_k-rc_p-rc_b/curriculum_reset/th1.0'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=3)
    
    # style_idx += 1
    # legend_label = 'random'
    # print(legend_label)
    # experiment_base = 'playpen_tree/all/random_curriculum'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'naive RL'
    # print(legend_label)
    # experiment_base = 'playpen_tree/all/naive'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'FBRL'
    # print(legend_label)
    # experiment_base = 'playpen_tree/all/rf_fixed'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'uniform / R3L + reset'
    # print(legend_label)
    # experiment_base = 'playpen_tree/rc_o-rc_k-rc_p-rc_b/uniform_reset'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'uniform / R3L + reset (big)'
    # print(legend_label)
    # experiment_base = 'playpen_tree/rc_o-rc_k-rc_p-rc_b/uniform_reset/bigger_square'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'R3L'
    # print(legend_label)
    # experiment_base = os.path.join(base_path, 'r3l_baseline_results/gym/PlayPen/OneObject4Goals-v0/playpen_tree_rf')
    # plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success', process_softlearning_data=True)

    style_idx += 1
    # style_idx = 19
    legend_label = 'FBRL (200)'
    print(legend_label)
    experiment_base = 'playpen_tree/rc_o-rc_k-rc_p-rc_b/oracle_reset'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1)


    style_idx += 1
    legend_label = 'FBRL (2k)'
    print(legend_label)
    experiment_base = 'playpen_tree/all/reset_freq_exp/20k'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1)

    style_idx += 1
    legend_label = 'FBRL (20k)'
    print(legend_label)
    experiment_base = 'playpen_tree/all/reset_freq_exp/2k'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1)

    style_idx += 1
    legend_label = 'FBRL (200k)'
    print(legend_label)
    experiment_base = 'experiments/tests/playpen_reduced'
    experiments = [os.path.join('/iris/u/nsardana/reset_free_rl/', experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

  elif plot_type == 'playpen_no_data':
    max_index = int(2.5e6)
    plot_name = 'playpen_no_data.png'
    title = 'table-top rearrangement'

    style_idx += 1
    legend_label = 'VaPRL'
    print(legend_label)
    experiment_base = 'playpen_tree/all/dense/vaprl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='return', linewidth=3)

    style_idx += 1
    legend_label = 'VaPRL + noise'
    print(legend_label)
    experiment_base = 'playpen_tree/all/dense/vaprl_noise_factor'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='return', linewidth=3)

    # style_idx += 1
    # legend_label = 'FBRL'
    # print(legend_label)
    # experiment_base = 'playpen_tree/all/dense/fbrl'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='return')

  # SAWYER DOOR PLOTS ----------------------------
  elif plot_type == 'door':
    max_index = int(1.5e6)
    plot_name = 'sawyer_door'
    if mode == 'transfer':
      plot_name += '_transfer.png'
    elif mode == 'll':
      plot_name += '_ll.png'

    title = 'sawyer door closing'

    style_idx += 1
    legend_label = 'VaPRL'
    print(legend_label)
    experiment_base = 'sawyer_door_v2/vaprl_init_state/reduced_door_range'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    # style_idx += 1
    # legend_label = 'random'
    # print(legend_label)
    # experiment_base = '../reset_free_rl/experiments/sawyer_door_v2/vaprl_init_state/random_goal'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # if mode == 'transfer':
    #   plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    # elif mode == 'll':
    #   compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_door_v2/fbrl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'naive'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_door_v2/naive'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)
    
    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, '../../benchmark_results/r3l/door_longer')
    if mode == 'transfer':
      plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success',
              process_softlearning_data=True, linewidth=1.5)
    if mode == 'll':
      compute_lifelong_return(experiment_base, legend_label, plot_style=style_map[style_idx],
              process_softlearning_data=True, linewidth=1.5)

    style_idx = 19
    legend_label = 'oracle'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_door_v2/oracle'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=1)

  elif plot_type == 'door_no_data':
    max_index = int(1.5e6)
    plot_name = 'sawyer_door_no_data.png'
    title = 'sawyer door closing'

    style_idx += 1
    legend_label = 'VaPRL'
    print(legend_label)
    experiment_base = 'sawyer_door_v1/bigger_door_range/no_data/curr/opt_values/goal_start'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=3)

    # style_idx += 1
    # legend_label = 'FBRL'
    # print(legend_label)
    # experiment_base = 'sawyer_door_v1/bigger_door_range/no_data/fixed'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'naive'
    # print(legend_label)
    # experiment_base = 'sawyer_door_v1/bigger_door_range/no_data/random'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success')

    # style_idx += 1
    # legend_label = 'R3L'
    # print(legend_label)
    # experiment_base = os.path.join(base_path, 'r3l_baseline_results/gym/Sawyer/DoorClose-v0/sawyer_door_rf_no_data')
    # plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success', process_softlearning_data=True)

    style_idx += 1
    legend_label = 'VaPRL (+ demo)'
    print(legend_label)
    experiment_base = 'sawyer_door_v1/bigger_door_range/rf/3traj/curr'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=3)

    # style_idx = 19
    # legend_label = 'oracle'
    # print(legend_label)
    # experiment_base = 'sawyer_door_v1/bigger_door_range/no_data/oracle'
    # experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2]]
    # plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot='success', linewidth=0.8)

    # style_idx += 1
    # legend_label = 'R3L'
    # print(legend_label)
    # experiment_base = os.path.join(base_path, 'r3l_baseline_results/gym/Sawyer/DoorClose-v0/sawyer_door_rf_no_data')
    # plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success', process_softlearning_data=True)

  elif plot_type == 'kitchen':
    max_index = int(4.8e6)
    plot_name = 'kitchen'
    yaxis_type = 'return'
    if mode == 'transfer':
      plot_name += '_transfer.png'
    elif mode == 'll':
      plot_name += '_ll.png'

    title = 'kitchen'

    # to account for VaPRL, whose presence will always be felt
    style_idx += 1

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = '../../benchmark_results/kitchen_smallreplay/fbrl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'naive'
    print(legend_label)
    experiment_base = '../../benchmark_results/kitchen_smallreplay/naive'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)
    
    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, '../../benchmark_results/r3l/kitchen_v2')
    if mode == 'transfer':
      plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success',
              process_softlearning_data=True, linewidth=1.5)
    if mode == 'll':
      compute_lifelong_return(experiment_base, legend_label, plot_style=style_map[style_idx],
              process_softlearning_data=True, linewidth=1.5)

    style_idx = 19
    legend_label = 'oracle'
    print(legend_label)
    experiment_base = '../../benchmark_results/kitchen_smallreplay/oracle'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1)

  elif plot_type == 'minitaur':
    max_index = int(3e6)
    plot_name = 'minitaur'
    yaxis_type = 'return'
    if mode == 'transfer':
      plot_name += '_transfer.png'
    elif mode == 'll':
      plot_name += '_ll.png'

    title = 'minitaur'

    # to account for VaPRL, whose presence will always be felt
    style_idx += 1

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = '../../benchmark_results/minitaur/fbrl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'naive'
    print(legend_label)
    experiment_base = '../../benchmark_results/minitaur/naive'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)
    
    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, '../../benchmark_results/r3l/minitaur')
    if mode == 'transfer':
      plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success',
              process_softlearning_data=True, linewidth=1.5)
    if mode == 'll':
      compute_lifelong_return(experiment_base, legend_label, plot_style=style_map[style_idx],
              process_softlearning_data=True, linewidth=1.5)

    style_idx = 19
    legend_label = 'oracle'
    print(legend_label)
    experiment_base = '../../benchmark_results/minitaur/oracle'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1)
  
  elif plot_type == 'bulb':
    max_index = int(3e6)
    plot_name = 'bulb'
    yaxis_type = 'return'
    if mode == 'transfer':
      plot_name += '_transfer.png'
    elif mode == 'll':
      plot_name += '_ll.png'

    title = 'bulb'

    # to account for VaPRL, whose presence will always be felt
    style_idx += 1

    style_idx += 1
    legend_label = 'FBRL'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_bulb/fbrl'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)

    style_idx += 1
    legend_label = 'naive'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_bulb/naive'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1.5)
    elif mode == 'll':
      compute_lifelong_return(experiments, legend_label, plot_style=style_map[style_idx], linewidth=1.5)
    
    style_idx += 1
    legend_label = 'R3L'
    print(legend_label)
    experiment_base = os.path.join(base_path, '../../benchmark_results/r3l/hand')
    if mode == 'transfer':
      plot_call(experiment_base, legend_label, plot_style=style_map[style_idx], y_plot='success',
              process_softlearning_data=True, linewidth=1.5)
    if mode == 'll':
      compute_lifelong_return(experiment_base, legend_label, plot_style=style_map[style_idx],
              process_softlearning_data=True, linewidth=1.5)

    style_idx = 19
    legend_label = 'oracle'
    print(legend_label)
    experiment_base = '../../benchmark_results/sawyer_bulb/oracle'
    experiments = [os.path.join(base_path, experiment_base, str(random_seed)) for random_seed in [0, 1, 2, 3, 4]]
    if mode == 'transfer':
      plot_call(experiments, legend_label, plot_style=style_map[style_idx], y_plot=yaxis_type, linewidth=1)   
  # final plot config
  plt.grid(False)
  # plt.legend(prop={'size': 12}, loc=2)

  # fig = legend.figure
  # fig.canvas.draw()
  # bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  # fig.savefig(os.path.join(base_path, 'plots', 'legend.png'), dpi=200, bbox_inches=bbox)
  # exit()

  ax = plt.gca()
  # plt.xlabel('Steps in Training Environment', fontsize=18)
  if mode == 'transfer':
    plt.ylabel('Deployed Policy Evaluation', fontsize=18)
  elif mode == 'll':
    plt.ylabel('Continuing Policy Evaluation', fontsize=18)

  # plt.title(title, fontsize=20)
  print(list(ax.get_xticks()))
  # ax.set_xticks(list(ax.get_xticks())[1:-1])
  ax.set_yticks(list(ax.get_yticks())[2:-1])
  @ticker.FuncFormatter
  def major_formatter(x, pos):
    return '{:.1e}'.format(x).replace('+0', '')

  ax.xaxis.set_major_formatter(major_formatter)
  if mode == 'll':
    ax.yaxis.set_major_formatter(major_formatter)
  plt.setp(ax.get_xticklabels(), fontsize=10)
  plt.setp(ax.get_yticklabels(), fontsize=10)
  plt.savefig(os.path.join('experiments', 'plots', plot_name), dpi=600, bbox_inches='tight')

