import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
# plt.rcParams["text.usetex"] = True

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": ["Palatino"]})

import matplotlib.cm as cm
import matplotlib.ticker as ticker

import numpy as np
import seaborn as sns
import re

import pickle as pkl
import os
import csv

# alpha = 0 -> no smoothing, alpha=1 -> perfectly smoothed to initial value
def smooth(x, alpha):
  if isinstance(x, list):
    size = len(x)
  else:
    size = x.shape[0]
  for idx in range(1, size):
    x[idx] = (1 - alpha) * x[idx] + alpha * x[idx - 1]
  return x

def make_graph_with_variance(vals, x_interval, max_index=int(1e8), use_standard_error=True):
  data_x = []
  data_y = []
  num_seeds = 0

  for y_coords, eval_interval in zip(vals, x_interval):
    num_seeds += 1
    data_y.append(smooth(y_coords, 0))
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
  print(np.mean(plot_dict[cur_max_index]), np.std(plot_dict[cur_max_index]) / np.sqrt(num_seeds))

  index, means, stds = [], [], []
  for key in sorted(plot_dict.keys()):  # pylint: disable=g-builtin-op
    index.append(key)
    means.append(np.mean(plot_dict[key]))
    if use_standard_error:
      stds.append(np.std(plot_dict[key]) / np.sqrt(num_seeds))
    else:
      stds.append(np.std(plot_dict[key]))

  means = np.array(smooth(means, 0.95))
  stds = np.array(smooth(stds, 0.95))

  return index, means, stds

def np_custom_load(fname):
  return np.load(fname).astype(np.float32)

def plotter(experiment_paths, mode, max_index=int(1e8), **plot_config):
  """Outermost function for plotting graphs with variance."""
  if mode == 'deployment':
    y_coords = [
        np_custom_load(os.path.join(experiment_path, 'deployed_eval.npy'))
        for experiment_path in experiment_paths
    ]
  elif mode == 'continuing':
    y_coords = [
        np_custom_load(os.path.join(experiment_path, 'continuing_eval.npy'))
        for experiment_path in experiment_paths
    ]

  eval_interval = [
        np_custom_load(os.path.join(experiment_path, 'eval_interval.npy'))
        for experiment_path in experiment_paths
  ]

  index, means, stds = make_graph_with_variance(y_coords, eval_interval, max_index=max_index)
  plt.plot(index, means, **plot_config)
  plt.fill_between(
      index, means - stds, means + stds, color=plot_config.get('color'), alpha=0.2)

if __name__ == '__main__':
  # basic configurations
  base_path = '../../../benchmark_results'
  plot_type = 'tabletop'
  mode = 'deployment'

  # color_map = ['#73BA68', 'r', 'c', 'm', 'y', '#9A9C99', 'b']
  # style_map = []
  # for line_style in ['-', '--', '-.', ':']:
  #   style_map += [(color, line_style) for color in color_map]

  plot_config = {
    'VaPRL':
      {'color':'#73BA68', 'linestyle':'-', 'label':'VaPRL', 'linewidth':1.5},
    'FBRL':
      {'color':'r', 'linestyle':'-', 'label':'FBRL', 'linewidth':1.5},
    'naive':
      {'color':'c', 'linestyle':'-', 'label':'naive', 'linewidth':1.5},
    'R3L':
      {'color':'m', 'linestyle':'-', 'label':'R3L', 'linewidth':1.5},
    'oracle':
      {'color':'#9A9C99', 'linestyle':'-', 'label':'oracle', 'linewidth':1, 'dashes':[6, 6]},
  }

  if plot_type == 'tabletop':
    max_index = int(2.5e6)
    plot_name = 'tabletop'
    title = 'tabletop organiztion'

    base_path = os.path.join(base_path, 'tabletop_organization')
    for method in ['VaPRL', 'FBRL', 'naive', 'R3L', 'oracle']:
      print(method)
      if mode == 'continuing' and method == 'oracle':
        continue
      experiment_base = os.path.join(base_path, method.lower())
      experiments = [os.path.join(experiment_base, str(run_id)) for run_id in [0, 1, 2, 3, 4]]
      plotter(experiments, mode=mode, max_index=max_index, **plot_config[method])
      
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
  
  elif plot_type == 'door':
    max_index = int(4e6)
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
  plt.legend(prop={'size': 12}, loc=2)

  # fig = legend.figure
  # fig.canvas.draw()
  # bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  # fig.savefig(os.path.join(base_path, 'plots', 'legend.png'), dpi=200, bbox_inches=bbox)
  # exit()

  if mode == 'deployment':
    plot_name += '_deployment.png'
  elif mode == 'continuing':
    plot_name += '_continuing.png'

  ax = plt.gca()
  # plt.xlabel('Steps in Training Environment', fontsize=18)
  if mode == 'deployment':
    plt.ylabel('Deployed Policy Evaluation', fontsize=18)
  elif mode == 'continuing':
    plt.ylabel('Continuing Policy Evaluation', fontsize=18)

  # plt.title(title, fontsize=20)
  print(list(ax.get_xticks()))
  # ax.set_xticks(list(ax.get_xticks())[1:-1])
  ax.set_yticks(list(ax.get_yticks())[2:-1])
  @ticker.FuncFormatter
  def major_formatter(x, pos):
    return '{:.1e}'.format(x).replace('+0', '')

  ax.xaxis.set_major_formatter(major_formatter)
  if mode == 'continuing':
    ax.yaxis.set_major_formatter(major_formatter)
  plt.setp(ax.get_xticklabels(), fontsize=10)
  plt.setp(ax.get_yticklabels(), fontsize=10)
  plt.savefig(os.path.join(base_path, plot_name), dpi=600, bbox_inches='tight')
