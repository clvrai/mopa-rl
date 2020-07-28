import sys
print(sys.executable)

import os
import glob
import pickle
from collections import namedtuple, defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as plticker



def save_fig(filename, file_format='pdf', tight=True, **kwargs):
    if tight:
        plt.tight_layout()
    filename = "./figures/{}.{}".format(filename, file_format)
    filename = filename.replace(' ', '-')
    plt.savefig(filename, format=file_format, dpi=1000, **kwargs)


def draw_line(log, method, methods_label, avg_step=3, mean_std=False, max_step=None, max_y=None, min_y=None, x_scale=1.0, ax=None, idx=0,
             smooth_steps=10, num_points=50, linestyle='-', no_fill=False):
    steps = {}
    values = {}
    max_step = max_step * x_scale
    seeds = log[method].keys()
    for seed in seeds:
        step = np.array(log[method][seed].steps)
        value = np.array(log[method][seed].values)

        if max_step:
            max_step = min(max_step, step[-1])
        else:
            max_step = step[-1]

        steps[seed] = step
        values[seed] = value

    data = []
    for seed in seeds:
        for i in range(len(steps[seed])):
            if steps[seed][i] <= max_step:
                data.append((steps[seed][i], values[seed][i]))

    data.sort()
    x_data = []
    y_data = []
    for step, value in data:
        x_data.append(step)
        y_data.append(value)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # smoothing


    n = len(x_data)
    print(n)
    ns = smooth_steps

    x_data = x_data[:n // ns * ns].reshape(-1, ns)
    y_data = y_data[:n // ns * ns].reshape(-1, ns)

    x_data, y_data = np.nanmean(x_data, axis=1), np.nanmean(y_data, axis=1)

    non_nan_index = np.invert(np.isnan(y_data))
    y_data = y_data[non_nan_index]
    x_data = x_data[non_nan_index]

#     y_data = np.nan_to_num(y_data)

    min_y = np.min(y_data)
    max_y = np.max(y_data)
    #l = sns.lineplot(x=x_data, y=y_data)
    #return l, min_y, max_y

    # filling
    if not no_fill:
        n = len(x_data)
        avg_step = int(n // num_points)
        print(avg_step)

        x_data = x_data[:n // avg_step * avg_step].reshape(-1, avg_step)
        y_data = y_data[:n // avg_step * avg_step].reshape(-1, avg_step)

        std_y = np.std(y_data, axis=1)

        avg_x, avg_y = np.mean(x_data, axis=1), np.mean(y_data, axis=1)
    else:
        avg_x, avg_y = x_data, y_data
    if not no_fill:
        ax.fill_between(avg_x, np.clip(avg_y - std_y, min_y, max_y), np.clip(avg_y + std_y, min_y, max_y),
                        alpha=0.1, color='C%d' % idx)#, facecolor=facecolor)

    l = ax.plot(avg_x, avg_y, label=methods_label[method])
    plt.setp(l, linewidth=2, color='C%d' % idx, linestyle=linestyle)

    return l, min_y, max_y


def draw_graph(logs, title, methods_label, xlabel='Step', ylabel='Success', legend=False, put_title=True,
               mean_std=False, min_step=0, max_step=None, min_y=None, max_y=None, smooth_steps=10,
               num_points=50, no_fill=False, num_x_tick=5, legend_loc=2, bbox_to_anchor=None, fontsize='small', label_fontsize='large'):
    max_value = -9999
    min_value = 9999

    fig, ax = plt.subplots(figsize=(5, 4))

    lines = []
    methods = methods_label.keys()
    num_colors = len(methods)
    for idx, method in enumerate(methods):
        seeds = logs[method].keys()
        print('method: ', method, 'seed: ', seeds)
        if len(seeds) == 0:
            continue

        linestyle = '--' if 'final_d' in method else '-'
        l_, min_, max_ = draw_line(logs, method, methods_label, mean_std=mean_std, max_step=max_step, max_y=max_y, min_y=min_y,
                                   x_scale=1.0, ax=ax, idx=num_colors - idx - 1,
                                   smooth_steps=smooth_steps, num_points=num_points, linestyle=linestyle,
                                   no_fill=no_fill)
        print(min_value, max_value)
        #lines += l_
        max_value = max(max_value, max_)
        min_value = min(min_value, min_)

    if min_y == None:
        min_y = int(min_value - 1)
    if max_y == None:
        max_y = max_value
        #max_y = int(max_value + 1)
    print('min_y', min_y, 'max_y', max_y)

    if max_y == 1:
        plt.yticks(np.arange(min_y, max_y + 0.1, 0.2))
    else:
        if max_y > 1:

            plt.yticks(np.arange(min_y, max_y, (max_y - min_y) / 5))
        elif max_y > 0.8:
            plt.yticks(np.arange(0, 1.0, 0.2))
        elif max_y > 0.5:
            plt.yticks(np.arange(0, 0.8, 0.2))
        elif max_y > 0.3:
            plt.yticks(np.arange(0, 0.5, 0.1))
        elif max_y > 0.2:
            plt.yticks(np.arange(0, 0.4, 0.1))
        elif max_y > 0:
            plt.yticks(np.arange(0, 0.2, 0.05))
        else:
            pass
#             plt.yticks(np.arange(min_y, max_y, (max_y-min_y)//50))

    plt.xticks(np.arange(min_step, max_step + 0.1, (max_step - min_step) / num_x_tick))
    ax.grid(b=True, which='major', color='lightgray', linestyle='--')

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    #plt.ylabel(ylabel, fontsize='large')
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    if min_y >= 0:
        ax.set_ylim(bottom=-0.01)
        ax.set_xlim(min_step, max_step)

    if legend:
        if bbox_to_anchor is not None:
            plt.legend(fontsize=fontsize, loc=legend_loc, bbox_to_anchor=bbox_to_anchor)
        else:
            plt.legend(fontsize=fontsize, loc=legend_loc)
        #labs = [l.get_label() for l in lines]
        #plt.legend(lines, labs, fontsize='small', loc=2)
        #plt.legend()#bbox_to_anchor=(1.03, 0.73))

    if put_title:
        plt.title(title, y=1.00, fontsize='x-large')
    save_fig(title + '_' + ylabel)


def build_logs(methods_label, runs, y_axis='train_ep/episode_success', seeds=None, use_global_step=False):
    Log = namedtuple('Log', ['values', 'steps'])
    logs = defaultdict(dict)
    for run_name in methods_label.keys():
        for run in runs:
            if run_name in run.name:
                x = run.history()
                non_nan_index = np.invert(np.isnan(x[y_axis]))
                values = x[y_axis][non_nan_index]
                x = run.history(samples=10000)

                values = x[y_axis]
#                 steps = x['_step'][non_nan_index] / 1000000
                if use_global_step:
                    steps = x['global_step'] / 1000000
                else:
                    steps = x['_step'] / 1000000
                name, seed = run.name.rsplit('.', 1)
                if seeds is not None:
                    if seed not in seeds[run_name]:
                        continue
                logs[run_name][seed] = Log(values, steps)
                print(run_name, seed, run, len(steps))
    return logs
