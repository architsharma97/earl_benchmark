import os
import tensorflow as tf
from tensorflow.core.util import event_pb2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


def smooth(x, y, n):
    xnew = np.linspace(x.min(), x.max(), n)
    spl = make_interp_spline(x, y, k=3)
    y_smooth=spl(xnew)
    return xnew, y_smooth


def getfile(trial, experiment):
    prefix = base + env + "-1000-"
    suffix = "-" + str(trial) + "/eval/"
    path = prefix + str(experiment) + suffix
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if len(files) == 0:
        raise Exception("Error: Cannot find run file.")
    if len(files) != 1:
        raise Exception("Error: Too many files in directory.")
    return str(files[0])


def readdata(trial, experiment):
    datafile = getfile(trial, experiment)
    x, y = [], []
    # Parse Record File For Step and Metric Data
    serialized_examples = tf.data.TFRecordDataset(datafile)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for value in event.summary.value:
            t = tf.make_ndarray(value.tensor)
            if (value.tag == "Metrics/AverageReturn"):
                x.append(event.step)
                y.append(t.tolist())

    x, y = np.asarray(x), np.asarray(y)
    if smoothbool:
        x, y = smooth(x, y, smoothfactor)
    return x, y

    
def plot():
    for exp in experiments:
        x_list, y_list = [], []
        for trial in trials:
            x, y = readdata(trial, exp)
            x_list.append(x)
            y_list.append(y)
        x_matrix = np.stack(x_list, axis=0)
        y_matrix = np.stack(y_list, axis=0)
        y_mean = np.mean(y_matrix, axis=0)
        y_std = np.std(y_matrix, axis=0)        
        # Plot data
        plt.plot(x_matrix[0], y_mean, label="H_t="+str(exp), zorder=2)
        plt.fill_between(x_matrix[0], y_mean-y_std, y_mean+y_std, alpha=0.2)
        
base = "/iris/u/nsardana/moreresults/"
env = "cheetah"
task = "run"
trials = [2, 3, 4, 5, 6]
experiments=[1000, 2000, 5000, 10000, 50000]

fig = plt.figure()
savefile = env + "results-mean"

# Whether to smooth or not
smoothbool = False
smoothfactor = 50
if smoothbool:
    savefile = savefile + "-smooth"


# Call main plot method
plot()


plt.title(label=env + " " + task + " Average Return vs. Steps")
plt.xlabel("Steps")
plt.ylabel("Average Return")
plt.legend()

fig.savefig(savefile + ".png", dpi=600)



