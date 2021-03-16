import os
import tensorflow as tf
from tensorflow.core.util import event_pb2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

def smooth(x, y, n):
    x_np = np.asarray(x)
    y_np = np.asarray(y)
    xnew = np.linspace(x_np.min(), x_np.max(), n)
    spl = make_interp_spline(x_np, y_np, k=3)
    y_smooth=spl(xnew)
    return xnew, y_smooth

#trial1="../results/cheetah-1000-1000/eval/events.out.tfevents.1614769015.iris4.stanford.edu.3998947.12.v2"
#trial2="../results/cheetah-1000-2000/eval/events.out.tfevents.1615199274.iris2.stanford.edu.2651917.12.v2"
#trial3="../moreresults/cheetah-1000-5000/eval/events.out.tfevents.1615199593.iris2.stanford.edu.2653308.12.v2"
#trial4="../moreresults/cheetah-1000-10000/eval/events.out.tfevents.1615200029.iris2.stanford.edu.2654970.12.v2"

base = "/iris/u/nsardana/moreresults/"
env = "fish"
task = "swim"
runNum = "1"
prefix = base + env + "-1000-"
suffix = "-" + runNum + "/eval/"

# Quadruped Fetch Run 1
#trial1 = prefix + "1000" + suffix + "events.out.tfevents.1615285547.iris2.stanford.edu.3020331.12.v2"
#trial2 = prefix + "2000" + suffix + "events.out.tfevents.1615285550.iris4.stanford.edu.1867708.12.v2"
#trial3 = prefix + "5000" + suffix + "events.out.tfevents.1615285550.iris4.stanford.edu.1867865.12.v2"
#trial4 = prefix + "10000" + suffix + "events.out.tfevents.1615285550.iris4.stanford.edu.1867863.12.v2"
#trial5 = prefix + "50000" + suffix + "events.out.tfevents.1615285550.iris4.stanford.edu.1867867.12.v2"

# Fish Swim Run 1
trial1 = prefix + "1000" + suffix + "events.out.tfevents.1615423565.iris2.stanford.edu.3576957.12.v2"
trial2 = prefix + "2000" + suffix + "events.out.tfevents.1615423565.iris2.stanford.edu.3576960.12.v2"
trial3 = prefix + "5000" + suffix + "events.out.tfevents.1615423565.iris2.stanford.edu.3576958.12.v2"
trial4 = prefix + "10000" + suffix + "events.out.tfevents.1615423565.iris2.stanford.edu.3577028.12.v2"
trial5 = prefix + "50000" + suffix + "events.out.tfevents.1615423565.iris2.stanford.edu.3577083.12.v2"

trials=[trial1, trial2, trial3, trial4, trial5]

fig = plt.figure()
labels=["H_t=1000", "H_t=2000", "H_t=5000", "H_t=10000", "H_t=50000"]
plt.title(label=env + " " + task + " Average Return vs. Steps")
figurefilename = env + "results" + runNum + ".png"

# Whether to smooth or not
smoothbool = True
smoothfactor = 50
if (smoothbool):
    figurefilename = env + "results" + runNum + "smooth.png"

for i in range(len(trials)):
    print("TRIAL:", i)
    data = trials[i]
    x = []
    y = []
    serialized_examples = tf.data.TFRecordDataset(data)
    for serialized_example in serialized_examples:
        event = event_pb2.Event.FromString(serialized_example.numpy())
        for value in event.summary.value:
            t = tf.make_ndarray(value.tensor)
            if (value.tag == "Metrics/AverageReturn"):
                #print(event.step, t)
                x.append(event.step)
                y.append(t.tolist())
    
    print(x)
    print(y)
    if (smoothbool):
        x, y = smooth(x, y, smoothfactor)
    plt.plot(x, y, label=labels[i])

plt.xlabel("Steps")
plt.ylabel("Average Return")
plt.legend()

fig.savefig(figurefilename, dpi=600)


