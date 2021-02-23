import numpy as np
from dm_control import suite
from cvrender import Video
from wrappers import dm_persistent_wrapper

# Load one task:
env = suite.load(domain_name="cheetah", task_name="run", visualize_reward=True)
env = dm_persistent_wrapper.Wrapper(env, 500)

# Setup video writer - mp4 at 30 fps
video = Video("video.mp4", 480, 600)

# Reset data
action_spec = env.action_spec()
time_step = env.reset()

# Step through an episode and print out reward, discount and observation.
i=0
for i in range(1200):
#iwhile not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
    time_step = env.step(action)
    i+=1
    print("Time step: ", i)
    video.record(env)
    print(time_step.reward, time_step.discount, time_step.observation)

video.save()
