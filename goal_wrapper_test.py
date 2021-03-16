import numpy as np
from dm_control import suite
from cvrender import Video
from wrappers import dm_goalcond_wrapper

# Test: Aim for speed of 5
def reward_func(cheetah_env, goal):
    return -((cheetah_env._physics.speed() - goal)**2)

# Load one task:
env = suite.load(domain_name="cheetah", task_name="run", visualize_reward=True)
env = dm_goalcond_wrapper.Wrapper(env, 5, reward_func)

# Setup video writer - mp4 at 30 fps
video = Video("cheetah-goalcond-test.mp4", 480, 600)

# Reset data
action_spec = env.action_spec()
time_step = env.reset()

# Step through an episode and print out reward, discount and observation.
i=0
while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
    time_step = env.step(action)
    i+=1
    print("Time step: ", i)
    video.record(env)
    print(time_step.reward, time_step.discount, time_step.observation)

video.save()
