import numpy as np
from dm_control import suite
from cvrender import Video

# Load one task:
env = suite.load(domain_name="humanoid", task_name="stand", visualize_reward=True)
# Iterate over a task set:
for domain_name, task_name in suite.BENCHMARKING:
    env = suite.load(domain_name, task_name)


# Setup video writer - mp4 at 30 fps
video = Video("video2.mp4", 480, 600)

# Reset data
action_spec = env.action_spec()
time_step = env.reset()

# Step through an episode and print out reward, discount and observation.
while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
    time_step = env.step(action)
    #pixels = env.physics.render()
    video.record(env)
    print(time_step.reward, time_step.discount, time_step.observation)

video.save()
