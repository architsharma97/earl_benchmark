import cv2
from dm_control import suite

class Video:
    def __init__(self, filename, height, width):
        self.filename = filename
        self.height = height
        self.width = width
        self.video = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (self.width, self.height))

    def grab_frame(self, env):
        # Get RGB rendering of env
        rgbArr = env.physics.render(self.height, self.width, camera_id=0)
        # Convert to BGR for use with OpenCV
        return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)

    def record(self, env):
        self.video.write(self.grab_frame(env))
    
    def save(self):
        self.video.release()
