import cv2
import os
import numpy as np
from os.path import isfile, join
from model import Image
import tensorflow as tf

class Video(object):
    def __init__(self, video_path, style_path, img_h, img_w, lr, n_iters, fps):
        self.video_path = video_path
        self.style_path = style_path
        self.img_height = img_h
        self.img_width = img_w
        self.lr = lr
        self.n_iters = n_iters
        self.fps = fps
    
    def _stylize_frame(self, img_path, frame_idx, prev_frame):
        tf.reset_default_graph()
        image_model = Image(img_path, self.style_path, self.img_height, self.img_width, self.lr, frame_idx, prev_frame)
        image_model.build()
        result = image_model.train(self.n_iters)
        return result

    # extracts frames from video; stylizes each frame
    def vid_to_frames(self):
        video = cv2.VideoCapture(self.video_path)
        print("Video successfully opened.")
        count = 0 # Frame count
        is_reading = 1
        prev_frame = None

        while is_reading:
            is_reading, img = video.read()
            if (is_reading == False):
                break
            cv2.imwrite("./frames/frame_%d.png" % count, img)
            img_path = "./frames/frame_" + str(count) + ".png"
            result = self._stylize_frame(img_path, count, prev_frame)
            prev_frame = result
            count += 1
        print("All frames are successfully stylized.")

    # funcation that puts frames back into video with selected fps
    def frames_to_vid(self, path_in, path_out):
        frame_array = []
        files = [f for f in os.listdir(path_in) if isfile(join(path_in, f)) if not f.startswith('.')]

        # Sort the file names according to * in frame_*.png
        files.sort(key = lambda x: int(x[6:-4]))

        for i in range(len(files)):
            filename = path_in + files[i]
            # Read each file
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            # Insert the frames into an image array
            frame_array.append(img)
        print("Successfully read all files.")

        out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, size) # Alternatively *'XVID' with .avi

        for i in range(len(frame_array)):
            # Write to a image array
            out.write(frame_array[i])
        out.release()
        print("Stylized video saved.")