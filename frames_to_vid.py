import cv2
import os
import numpy as np

from os.path import isfile, join

def frames_to_vid(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f)) if not f.startswith('.')]

    # sorting the file names according to * in frame_*.png
    files.sort(key = lambda x: int(x[6:-4]))
    n = len(files)

    print('Reading frames ...')
    for i in range(n):
        filename = pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        # print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    print('Successfully read %d frames.' % n)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size) # alternatively *'XVID' with .avi

    print('Processing video ...')
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    print('Output video saved. Video legnth: %f seconds.' % (len(frame_array) / fps))


def main():
    frames_to_vid('./output/', 'stylized_video.mp4', 15) # fps would prob be 30

if __name__ == "__main__":
    main()