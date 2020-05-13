import cv2

def read_frames(filepath):
    video = cv2.VideoCapture(filepath)
    print("Video successfully opened.")
    count = 0  # frame count starts from one
    is_reading = 1

    while is_reading:
        is_reading, img = video.read()
        if (is_reading == False):
            break
        cv2.imwrite("frames/frame_%d.png" % count, img)
        count += 1
    print("%d frames successfully generated." % count)

# def main():
#     read_frames('test.mp4')

# if __name__ == "__main__":
#     main()