import argparse
import os
import tensorflow as tf
from video import Video

def main():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True, help='path to the video')
    parser.add_argument('-s', '--style_img', required=True, help='path to the style image')
    parser.add_argument('-o', '--output', required=True, help='output path')
    parser.add_argument('--width', required=False, default=400, type=int, help='output image width')
    parser.add_argument('--height', required=False, default=300, type=int, help='output image height')
    parser.add_argument('--lr', required=False, default=2., type=float, help='hyperparameter: learning rate')
    parser.add_argument('--iter', required=False, default=100, type=int, help='hyperparameter: number of training iterations')
    parser.add_argument('--fps', required=False, default=15, type=int, help='frames per second')

    args = parser.parse_args()

    # Parse arguments
    video_path = args.video
    style_path = args.style_img
    output_path = args.output
    img_height = args.height
    img_width = args.width
    lr = args.lr
    n_iters = args.iter
    fps = args.fps

    video_processor = Video(video_path, style_path, img_height, img_width, lr, n_iters, fps)
    video_processor.vid_to_frames()
    video_processor.frames_to_vid('./output/', output_path)

if __name__ == "__main__":
    main()