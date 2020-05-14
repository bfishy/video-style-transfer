# **video-style-transfer**
#### Performing neural artistic style transfer on video inputs

Neural style transfer on videos allows us to "re-draw" the original content frames of a video with the style from another style input, such as a painting. It is accomplished by independently extracting the content from the original input video, and the style representation from the style input, and combining both into one frame, while maintaining a certain level of style consistency between the frames.

Code in this repository is built and modeled on methods provided in [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf) by Gatys et al. and [Artistic Style Transfer for Videos](https://arxiv.org/pdf/1604.08610.pdf) by Ruder et al.

#### Image Example:
Content Image + Style:

<img src="./image content/castle.jpg" alt="castle image" width="200" height="150"> + <img src="./styles/starry_night.jpg" alt="style input" width="200" height="150">

Result:

<img src="./stylized images/castle_starry_300.png" alt="result" width="200" height="150">

#### Video Example:
Content Video + Style:

<img src="./video content/chair_gif.gif" alt="video input" width="200" height="150"> + <img src="./styles/starry_night.jpg" alt="style input" width="200" height="150">

Result:

<img src="./stylized video/stylized_vid_gif.gif" alt="stylized video" width="200" height="150">

## Setup 
- In order for the code to run correctly as designed, a loaded version of TensorFlow 1 is required (version 1.15 or 1.14 preferred).
- To install TensorFlow 1.15 on macOS, run `pip install --upgrade tensorflow==1.15` in the Terminal (it is advised to do so in a virtual environment)
- In cases where installing Tensorflow 1 is not feasible, it is advised to change the TensorFlow 1 syntax into TensorFlow 2 by importing functions from `tensorflow.compat.v1`
- In order for the program to successfully save the stylized frames, a folder named `output` needs to be created in the same directory before running the program. 

## Usage
To run the program, call `python main.py` with the following tags with their corresponding inputs:
- `-v` or `--video`: filepath of the input video
- `-s` or `--style_img`: filepath of the style image
- `-o` or `--output`: the filepath + name of the output file, ending with `.mp4`
- `--width`: the width of the output image, default `width==400`
- `--height`: the height of the output image, default `height==300`
- `--lr`: the learning rate of the model, default `lr==2`
- `--iter`: the number of training iternations per frame, default `iter==100`
- `--fps`: the frame rate (frames per second) of the output video, default `fps==15`
