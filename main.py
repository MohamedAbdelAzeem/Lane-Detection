import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from IPython.core.display import HTML, Video
from moviepy.editor import VideoFileClip

from LaneLines import *



"""
mounting google drive to google colab 

from google.colab import drive
drive.mount('/content/drive')
%cd  MyDrive/A_ImageProcessingProject/

"""



lanelines = LaneLines()
def process_image(img):
    # step 1
    img1 = To_Birdeye(img)
#     img1 = np.copy(img)
    
    # step 2
    hls = cv2.cvtColor(img1, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    v_channel = hsv[:,:,2]

    right_lane = threshold_rel(l_channel, 0.8, 1.0)
    right_lane[:,:750] = 0

    left_lane = threshold_abs(h_channel, 20, 30)
    left_lane &= threshold_rel(v_channel, 0.7, 1.0)
    left_lane[:,550:] = 0

    img2 = left_lane | right_lane
    
#     img2 = birdeye.forward(img2)
    
    # step 3
    img3 = lanelines.forward(img2)
    

    # step 4
    img4 = From_Birdeye(img3)

    out_img = cv2.addWeighted(img, 1, img4, 1, 0)
    
    out_img = lanelines.plot(out_img)

    return out_img
  
  
  
  
def process_video(self, input_path, output_path):
      clip = VideoFileClip(input_path)
      out_clip = clip.fl_image(self.forward)
      out_clip.write_videofile(output_path, audio=False)  
  
  
def main():
  args = docopt(__doc__)
  input = args['INPUT_PATH']
  output = args['OUTPUT_PATH']

  findLaneLines = FindLaneLines()
  if args['--video']:
      findLaneLines.process_video(input, output)
  else:
      findLaneLines.process_image(input, output)


if __name__ == "__main__":
    main()
