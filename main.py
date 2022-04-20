import numpy as np
import matplotlib.image as mpimg
import cv2
import sys
from docopt import docopt
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *



"""
mounting google drive to google colab 
from google.colab import drive
drive.mount('/content/drive')
%cd  MyDrive/A_ImageProcessingProject/
"""

def apply_threshold(img):
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
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

  return img2

def To_Birdeye(img):
    """Init PerspectiveTransformation."""
    src = np.float32([(550, 460),     # top-left
                          (150, 720),     # bottom-left
                          (1200, 720),    # bottom-right
                          (770, 460)])    # top-right
    dst = np.float32([(100, 0),
                          (100, 720),
                          (1100, 720),
                          (1100, 0)])

    img_size=(1280, 720)
    flags=cv2.INTER_LINEAR
    
    M = cv2.getPerspectiveTransform(src, dst)

    result_Image = cv2.warpPerspective(img, M, img_size, flags)

    return result_Image


def From_Birdeye(img):
  """ Take a top view image and transform it to front view """
  src = np.float32([(550, 460),     # top-left
                        (150, 720),     # bottom-left
                        (1200, 720),    # bottom-right
                        (770, 460)])    # top-right
  dst = np.float32([(100, 0),
                        (100, 720),
                        (1100, 720),
                        (1100, 0)])

  img_size=(1280, 720)
  flags=cv2.INTER_LINEAR
  
  M_inv = cv2.getPerspectiveTransform(dst, src)

  result_Image = cv2.warpPerspective(img, M_inv, img_size, flags=flags)

  return result_Image

def process_image(img):
    # step 1
    img1 = To_Birdeye(img)

    
    # step 2
    img2 = apply_threshold(img1)
    imgTh = apply_threshold(img)
        
    # step 3
    img3 = lanelines.forward(img2)
    

    # step 4
    img4 = From_Birdeye(img3)

    out_img = cv2.addWeighted(img, 1, img4, 1, 0)
    

################## Draw images ###########################################


   
    #percent by which the image is resized
    scale_percent = 70

    #calculate the  dimensions
    width = 320
    height = 180
  
    # dsize
    dsize = (width, height)
    if(debug):
        OutImageWindows=  cv2.resize(img2, dsize)

        OutImagTH =  cv2.resize(imgTh, dsize)

        img3Small =  cv2.resize(img3, dsize)

        #resize images with windows
        lanelines.OutImageWindow =  cv2.resize(lanelines.OutImageWindow, dsize)

        out_img[180:360,960:1280,:]  = img3Small

        out_img[0:180,320:640,0]  = OutImagTH
        out_img[0:180,320:640,1]  =  OutImagTH
        out_img[0:180,320:640,2]  =  OutImagTH


        out_img[0:180,640:960,0]  =  OutImageWindows
        out_img[0:180,640:960,1]  =  OutImageWindows
        out_img[0:180,640:960,2]  =  OutImageWindows

        out_img[0:180,960:1280,:] =  lanelines.OutImageWindow

########################################################################

    #print curvature
    out_img = lanelines.plot(out_img)

    return out_img
  
  
  
  
def process_video(input_path, output_path,debug):
      clip = VideoFileClip(input_path)
      out_clip = clip.fl_image(process_image)
      out_clip.write_videofile(output_path, audio=False)  
      


# Main Code

lanelines = LaneLines()
input_path = sys.argv[1]
output_path = sys.argv[2]
debug = int(sys.argv[3])


process_video(input_path, output_path,debug)
 
