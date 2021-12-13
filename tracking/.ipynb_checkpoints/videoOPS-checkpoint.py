import numpy as np
import cv2
import time


def to_gray(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_binary(image, threshold=100):
    foreground = np.where(image >= threshold)
    x = foreground[0]
    y = foreground[1]
    im = np.ones(image.shape)
    im = im * 255
    im[x,y] = 0
    return im

def diff_imgs(img0,img1):
    img = img1 - img0
    img[np.where(img < 0)] = 0
    return img
    


class ImgViewer(object):
	"""docstring for ImgViewer"""
	def __init__(self,fps=30):
		super(ImgViewer, self).__init__()
		self.fps = fps
		self.timestep = 1.0/self.fps
		self.windowname = 'ImgViewerFrame'
		cv2.namedWindow(self.windowname)
		cv2.moveWindow(self.windowname,2600,40)

		self.time_started = None



	def __call__(self,img):
		cv2.imshow(self.windowname,img)
		time.sleep(self.timestep)


		


