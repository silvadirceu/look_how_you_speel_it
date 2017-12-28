# Copyright (C) 2015 Ross D Milligan
# GNU GENERAL PUBLIC LICENSE Version 3 (full notice can be found at https://github.com/rdmilligan/SaltwashAR)

import cv2
#from threading import Thread
from utils.threadstopable import ThreadStop

class Webcam:

	def __init__(self,cameraId=0):
		self.video_capture = cv2.VideoCapture(cameraId)
		self.camIsOpened = True
		
		if (self.video_capture.isOpened() is False ):
			self.camIsOpened = False
			print ( "Unable to connect to camera" )
		else:
			#self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
			#self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
			
			self.current_frame = self.video_capture.read()[1]

	# create thread for capturing images
	def start(self):
		self.thread = ThreadStop(target=self._update_frame, args=())
		self.thread.start()

	def _update_frame(self):
		while(True):
			ret, frame = self.video_capture.read()
			if ret:
				self.current_frame = frame
				#self.ret = ret
				 
	# get the current frame
	def get_current_frame(self):
		return self.current_frame
		
	def cameraIsOpened(self):
		return self.camIsOpened
	
	def stop_video(self):
		self.video_capture.release()
		self.thread.stop()
		self.thread.join()
		
