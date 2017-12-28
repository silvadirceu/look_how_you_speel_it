import cv2
import numpy as np 
import math, sys

	
class Stabilization:
	def __init__(self, size  = (15, 15), maxlevel = 5, Criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03)):
		self.points=[]
		self.pointsPrev=[] 
		self.pointsDetectedCur=[] 
		self.pointsDetectedPrev=[]
		self.set_lk_params(size, maxlevel, Criteria)
		
	def set_lk_params(self,size, maxlevel, Criteria):
		self.lk_params = dict(winSize = size, maxLevel = maxlevel, criteria = Criteria)
		
	def setImagePrev(self,image):
		self.imPrev = image

	def getImagePrev(self):
		return self.imPrev
		
	def setPointsPrevArr(self,P):
		self.pointsPrev = P

	def setPointsDetectedPrevArr(self,P):
		self.pointsDetectedPrev = P
		
	def setPointsArr(self,P):
		self.points = P

	def getPoints(self):
		return self.points

	def getPointsNP(self):
		return np.asarray(self.points)
		
	def setPointsDetectedCurArr(self,P):
		self.pointsDetectedCur = P
	
	def getPointsDetectedCur(self):
		return self.pointsDetectedCur
		
	def resetPointsPrev(self):
		self.pointsPrev=[] 
		self.pointsDetectedPrev=[]

	def resetPointsCur(self):
		self.points=[] 
		self.pointsDetectedCur=[]

	def selectPoints(self,PArr):
		pass
	
	def checkedTrace(self,img0, img1, p0, back_threshold = 1.0):
		p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
		p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
		d = abs(p0-p0r).reshape(-1, 2).max(-1)
		status = d < back_threshold
		return p1, status
		
	def procesStabilization(self, imGray, isFirstFrame, PArr, sigma=100):
		
		self.resetPointsPrev()
		K = 0
		
		if (isFirstFrame==True):
			self.setPointsPrevArr(PArr)
			self.setPointsDetectedPrevArr(PArr)
			K = len(PArr)
		# If not the first frame, copy points from previous frame.
		else:
			K = min(len(PArr),len(self.points),len(self.pointsDetectedCur))
			self.pointsPrev = self.points[:K]
			self.pointsDetectedPrev = self.pointsDetectedCur[:K]

		# pointsDetectedCur stores results returned by the facial landmark detector
		# points stores the stabilized landmark points
		self.resetPointsCur()
		#self.setPointsArr(PArr)
		self.setPointsDetectedCurArr(PArr[:K])

		# Convert to numpy float array
		#pointsArr = np.array(self.points,np.float32)
		pointsPrevArr = np.array(self.pointsPrev,np.float32)

		#  Set up optical flow params
		#self.set_lk_params((s, s), 5, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))
		# Python Bug. Calculating pyramids and then calculating optical flow results in an error. So directly images are used.
		# ret, imGrayPyr= cv2.buildOpticalFlowPyramid(imGray, (winSize,winSize), maxLevel)

		#pointsArr,status, err = cv2.calcOpticalFlowPyrLK(self.imPrev,imGray,pointsPrevArr,None,**self.lk_params)
		p2,trace_status = self.checkedTrace(self.imPrev,imGray,pointsPrevArr)
		pointsArr = p2[trace_status].copy()
		self.pointsDetectedPrev = self.pointsDetectedPrev[trace_status].copy() 
		
		print(trace_status)
		# Converting to float and Converting back to list
		self.points = np.array(pointsArr,np.float32).tolist()

		# Final landmark points are a weighted average of
		# detected landmarks and tracked landmarks
		print(len(PArr),len(self.points),len(self.pointsDetectedPrev),len(self.pointsDetectedCur))
		
		for k in range(0,K):
			d = cv2.norm(np.array(self.pointsDetectedPrev[k]) - np.array(self.pointsDetectedCur[k]))
			alpha = math.exp(-d*d/sigma)
			self.points[k] = (1 - alpha) * np.array(self.pointsDetectedCur[k]) + alpha * np.array(self.points[k])

