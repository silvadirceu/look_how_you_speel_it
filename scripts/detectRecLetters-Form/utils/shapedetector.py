# import the necessary packages
import cv2
import numpy as np
from utils.nms import non_max_suppression_fast
import imutils

class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.1 * peri, True)
		rect = []
		# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"

		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)
			areaRect = float(w)*h
			
			rect = [x,y,x+w,y+h]
			
			area = cv2.contourArea(c)
			
			delta = 0.1
			
			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			#if -delta < ang1 < delta and -delta < ang2 < delta and -delta < ang3 < delta:
			if area > 0: 
				if abs(area-areaRect)/area < delta:
					shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
				else:
					shape = "quadrilateralNR"

		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"

		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"

		# return the name of the shape
		return (shape,rect)
		
		
	def detectRectangles(self, contourns, areaFactor, nameImage, image):
	# find the main island (biggest area)
		cnt = contourns[0]
		max_area = cv2.contourArea(cnt)

		for cont in contourns:
			area = cv2.contourArea(cont)
			peri = cv2.arcLength(cont, True)
			approx = cv2.approxPolyDP(cont, 0.1 * peri, True)
			
			if area > max_area and len(approx)==4:
				cnt = cont
				max_area = area
		
		boundingBoxes = []
		rectanglesCnts = []
		
		# loop over the contours
		for c in contourns:
			# compute the center of the contour, then detect the name of the
			# shape using only the contour
			#if cv2.contourArea(c) > 900:
			
			if cv2.contourArea(c) > areaFactor*max_area:
				shape, rectBox = self.detect(c)
				
				# cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
				# cv2.drawContours(image, approx, -1, (0, 255, 0), 3)
				
				# cv2.imshow("Contourn", image)
				# cv2.waitKey(0)
				# peri = cv2.arcLength(c, True)
				# approx = cv2.approxPolyDP(c, 0.1 * peri, True)
				# print("I area {} maxArea {} poly {}".format(cv2.contourArea(c),areaFactor*max_area, len(approx)))			
				
				if shape in ["rectangle","square"]:
					rectanglesCnts.append(c)
					boundingBoxes.append(rectBox)
				
		
		(boxesPicked, pickIdx) = non_max_suppression_fast(np.array(boundingBoxes), 0.3)
		#boxesPicked = np.array(boundingBoxes)
		boxesPickedSorted = self.sortBoxes(boxesPicked)
		
		# correction of Y coordinate
		y = boxesPickedSorted[:,1]
		
		limits = []
		count = 0
		for k in range(1,len(y)):
			if y[k-1]-5 <= y[k] <= y[k-1]+5:
				count += 1
				
			else:
				count += 1
				limits.append(count)

		count += 1
		limits.append(count)
		
		inic = 0
		for k in range(len(limits)):
			fim = limits[k]
			y[inic:fim] = np.median(y[inic:fim])
			inic = limits[k]
			
		boxesPickedSorted[:,1] = y
		
		# sort with new y coordinate
		boxesPickedSorted = self.sortBoxes(boxesPickedSorted)

		#np.array(rectanglesCnts)[pickIdx],
		return  boxesPickedSorted
		
	def sortBoxes(self,boxes):
		
		N = boxes[:,0].max()
		
		val = np.zeros(len(boxes))
		count = 0;
		for (x1, y1, x2, y2) in boxes:
			val[count] = y1*N + x1
			count += 1
		
		Bsorted = boxes[val.argsort()]
		return Bsorted
	
	def processRectangles(self, image):
		
		resized = imutils.resize(image, width=600)
		ratio = image.shape[0] / float(resized.shape[0])
		
		# convert the resized image to grayscale, blur it slightly,
		# and threshold it
		gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (7, 5), 0)
				
		#thresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY_INV)[1]
		thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV,45,10) #45
		#thresh = cv2.bitwise_not(thresh)
		edges = cv2.Canny(thresh,50,150,apertureSize = 5)

		#kernel = self.setLocalMaxWindowSize(1)
		#edges = cv2.dilate(edges, kernel)
		
		cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL);
		# cv2.imshow("Thresh", edges)
		# cv2.waitKey(0)

		# find contours in the thresholded image and initialize the
		# shape detector

		# check to see if we are using OpenCV 2.X
		if imutils.is_cv2():
			(cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		# check to see if we are using OpenCV 3
		elif imutils.is_cv3():
			(_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

		print("I count {} contours in this image".format(len(cnts)))

		# cv2.imshow("Thresh", edges)
		# cv2.waitKey(0)

		# rectanglesIMG = resized.copy()
		# cv2.drawContours(rectanglesIMG, cnts, -1, (0, 0, 255), 2)
		
		# cv2.imshow("Thresh", rectanglesIMG)
		# cv2.waitKey(0)

		sd = ShapeDetector()
		areaFactor = 0.7
		boxesPicked = sd.detectRectangles(cnts, areaFactor, "Thresh", resized.copy())

		# img = image.copy()
		# cv2.namedWindow("Image", cv2.WINDOW_NORMAL);

		print("I count {} rectangles in this image".format(len(boxesPicked)))

		boxesPicked = boxesPicked.astype("float")
		boxesPicked *= ratio
		boxesPicked = boxesPicked.astype("int")

		# for (x1, y1, x2, y2) in boxesPicked:
			# # multiply the contour (x,  y)-coordinates by the resize ratio,
			# # then draw the contours and the name of the shape on the image
				
			# cv2.rectangle(img, (x1+delta,y1+delta2),(x2-delta,y2-delta), (0, 255, 0), 2)
			# # show the output image
			# cv2.imshow("Image", img)
		# cv2.waitKey(0)	
		
		return boxesPicked
		
	def setLocalMaxWindowSize(self,size):
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * size + 1, 2 * size + 1) );
		return kernel
	
	def cutLetters(self,image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (7, 5), 0)
				
		thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV,45,10) #45
		edges = cv2.Canny(thresh,50,150,apertureSize = 5)
		
		if imutils.is_cv2():
			(cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		# check to see if we are using OpenCV 3
		elif imutils.is_cv3():
			(_, cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
			
		image2 = image.copy()
		cv2.namedWindow("Letters", cv2.WINDOW_NORMAL);
		cv2.imshow("Letters", edges)
		cv2.waitKey(0)
		
		delta = 8
		Deltas = [[[delta,delta]],[[delta,-delta]],[[-delta,-delta]],[[-delta,+delta]]] 
		for c in cnts:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.1 * peri, True)
			if len(approx) == 4:
				idx = self.sortPoints(approx[:,0])
				approx = approx[idx]
				
				approxFinal = approx+Deltas
								
				cv2.drawContours(image2, [c], -1, (0, 0, 255), 2)
				cv2.drawContours(image2, approxFinal, -1, (0, 255, 0), 3)
				cv2.imshow("Letters", image2)
				cv2.waitKey(0)
		return approxFinal
		
	def sortPoints(self,points):
		
		soma = [0,0]
		for p in points:
			soma += p
		soma = soma/4
			
		cx = soma[1]
		cy = soma[0]
		
		idxs1 = np.where(points[:,0] < cy)[0]
		idxs2 = np.where(points[:,0] > cy)[0]
		
		idxSorted = []
		if (points[idxs1[0],1] < points[idxs1[1],1]):
			idxSorted.append(idxs1[0])  
			idxSorted.append(idxs1[1])  
		else: 
			idxSorted.append(idxs1[1])
			idxSorted.append(idxs1[0])  
			
		if (points[idxs2[0],1] < points[idxs2[1],1]):
			idxSorted.append(idxs2[1])  
			idxSorted.append(idxs2[0])  
		else: 
			idxSorted.append(idxs2[0])  
			idxSorted.append(idxs2[1])  
		
		return idxSorted
		
	