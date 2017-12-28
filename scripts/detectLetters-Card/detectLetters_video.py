import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils.descriptor import Descriptor
from utils.matcher import Matcher
from utils.detectletters import DetectLetters
from utils.webcam import Webcam
from utils.stabilization import Stabilization
import math, sys
import difflib

MIN_MATCH_COUNT = 10
SKIP_FRAMES = 2
WIDTH = 480

Words = ["CACHORRO","CAVALO","ELEFANTE","GATO","TIGRE","VACA"]

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized


	
def lineDist(lines, dist):
	if np.size(lines) > 4:
		y = lines[:,0,1]
		lines = lines[np.argsort(y)]
		
		linesFinal = []
		linesFinal.append(lines[0])
		 
		diff = abs(lines[1:,0,1] - lines[:-1,0,1])

		for i in range(len(diff)):
			if diff[i] > dist:
				linesFinal.append(lines[i+1])
	else:
		return lines, -1

	return np.array(linesFinal), 0

	# lines = cv2.HoughLinesP(edges,1,np.pi/180,275, minLineLength = 600, maxLineGap = 100)[0].tolist()
	# for x1,y1,x2,y2 in lines:
		# for index, (x3,y3,x4,y4) in enumerate(lines):

			# if y1==y2 and y3==y4: # Horizontal Lines
				# diff = abs(y1-y3)
			# elif x1==x2 and x3==x4: # Vertical Lines
				# diff = abs(x1-x3)
			# else:
				# diff = 0

			# if diff < 10 and diff is not 0:
				# del lines[index]

	# gridsize = (len(lines) - 2) / 2
	
	
def lineDetect(gray):

	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV,15,2) #45
	#ret3,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#thresh = cv2.bitwise_not(thresh)
	
	edges = cv2.Canny(thresh,30,150,apertureSize = 5)
	minLineLength= 30#gray.shape[1]-50
	
	lines = cv2.HoughLinesP(image=edges,rho=0.1,theta=np.pi/180, threshold=5,lines=np.array([]), minLineLength=minLineLength,maxLineGap=10)
	lines, res = lineDist(lines,10)
	
	imageO = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	
	if res != -1:
		a,b,c = lines.shape
		for i in range(a):
			print(lines[i][0][0], lines[i][0][1])
			cv2.line(imageO, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0,0,255), 2, cv2.LINE_AA)

	return imageO


	
	
if __name__ == '__main__':

	# trainImage
	img1 = cv2.imread('images/Olha-Como-se-Escreve-lamina.jpg',0)
	img1 = resize(img1, width = WIDTH)
	cv2.imshow("Original Card",img1)
		
	descriptor = Descriptor(useSIFT = False)
	(refKps, refDescs) = descriptor.describe(img1)
	
	matcher = Matcher(descriptor, ratio = 0.7, minMatches = MIN_MATCH_COUNT, distanceMethod = "FlannBased", useHamming = 1)
	
	letters = DetectLetters('letters_svm2.dat')
	
	#video = cv2.VideoCapture(1)
	# initialise webcam
	video = Webcam(1)
	if (video.cameraIsOpened() is False):
		print ( "Unable to connect to camera" )
		sys.exit()
		
	video.start()
	
	OFrame = video.get_current_frame()
	imGrayPrev = cv2.cvtColor(OFrame, cv2.COLOR_BGR2GRAY)
	imGrayPrev = resize(imGrayPrev, width = WIDTH)
	
	#stabilization = Stabilization()
	
	#stabilization.setImagePrev(imGrayPrev)
		
	fps = 30.0
	
	cv2.namedWindow("Main Frame", cv2.WINDOW_AUTOSIZE)
	cv2.namedWindow("Corrected Perspective", cv2.WINDOW_AUTOSIZE)
	
	t = cv2.getTickCount()
	count = 0
	isFirstFrame = True
	
	while True:
		if count==0:
			t = cv2.getTickCount()
			
		#ret, OriginalFrame = video.read()
		OriginalFrame = video.get_current_frame()
		image = cv2.cvtColor(OriginalFrame, cv2.COLOR_BGR2GRAY)
		img2 = resize(image, width = WIDTH)	
		
		#if ret:
		
		if (count % SKIP_FRAMES == 0):

			(Kps, Descs) = descriptor.describe(img2)
		
			if np.size(Kps) > 4:

				# find the keypoints and descriptors with BRISK
				ptsA, ptsB = matcher.match(Kps, Descs, refKps, refDescs)
				
				if np.array(ptsA).size > 5 and np.array(ptsB).size  > 5:
					#print("Antes Stab")
					#print(ptsA.shape,ptsB.shape)
					
					#stabilization.procesStabilization(img2, isFirstFrame, ptsB.tolist())
					#ptsB = stabilization.getPointsNP()
					
					#if np.array(ptsB).size:
					#	print("Antes Homography")
						
					#	K = min(ptsA.shape[0],ptsB.shape[0])
						
					M, score =  matcher.calcHomography(ptsA, ptsB, inverseHomography = False)
					
					if np.size(M) == 9:
						if letters.isInProcess() == False:
							L,im_O = letters.detect(img2.copy(),M)
							cv2.imshow("Corrected Perspective",im_O)
							W = difflib.get_close_matches(L, Words,1)
								
							if W and (W[0] in Words):
								letters.setInProcess(True)
								print(L,W[0])
						
						# size = img2.shape
						# im_dst = cv2.warpPerspective(img2, M, (size[1],size[0]))
						# im_O2 = cv2.cvtColor(im_dst, cv2.COLOR_GRAY2BGR)
						# cv2.imshow("Homography",im_O2)
				else:
					letters.setInProcess(False)
		
		isFirstFrame = False
		#stabilization.setImagePrev(img2)
		
		cv2.imshow("Main Frame",OriginalFrame)
		
		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break
		# increment frame counter
		count = count + 1
		# calculate fps at an interval of 100 frames
		if (count == 100):
			t = (cv2.getTickCount() - t)/cv2.getTickFrequency()
			fps = 100.0/t
			count = 0
		
		
	video.stop_video()
	cv2.destroyAllWindows()
