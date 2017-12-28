# import the necessary packages
import numpy as np
import cv2
import mahotas

LettersMap = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":5, "H":7, "I":8, "J":9, "K":10, "L":11, "M":12, "N":13, "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19, "U":20, "V":21, "W":22, "X":23, "Y":24,"Z":25,"1":26, "2":27, "3":28, "4":29, "5":30, "6":31, "7":32, "8":33, "9":34,"0":35}
SZ = 20

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

def setLocalMaxWindowSize(size):
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * size + 1, 2 * size + 1) );
	return kernel
	
class DetectLetters:
	def __init__(self, modelFile):
		self.model = SVM()
		self.model.load(modelFile)
		self.hog = self.get_hog()
		self.inProcess = False
		self.kernel = setLocalMaxWindowSize(2)
		
	def detectLetters(self,gray,P1,P2):
		
		grayCutted = gray[P1[1]:P2[1],P1[0]:P2[0]]
		
		#ret2,grayBin = cv2.threshold(grayCutted,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #*****
		
		blurred = cv2.GaussianBlur(grayCutted, (5, 5), 0)
		thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV,15,2) #45
		
		#thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel) #*****
		
		edged = cv2.Canny(thresh,30,150,apertureSize = 5)
			
		(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		# sort the contours by their x-axis position, ensuring
		# that we read the numbers from left to right
		cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key = lambda x: x[1])
		
		imageO = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
		
		# cv2.imshow("Thresh", thresh)
		# cv2.imshow("Edge", edged)
		
		# cv2.namedWindow("ROI", cv2.WINDOW_NORMAL);
		
		# loop over the contours
		L = []
		for (c, _) in cnts:
			# compute the bounding box for the rectangle
			(x, y, w, h) = cv2.boundingRect(c)
			# if the width is at least 7 pixels and the height
			# is at least 20 pixels, the contour is likely a digit
			
			if (h >= 13 and ((0.4*h < w < 1.6*h) or (0.4*w < h < 1.6*w))):
				cv2.rectangle(imageO, (x+P1[0],y+P1[1]),(x+P1[0]+w,y+P1[1]+h), (0,0,255), 1)
			
				roi = thresh[y:y + h, x:x + w]
				
				roi = self.deskew(roi,SZ)
				roi = self.center_extent(roi, (SZ, SZ))
				let = self.classify(roi)
				#print(let)
				L.append(let)
				
				#cv2.imshow("ROI", roi)
				#cv2.waitKey(0)
				
				
		return ''.join(L),imageO

		
	def correctPerspective(self,image, M):
		size = image.shape
		im_dst = cv2.warpPerspective(image, M, (size[1],size[0]))
		return im_dst

	
	def detect(self, imageGray, M):
		_,M_inv = cv2.invert(M)
		im_dst = self.correctPerspective(imageGray, M_inv)
		im_Color = cv2.cvtColor(im_dst, cv2.COLOR_GRAY2BGR)
		
		#self.inProcess = True
		
		#im_O = lineDetect(im_dst)
		x1 = 20
		y1 = int(im_dst.shape[0]*0.15)
		x2 = im_dst.shape[1]-20
		y2 = y1 + 55
		cv2.rectangle(im_Color, (x1,y1),(x2,y2), (0, 0, 255), 1)
		
		L,im_Color=self.detectLetters(im_dst,(x1,y1),(x2,y2))
	
		return L,im_Color

	def deskew(self,img,width):
		(h, w) = img.shape[:2]
		m = cv2.moments(img)
		if abs(m['mu02']) < 1e-2:
			return img.copy()
		skew = m['mu11']/m['mu02']
		M = np.float32([[1, skew, -0.5*w*skew], [0, 1, 0]])
		img = cv2.warpAffine(img, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
		
		# resize the image to have a constant width
		image = resize(img, width = width)
		return image

	def center_extent(self,image, size):
		# grab the extent width and height
		(eW, eH) = size

		# handle when the width is greater than the height
		if image.shape[1] > image.shape[0]:
			image = resize(image, width = eW)

		# otherwise, the height is greater than the width
		else:
			image = resize(image, height = eH)

		# allocate memory for the extent of the image and
		# grab it
		extent = np.zeros((eH, eW), dtype = "uint8")
		offsetX = (eW - image.shape[1]) // 2
		offsetY = (eH - image.shape[0]) // 2
		extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image

		# compute the center of mass of the image and then
		# move the center of mass to the center of the image
		(cY, cX) = np.round(mahotas.center_of_mass(extent)).astype("int32")
		(dX, dY) = ((size[0] // 2) - cX, (size[1] // 2) - cY)
		M = np.float32([[1, 0, dX], [0, 1, dY]])
		extent = cv2.warpAffine(extent, M, size)

		# return the extent of the image
		return extent
		


	def classify(self, sample):
		hog_descriptor = self.hog.compute(sample).T
		respNr = self.model.predict(hog_descriptor)
		respLett = list(LettersMap.keys())[list(LettersMap.values()).index(respNr)]
		
		return respLett

	def get_hog(self) : 
		winSize = (20,20)
		blockSize = (8,8)
		blockStride = (4,4)
		cellSize = (8,8)
		nbins = 9
		derivAperture = 1
		winSigma = -1.
		histogramNormType = 0
		L2HysThreshold = 0.2
		gammaCorrection = 1
		nlevels = 64
		signedGradient = True

		hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

		return hog

	def isInProcess(self):
		return self.inProcess
	
	def setInProcess(self, value):
		self.inProcess = value
		
class SVM():
	def __init__(self, C = 12.5, gamma = 0.50625):
		self.model = cv2.ml.SVM_create()
		self.model.setGamma(gamma)
		self.model.setC(C)
		self.model.setKernel(cv2.ml.SVM_RBF)
		self.model.setType(cv2.ml.SVM_C_SVC)
				
	def train(self, samples, responses):
		self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

	def predict(self, samples):
		return self.model.predict(samples)[1].ravel()
		
	def load(self,modelFile):
		self.model = cv2.ml.SVM_load(modelFile)
		
	def save(self,modelFile):
		self.model.save(modelFile)