#!/usr/bin/env python

import cv2

import numpy as np
import os, shutil
import mahotas

SZ = 20
CLASS_N = 26

# local modules
from common import clock, mosaic

LettersMap = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "J":9, "K":10, "L":11, "M":12, "N":13, "O":14, "P":15, "Q":16, "R":17, "S":18, "T":19, "U":20, "V":21, "W":22, "X":23, "Y":24,"Z":25,"1":26, "2":27, "3":28, "4":29, "5":30, "6":31, "7":32, "8":33, "9":34,"0":35}

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def load_digits(fn):
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (SZ, SZ))
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    return digits, labels

def setLocalMaxWindowSize(size):
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2 * size + 1, 2 * size + 1) );
	return kernel
		
def preProcess(gray,cell_size):
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray[:3,:]   = 255
	gray[:,:3]   = 255
	gray[:,-3:]  = 255
	gray[-3:,:]  = 255
	
	#cv2.imshow("ORIGINAL", gray)
	ret2,gray = cv2.threshold(gray,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#kernel = setLocalMaxWindowSize(1)
	
	# blur the image, find edges, and then find contours along
	# the edged regions
	blurred = cv2.GaussianBlur(gray, (7, 7), 0)
	
	thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV,45,10) #45
	
	# kernel = setLocalMaxWindowSize(2)
	# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	kernel = setLocalMaxWindowSize(2)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	
	edged = cv2.Canny(thresh,30,150,apertureSize = 5)
		
	#edged = cv2.Canny(blurred, 30, 150)
	(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.imshow("CANNY", edged)
	# # cv2.waitKey(0)
	# cv2.namedWindow("ROI", cv2.WINDOW_NORMAL);
	# cv2.namedWindow("ROISkewed", cv2.WINDOW_NORMAL);
	# cv2.namedWindow("roiCentered", cv2.WINDOW_NORMAL);
	
	# loop over the contours
	max_area = 0
	#print("I area {} cnts" .format(len(cnts)))
	for c in cnts:
		# compute the bounding box for the rectangle
		(x, y, w, h) = cv2.boundingRect(c)
		# if the width is at least 7 pixels and the height
		# is at least 20 pixels, the contour is likely a digit
		area = w*h
		
		if area > max_area:
			cnt = c
			max_area = area
		
		#if (w >= 7) and (h >= 20):
	
	(x, y, w, h) = cv2.boundingRect(cnt)
	roi = thresh[y:y + h, x:x + w]

	roi = deskew(roi,SZ)
	roi = center_extent(roi, (20, 20))
	
	# deskew the image center its extent
	#thresh = dataset.center_extent(thresh, (20, 20))
	# img = gray.copy()
	# cv2.drawContours(img, [cnt], -1, (0), 2)
	# cv2.rectangle(img, (x,y),(x+w,y+h), (0), 2)
	# cv2.imshow("ORIGINAL", img)
	
	# cv2.imshow("ROI", roi)
	
	# cv2.imshow("ROISkewed", roiSkewed)
	# cv2.imshow("roiCentered", roiCentered)
	# cv2.waitKey(0)
	
	return roi
	
def load_letters(fn,filename):
    letter_img = cv2.imread(os.path.join(fn, filename), 0)
    letter = preProcess(letter_img, (SZ, SZ))
    label = LettersMap[filename[0]]
    return letter, label
	
	
def load_folder(fn):
	images = []
	labels = []
	for filename in os.listdir(fn):
		if filename.endswith('jpg'):
			image,label = load_letters(fn,filename)
			images.append(image)
			labels.append(label)
	return np.array(images), np.array(labels)

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
	
def deskew(img,width):
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

def center_extent(image, size):
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
	
class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
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


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err)*100))

    confusion = np.zeros((CLASS_N, CLASS_N), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        
        vis.append(img)
    return mosaic(25, vis)

def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0


def get_hog() : 
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


def get_train_test_indx(y,train_proportion=0.9):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
    '''

    y=np.array(y)
    train_indx = np.zeros(len(y),dtype=bool)
    test_indx = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_indx[value_inds[:n]]=True
        test_indx[value_inds[n:]]=True

    return train_indx,test_indx
	
	
	
	

if __name__ == '__main__':

	print('Loading letters from letters Folder ... ')
	# Load data, Deskew and Center
	letters, labels = load_folder("../../images/letters")
	print(letters[0].shape)
	
	print('Shuffle data ... ')
	# Shuffle data
	rand = np.random.RandomState(10)
	shuffle = rand.permutation(len(letters))
	letters = letters[shuffle]
	labels = labels[shuffle]
	
	print('Defining HoG parameters ...')
	# HoG feature descriptor
	hog = get_hog();

	print('Calculating HoG descriptor for every image ... ')
	hog_descriptors = []
	for img in letters:
		hog_descriptors.append(hog.compute(img))
	hog_descriptors = np.squeeze(hog_descriptors)
	
	print(hog_descriptors.shape)
	
	print('Spliting data into training (90%) and test set (10%)... ')

	train_inds,test_inds = get_train_test_indx(labels,train_proportion=0.9)
	
	letters_train 			= letters[train_inds]
	letters_test 			= letters[test_inds]
	hog_descriptors_train 	= hog_descriptors[train_inds]
	hog_descriptors_test 	= hog_descriptors[test_inds]
	labels_train 			= labels[train_inds]
	labels_test 			= labels[test_inds]
	
	
	#train_n=int(0.9*len(hog_descriptors))
	#letters_train, letters_test = np.split(letters, [train_n])
	#hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
	#labels_train, labels_test = np.split(labels, [train_n])


	print('Training SVM model ...')
	model = SVM()
	model.train(hog_descriptors_train, labels_train)

	print('Saving SVM model ...')
	model.save('letters_svm2.dat')


	print('Evaluating model ... ')
	vis = evaluate_model(model, letters_test, hog_descriptors_test, labels_test)
	cv2.imwrite("letters-classification2.jpg",vis)
	cv2.imshow("Vis", vis)
	cv2.waitKey(0)



