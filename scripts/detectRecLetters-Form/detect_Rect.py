# USAGE
# python detect_Rect.py -f "folder to the input images" -d "folder to the output images"
#						-l "folder to the output cutted letters" -e "jped"	

# import the necessary packages
from utils.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np
import os, shutil

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,help="path to the input image")
ap.add_argument("-d", "--dst", required=True,help="path to the output image")
ap.add_argument("-l", "--letters", required=True,help="path to the output letters")
ap.add_argument("-e", "--ext", required=True,help="files extension")
args = vars(ap.parse_args())

subfolder = args["folder"]
outputfiles = args["dst"]
outputletters = args["letters"]
ext = args["ext"]

imagePaths = []
imagePathsOut = []
for x in os.listdir(subfolder):
	xpath = os.path.join(subfolder, x)
	xoutpath = os.path.join(outputfiles, x)
	if x.endswith(ext):
		imagePaths.append(xpath)
		imagePathsOut.append(xoutpath)


Letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y","Z","1", "2", "3", "4", "5", "6", "7", "8", "9", "0" ]
print(len(imagePaths))
count = 58
delta = 15
delta2 = 21

for imagePath, imagePathOut in zip(imagePaths,imagePathsOut):
	print("processing: {}".format(imagePath))
	
	# read image and convert it to RGB
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	#if count == 15:
	image = cv2.imread(imagePath)

	sd = ShapeDetector()
	boxRects = sd.processRectangles(image)
	
	if len(boxRects) == 36:
		shutil.move(imagePath,imagePathOut)
		count += 1
		for (x1, y1, x2, y2), letter in zip(boxRects,Letters):
			crop = image[y1+delta2:y2-delta,x1+delta:x2-delta]
			
			arq = ''.join([letter ,"_",str(count),".jpg"])
			dirout = os.path.join(outputletters,arq)
			print(dirout)

			#approxFinal = sd.cutLetters(crop)
			#print(approxFinal)
			
			#crop = image[y1-5:y2+5,x1-5:x2+5]
			cv2.imwrite(dirout, crop)
			
			#cv2.imshow(dirout, crop)
			#cv2.waitKey(0)
	