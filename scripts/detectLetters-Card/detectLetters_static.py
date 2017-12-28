import numpy as np
import cv2
from matplotlib import pyplot as plt

FLANN_INDEX_LSH = 6
MIN_MATCH_COUNT = 10

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



if __name__ == '__main__':

	# Initiate BRISK detector
	brisk = cv2.BRISK_create()

	# trainImage
	img1 = cv2.imread('C:/Users/Dirceu/Documents/Estudos/CursoOpenCV/Projects/Final/images/Cartoes/Olha-Como-se-Escreve-lamina.jpg',0)
	img1 = resize(img1, width = 640)
	
	kp1, des1 = brisk.detectAndCompute(img1,None)
	FLANN_INDEX_KDTREE = 0
	#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	index_params= dict(algorithm = FLANN_INDEX_LSH,
					   table_number = 6, # 12
					   key_size = 12,     # 20
					   multi_probe_level = 1) #2
	search_params = dict(checks = 50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)	
	 # queryImage
	img2 = cv2.imread('C:/Users/Dirceu/Documents/Estudos/CursoOpenCV/Projects/Final/images/Cartoes/vaca.jpg',0)
	img2 = resize(img2, width = 640)

	height, width = img2.shape
	
	# find the keypoints and descriptors with SIFT

	kp2, des2 = brisk.detectAndCompute(img2,None)

	matches = flann.knnMatch(des1,des2,k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)
			
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		#M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		#matchesMask = mask.ravel().tolist()

		#h,w = img1.shape
		#pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		#dst = cv2.perspectiveTransform(pts,M)
		
		#M2 = cv2.getPerspectiveTransform(src_pts, dst_pts)
		M2, mask2 = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC,5.0)
		im_dst = cv2.warpPerspective(img2, M2, (width,height))
		
		#img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		cv2.imshow("Corrected",im_dst)
		cv2.waitKey(0)

	else:
		print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
		#matchesMask = None
		
	# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
					   # singlePointColor = None,
					   # matchesMask = matchesMask, # draw only inliers
					   # flags = 2)

	#img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

	#plt.imshow(img3, 'gray'),plt.show()


