# import the necessary packages
import numpy as np
import cv2

FLANN_INDEX_LINEAR = 0
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_KMEANS = 2
FLANN_INDEX_COMPOSITE = 3
FLANN_INDEX_KDTREE_SINGLE = 4
FLANN_INDEX_HIERARCHICAL = 5
FLANN_INDEX_LSH = 6
FLANN_INDEX_SAVED = 254
FLANN_INDEX_AUTOTUNED = 255


class Matcher:
	def __init__(self, descriptor, ratio = 0.7, minMatches = 40, distanceMethod = "FlannBased", useHamming = True):
		# store the descriptor, book cover paths, ratio and minimum
		# number of matches for the homography calculation, then
		# initialize the distance metric to be used when computing
		# the distance between features
		self.descriptor = descriptor
		self.ratio = ratio
		self.minMatches = minMatches
		self.distanceMethod = distanceMethod
	
		# if the Hamming distance should be used, then update the
		# distance method
		if useHamming and self.distanceMethod == "BruteForce":
			self.distanceMethod += "-Hamming"
		
		self.matcher = cv2.DescriptorMatcher_create(self.distanceMethod)
		
		if self.distanceMethod == "FlannBased":
			#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

			index_params= dict(algorithm = FLANN_INDEX_LSH,
						   table_number = 6, # 12
						   key_size = 12,     # 20
						   multi_probe_level = 1) #2
			search_params = dict(checks = 50)
			self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

		
	def match(self, kpsA, featuresA, kpsB, featuresB):
		# compute the raw matches and initialize the list of actual
		# matches
		
		rawMatches = self.matcher.knnMatch(featuresB, featuresA, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other
			if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
		
		ptsA = np.float32([])
		ptsB = np.float32([])
		
		# check to see if there are enough matches to process
		if len(matches) > self.minMatches:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (i, _) in matches])
			ptsB = np.float32([kpsB[j] for (_, j) in matches])

		return ptsA, ptsB

		
	def calcHomography(self, ptsA, ptsB, inverseHomography = True):
		
		if inverseHomography:
			(M, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0) #A -> B
		else:
			(M, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0) # B -> A

		# return the ratio of the number of matched keypoints
		# to the total number of keypoints
		return M, float(status.sum()) / status.size
	
		
	def match_flann(self, desc1, desc2, r_threshold = 0.6):
		flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
		flann = cv2.flann_Index(desc2, flann_params)
		idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
		mask = dist[:,0] / dist[:,1] < r_threshold
		idx1 = np.arange(len(desc1))
		pairs = np.int32( zip(idx1, idx2[:,0]) )
		return pairs[mask]
		
	def matchIMG(self, image, kp1, des1, brisk, flann):

		height, width = image.shape
		kp2, des2 = brisk.detectAndCompute(image,None)
		matches = flann.knnMatch(des1,des2,k=2)
		
		sizeMatches = np.shape(matches)
		
		# # store all the good matches as per Lowe's ratio test.
		good = []
		if len(sizeMatches) > 1:
			for m,n in matches:
				if m.distance < 0.7*n.distance:
					good.append(m)
					
			if len(good)>MIN_MATCH_COUNT:
				src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
				dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
				
				M2, mask2 = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC,5.0)
				im_dst = cv2.warpPerspective(image, M2, (width,height))
				
				cv2.imshow("Corrected Perspective",im_dst)
				
