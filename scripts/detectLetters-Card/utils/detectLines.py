import numpy as np
import cv2

def lineDist(lines, dist):
	if lines.shape[0] > 1:
		y = lines[:,0,1]
		lines = lines[np.argsort(y)]
		
		linesFinal = []
		linesFinal.append(lines[0])
		 
		diff = abs(lines[1:,0,1] - lines[:-1,0,1])

		for i in range(len(diff)):
			if diff[i] > dist:
				linesFinal.append(lines[i+1])
	else:
		linesFinal = lines

	return np.array(linesFinal)

def lineDetect(img,gray,minLineLength):
	edges = cv2.Canny(gray,50,150,apertureSize = 3)
	
	lines = cv2.HoughLinesP(image=edges,rho=0.2,theta=np.pi/500, threshold=10,lines=np.array([]), minLineLength=minLineLength,maxLineGap=20)
	lines = lineDist(lines,10)
	
	a,b,c = lines.shape
	for i in range(a):
		print(lines[i][0][0], lines[i][0][1])
		cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0,0,255), 2, cv2.LINE_AA)

	return img
	
if __name__ == '__main__':

	img = cv2.imread('C:/Users/Dirceu/Documents/Estudos/CursoOpenCV/Projects/Final/images/Cartoes/Olha-Como-se-Escreve-lamina.jpg')
	minLineLength=img.shape[1]-50
	
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	img2 = lineDetect(img,gray,minLineLength)
	
	cv2.imshow('result', img2)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()