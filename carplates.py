import cv2
import numpy as np
import  matplotlib.pyplot as plt
import glob

def mouseHandler(event,x,y,flags,param):
	global im_temp, pts_src

	if event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(im_temp,(x,y),3,(0,255,255),5,cv2.LINE_AA)
		cv2.imshow("Image", im_temp)
		if len(pts_src) < 4:
			pts_src = np.append(pts_src,[(x,y)],axis=0)

boool = False
sift = cv2.SIFT_create()

 # Read in the image.
im_src = cv2.imread('./archive/Cars0.png')
 
# Destination image
height, width = 400, 600
im_dst = np.zeros((height,width,3),dtype=np.uint8)
 
 
 # Create a list of points.
pts_dst = np.empty((0,2))
pts_dst = np.append(pts_dst, [(0,0)], axis=0)
pts_dst = np.append(pts_dst, [(width-1,0)], axis=0)
pts_dst = np.append(pts_dst, [(width-1,height-1)], axis=0)
pts_dst = np.append(pts_dst, [(0,height-1)], axis=0)
 
 # Create a window
cv2.namedWindow("Image", 1)
 
im_temp = im_src
pts_src = np.empty((0,2))
 
cv2.setMouseCallback("Image",mouseHandler)
 
 
cv2.imshow("Image", im_temp)
cv2.waitKey(0)
 
tform, status = cv2.findHomography(pts_src, pts_dst)
im_dst = cv2.warpPerspective(im_src, tform,(width,height))
 
cv2.imshow("Image", im_dst)
cv2.imwrite("out.png", im_dst)
cv2.waitKey(0)
 
matricules = glob.glob ('./queries/*.png')
for matricule1 in matricules:
	matricule = cv2.imread(matricule1)
	matricule_gray = cv2.cvtColor(matricule,cv2.COLOR_BGR2GRAY)
	matricule_keypoints, matricule_descriptor = sift.detectAndCompute(matricule, None)
	matricule_features = cv2.drawKeypoints(matricule,matricule_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
   
	   # Convert the training image to RGB
	matricule = cv2.cvtColor(matricule, cv2.COLOR_BGR2RGB)
	   
	   # Convert the training image to gray scale
	matricule_gray = cv2.cvtColor(matricule, cv2.COLOR_RGB2GRAY)
	   
	   
	   #fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,20))
	   #ax1.axis('off')
	   #ax2.axis('off')
	   #ax1.imshow(matricule,cmap='gray')
	   #ax2.imshow(gray1,cmap='gray')
	gray_keypoints, gray_descriptor = sift.detectAndCompute(im_dst, None)
	gray_features = cv2.drawKeypoints(im_dst,gray_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    
	   #plt.figure(figsize=(10,10))
	   #plt.axis('off')
	   #plt.imshow(gray_features)
	   #plt.show()
	   
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(matricule_descriptor,gray_descriptor, k=2)
	good_features = []
	for m,n in matches:
		if m.distance < 0.70*n.distance:
			good_features.append([m])
	if len(good_features)>20 :       
		
		imd = cv2.imread("granted.jpg")
		cv2.imshow("denied", imd)
		cv2.waitKey(0)
		plt.figure(figsize=(20,10))
		plt.axis('off')
	   
				 
if boool == False :
		imd = cv2.imread("denied.jpg")
		cv2.imshow("denied", imd)
		cv2.waitKey(0)	 
else:        
	imd = cv2.imread("granted.png")		
	cv2.imshow("granted", imd)	
	cv2.waitKey(0)		
	plt.figure(figsize=(20,10))
	plt.axis('off')
	plt.title('KNN Matching Points')
	comparaison = cv2.drawMatchesKnn(matricule, matricule_keypoints, im_dst, gray_keypoints, good_features,None, flags = 2)
	plt.imshow(comparaison)
	plt.show()
    