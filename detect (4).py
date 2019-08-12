import numpy as np
import cv2
import cv2.aruco as aruco
import math
from numpy import array
import matplotlib.pyplot as plt 
def getCameraMatrix():
	with np.load('System.npz') as X:
		camera_matrix, dist_coeff, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
	return camera_matrix, dist_coeff

def sin(angle):
	return math.sin(math.radians(angle))

def cos(angle):
	return math.cos(math.radians(angle))

def detect_markers(img, camera_matrix, dist_coeff):
    markerLength = 100
    aruco_list = []
    aruco_centre=[]
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    parameters = aruco.DetectorParameters_create()

    #lists of ids and the corners beloning to each id
    corners, aruco_id, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
   # np.array(aruco_id).tolist()
    
    
    
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    if np.all(aruco_id != None):
        rvec,tvec = aruco.estimatePoseSingleMarkers(corners,markerLength,camera_matrix, dist_coeff)
       
        aruco.drawDetectedMarkers(img, corners)
        for i in range(len(corners)):
            
            x = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
            y = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4
            cv2.putText(img, "Id: " + str(aruco_id[i]), (int(x),int(y)), font, 1, (255,0,0),2,cv2.LINE_AA)
            aruco_centre.extend([(int(x),int(y))])
           
        aruco_list=np.array(list(zip(aruco_id.astype(int),aruco_centre,rvec,tvec)))
        
        
    
        return aruco_list
            
            
      
    
    
	######################## INSERT CODE HERE ########################

	
	##################################################################
	#return aruco_list

def drawAxis(img, aruco_list, aruco_id, camera_matrix, dist_coeff):
   
     for x in aruco_list:
         if aruco_id == x[0]:
			rvec, tvec = x[2], x[3]
     markerLength=100
     m = markerLength/2
     pts = np.float32([[-m,m,0],[m,m,0],[-m,-m,0],[-m,m,m]])
     pt_dict = {}
     print(pts)
     imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
     print(imgpts)
     for i in range(len(pts)):
    		 pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel())    
     src = pt_dict[tuple(pts[0])];   dst1 = pt_dict[tuple(pts[1])];
     dst2 = pt_dict[tuple(pts[2])];  dst3 = pt_dict[tuple(pts[3])];
     
     img = cv2.line(img, src, dst1, (0,255,0), 1)
     img = cv2.line(img, src, dst2, (255,0,0), 1)
     img = cv2.line(img, src, dst3, (0,0,255), 1)
     return img

def drawCube(img, ar_list, ar_id, camera_matrix, dist_coeff):
    for x in ar_list:
        if ar_id == x[0]:
            rvec, tvec = x[2], x[3]
    markerLength = 100
    m = markerLength/2
    pts = np.float32([[-m,m,0],[-m,-m, 0], [m,- m, 0], [m, m, 0],[-m,m,m],[-m,-m,m],[m,-m,m],[m,m,m]])
    imgpts, _ = cv2.projectPoints(pts, rvec, tvec,camera_matrix,dist_coeff)
    imgpts=np.int32(imgpts).reshape(-1,2)
    img=cv2.drawContours(img,[imgpts[:4]],-1,(255,122,0),2)
    for i,j in zip(range(4),range(4,8)):
        img=cv2.line(img,tuple(imgpts[i]),tuple(imgpts[j]),(255,122,0),2)
    img=cv2.drawContours(img,[imgpts[4:]],-1,(255,122,0),2)
    cv2.imshow("img",img)
    return img
	######################## INSERT CODE HERE ########################
    
	
	##################################################################

def drawCylinder(img, ar_list, ar_id, camera_matrix, dist_coeff):
     for x in ar_list:
		if ar_id == x[0]:
			 rvec, tvec ,ar_center = x[2], x[3] ,x[1]
     markerLength = 100
     m=markerLength/2
     radius = markerLength/2;
     pts=[]
     
     h = markerLength*1.5
     """
     for i in range(0,360,60):
         pt=[m*cos(i),m*sin(i),0]
         pt=np.round(pt,0)
         pts.append(pt)
     pts=array(pts)
     pt_dict={}
     print(pts)
     imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
     print(imgpts)
     for i in range(len(imgpts)):
         imgpts[i]=np.round(imgpts[i],2)
     for i in range(len(pts)):
         pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel()) 
     src = pt_dict[tuple(pts[0])]
     print(src)
     d1 = pt_dict[tuple(pts[1])]
     print(d1)
     im = plt.imread(img)
     implot = plt.imshow(im)
     plt.scatter(src)
     plt.show()
     """
     
     
     pts = np.float32([[0,0,0],[0,m,0],[m,0,0],[-m,0,0],[0,-m,0],[m*cos(30),m*sin(30),0],[m*cos(60),m*sin(60),0],[m*cos(30),-m*sin(30),0],[m*cos(60),-m*sin(60),0],[-m*cos(30),-m*sin(30),0],[-m*cos(60),-m*sin(60),0],[-m*cos(30),m*sin(30),0],[-m*cos(60),m*sin(60),0],[0,0,h],[0,m,h],[m,0,h],[-m,0,h],[0,-m,h],[m*cos(30),m*sin(30),h],[m*cos(60),m*sin(60),h],[m*cos(30),-m*sin(30),h],[m*cos(60),-m*sin(60),h],[-m*cos(30),-m*sin(30),h],[-m*cos(60),-m*sin(60),h],[-m*cos(30),m*sin(30),h],[-m*cos(60),m*sin(60),h]])
     pt_dict = {}
     print(pts)
     imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeff)
     print(imgpts)
     for i in range(len(pts)):
    		 pt_dict[tuple(pts[i])] = tuple(imgpts[i].ravel()) 
     src = pt_dict[tuple(pts[0])]  
     dst5= pt_dict[tuple(pts[1])]
     dst6= pt_dict[tuple(pts[2])]
     dst7= pt_dict[tuple(pts[3])]
     dst8= pt_dict[tuple(pts[4])]
     p5=pt_dict[tuple(pts[5])]
     p6=pt_dict[tuple(pts[6])]
     p7=pt_dict[tuple(pts[7])]
     p8=pt_dict[tuple(pts[8])] 
     p9=pt_dict[tuple(pts[9])]   
     p10=pt_dict[tuple(pts[10])]   
     p11=pt_dict[tuple(pts[11])]   
     p12=pt_dict[tuple(pts[12])]   
     src_1 = pt_dict[tuple(pts[13])]  
     dst5_1= pt_dict[tuple(pts[14])]
     dst6_1= pt_dict[tuple(pts[15])]
     dst7_1= pt_dict[tuple(pts[16])]
     dst8_1= pt_dict[tuple(pts[17])]
     p5_1=pt_dict[tuple(pts[18])]
     p6_1=pt_dict[tuple(pts[19])]
     p7_1=pt_dict[tuple(pts[20])]
     p8_1=pt_dict[tuple(pts[21])] 
     p9_1=pt_dict[tuple(pts[22])]   
     p10_1=pt_dict[tuple(pts[23])]   
     p11_1=pt_dict[tuple(pts[24])]   
     p12_1=pt_dict[tuple(pts[25])]   
     
     img = cv2.line(img, src, dst5, (255,255,0), 1)
     img = cv2.line(img, src, dst6, (255,255,0), 1)
     img = cv2.line(img, src, dst7, (255,255,0), 1)
     img = cv2.line(img, src, dst8, (255,255,0), 1)
     img = cv2.line(img, src, p5, (255,255,0), 1)
     img = cv2.line(img, src, p6, (255,255,0), 1)
     img = cv2.line(img, src, p7, (255,255,0), 1)
     img = cv2.line(img, src, p8, (255,255,0), 1)
     img = cv2.line(img, src, p9, (255,255,0), 1)
     img = cv2.line(img, src, p10, (255,255,0), 1)
     img = cv2.line(img, src, p11, (255,255,0), 1)
     img = cv2.line(img, src, p12, (255,255,0), 1)
    
     img = cv2.line(img, src_1, dst5_1, (255,0,145), 1)
     img = cv2.line(img, src_1, dst6_1, (255,0,145), 1)
     img = cv2.line(img, src_1, dst7_1, (255,0,145), 1)
     img = cv2.line(img, src_1, dst8_1, (255,0,145), 1)
     img = cv2.line(img, src_1, p5_1, (255,0,145), 1)
     img = cv2.line(img, src_1, p6_1, (255,0,145), 1)
     img = cv2.line(img, src_1, p7_1, (255,0,145), 1)
     img = cv2.line(img, src_1, p8_1, (255,0,145), 1)
     img = cv2.line(img, src_1, p9_1, (255,0,145), 1)
     img = cv2.line(img, src_1, p10_1, (255,0,145), 1)
     img = cv2.line(img, src_1, p11_1, (255,0,145), 1)
     img = cv2.line(img, src_1, p12_1, (255,0,145), 1)
     
     img = cv2.line(img, src_1, src, (255,255,255), 1)
     img = cv2.line(img, dst5_1, dst5, (255,255,255), 1)
     img = cv2.line(img, dst6_1, dst6, (255,255,255), 1)
     img = cv2.line(img, dst7_1, dst7, (255,255,255), 1)
     img = cv2.line(img, dst8_1, dst8, (255,255,255), 1)
     img = cv2.line(img, p5, p5_1, (255,255,255), 1)
     img = cv2.line(img, p6, p6_1, (255,255,255), 1)
     img = cv2.line(img, p7, p7_1, (255,255,255), 1)
     img = cv2.line(img, p8, p8_1, (255,255,255), 1)
     img = cv2.line(img, p9, p9_1, (255,255,255), 1)
     img = cv2.line(img, p10, p10_1, (255,255,255), 1)
     img = cv2.line(img, p11, p11_1, (255,255,255), 1)
     img = cv2.line(img, p12, p12_1, (255,255,255), 1)
	 
	##################################################################
     return img

if __name__=="__main__":
    cam, dist = getCameraMatrix()
    img = cv2.imread("/home/abhi/Downloads/Task 1/Task 1.1/TestCases/image_1.jpg")
    aruco_list = detect_markers(img, cam, dist)
    for i in aruco_list:
    # img = drawAxis(img, aruco_list, i[0], cam, dist)
     #img = drawCube(img, aruco_list, i[0], cam, dist)
      img = drawCylinder(img, aruco_list, i[0], cam, dist)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
