imge1 = cv.imread('notredom_test.jpg')   
img1 = cv.cvtColor(imge1, cv.COLOR_BGR2GRAY)       # queryImage

imge2 = cv.imread('notredom_train.jpg')   
img2 = cv.cvtColor(imge2, cv.COLOR_BGR2GRAY)       # trainImage

# creating SIFT feature detector object, (Scale-Invariant Feature Transform)
# https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
sift = cv.xfeatures2d.SIFT_create()                

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
