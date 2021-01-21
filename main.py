import cv2
import numpy as np
import sys
import time

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

    cap = cv2.VideoCapture(0)

    #feature
    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.SIFT()
    kp_image, desc_image = sift.detectAndCompute(img, None)

    img = cv2.drawKeypoints(img, kp_image, img)

    #feature matching
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # bf = cv2.BFMatcher()
    while True:
        _, frame = cap.read()

        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
        

        grayframe = cv2.drawKeypoints(grayframe, kp_grayframe, grayframe)



        # delta_time = time.process_time()
        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
        # delta_time = time.process_time() - delta_time
        
        # delta_time = time.process_time()
        # matches = bf.knnMatch(desc_image, desc_grayframe, k=2)

        # delta_time = time.process_time() - delta_time
        # print("matching: {}".format(delta_time))

        good_points = []
        #distance is the difference of 2 vector
        for m, n in matches:
            if m.distance < 0.7  * n.distance:
            # if True:
                good_points.append(m)


        img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe, flags= 2)#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #homography
        # print(len(kp_grayframe))
        # print(desc_grayframe.shape)
        # print(desc_grayframe[0])
        # print()
        # print(len(good_points))
        if len(good_points) > 10:
            
            # delta_time = time.process_time()
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        
            # print(np.linalg.norm((desc_image[good_points[0].queryIdx]) - (desc_grayframe[good_points[0].trainIdx])))
            # print()
            train_pts =np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            

            mat, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            # mat_, mask_ = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

            
            matches_mask = mask.ravel().tolist()

            h, w = img.shape[:2]
            pts = np.float32([
                [0, 0], [0, h], [w, h], [w, 0]
            ]).reshape(-1, 1, 2)

            warped = cv2.warpPerspective(frame, mat, (h, w))
            dst = cv2.perspectiveTransform(pts, mat)
            real_train_ptrs = cv2.perspectiveTransform(query_pts, mat)
            for element in train_pts:
                print(element)
            # print(real_train_ptrs)
            # M = cv2.getPerspectiveTransform(np.int32(dst), pts)
            # warped = cv2.warpPerspective(frame, M, (img.shape[1], img.shape[0]))
            # print(dst)
            # tl, tr, br, bl = dst
            # M = cv2.getPerspectiveTransform(dst, pts)
            # warped = cv2.warpPerspective(frame, M, (w, h))
            # cv2.imshow("warped", warped)
            # cv2.imshow("dst", dst)
            # print(dst)
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            
            # delta_time = time.process_time() - delta_time
            # print(delta_time)

            cv2.imshow("Homography", homography)
        else:
            cv2.imshow("Homography", frame)

        # cv2.imshow("Image", img)
        # cv2.imshow("gray frame", grayframe)
        cv2.imshow("img3", img3)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('c'):
            cv2.imwrite('frame.png', frame)
            cv2.imwrite('img3.png', img3)
            cv2.imwrite('homography.png', homography)
    

    cap.release()
    cv2.destroyAllWindows()
