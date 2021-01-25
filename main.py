import cv2
import numpy as np
import sys
import time

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

    cap = cv2.VideoCapture(0)

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher()
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)


    while True:
        _, frame = cap.read()

        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kp_grayframe, desc_grayframe = orb.detectAndCompute(grayframe, None)
        kp_image, desc_image = orb.detectAndCompute(img, None)

        # matches = bf.match(desc_grayframe, desc_image)
        matches = flann.knnMatch(desc_image.astype(np.float32), desc_grayframe.astype(np.float32), k=2)
        good_points = []
        #distance is the difference of 2 vector
        for m, n in matches:
            if m.distance < 0.7  * n.distance:
            # if True:
                good_points.append(m)


        matching_img = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe, flags=2) 
        # matches = flann.match(desc_grayframe, desc_image)
            # matching_img = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, matches, None)
             
            # query_pts = np.float32([kp_image[m.queryIdx].pt for m in matches]).reshape(-1, -1, 2)
            # train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # mat, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

            # matches_mask = mask.revel().tolist()

            # h, w = img.shape[:2]
            # pts = np.float32([
            #     [0, 0], [0, h], [w, h], [w, 0]
            # ]).reshape(-1, 1, 2)

            # warped = cv2.warpPerspective(frame, mat, (h, w))
            # dst = cv2.perspectiveTransform(pts, mat)
            # homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        homography = None
        print(len(good_points))
        if len(good_points) >  5:
            
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
                # real_train_ptrs = cv2.perspectiveTransform(query_pts, mat)
                # for element in train_pts:
                #     print(element)
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
        if homography is not None:
            cv2.imshow("Homography", homography)
        else:
            cv2.imshow("Homography", frame)


        cv2.imshow("matching", matching_img)
        # cv2.imshow("Homography", homography)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
