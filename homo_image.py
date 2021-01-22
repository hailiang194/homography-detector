import cv2
import numpy as np
import sys

if __name__ == "__main__":
    frame = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.ORB_create()

    kp_image, desc_image = sift.detectAndCompute(image, None)

    
    index_params = dict(algorithm=1, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    kp_frame, desc_frame = sift.detectAndCompute(frame, None)

    
    matches = flann.knnMatch(desc_image, desc_frame, k=2)
    # matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).knnMatch(desc_image, desc_frame, k=2)
    good_points = []
    #distance is the difference of 2 vector
    for m, n in matches:
        if m.distance < 0.6  * n.distance:
        # if True:
            good_points.append(m)

    mapping_img = None
    mapping_img = cv2.drawMatches(image, kp_image, frame, kp_frame, good_points, mapping_img, flags=2)

    if len(good_points) > 10:
        # delta_time = time.process_time()
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        
        # print(np.linalg.norm((desc_image[good_points[0].queryIdx]) - (desc_grayframe[good_points[0].trainIdx])))
        # print()
        train_pts =np.float32([kp_frame[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            

        mat, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        # mat, mask = cv2.findFundamentalMat(query_pts, train_pts)
        # mat = cv2.getPerspectiveTransform(query_pts, train_pts)
        # mat_, mask_ = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

        real_train_pts = cv2.perspectiveTransform(query_pts, mat) 
        # for real_point, train_point in zip(real_train_pts, train_pts):
        #     print("{} {}".format(real_point, train_point))
        # print(len(query_pts))
        distance_filtered_query_pts = []
        distance_filtered_train_pts = []
        for i in range(len(query_pts)):
            distance = np.linalg.norm(real_train_pts[i][0] - train_pts[i][0])
            # print(distance)
            if distance <= 10.0:
            # if True:
                # print(distance)
                distance_filtered_query_pts.append(query_pts[i])
                distance_filtered_train_pts.append(train_pts[i])

        query_pts = np.float32(distance_filtered_query_pts)
        # print(query_pts)
        train_pts = np.float32(distance_filtered_train_pts)

        mat, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        # mat, mask = cv2.findFundamentalMat(query_pts, train_pts)

        # mat = cv2.getPerspectiveTransform(query_pts.astype(np.float32), train_pts.astype(np.float32))
        # matches_mask = mask.ravel().tolist()

        h, w = image.shape[:2]
        pts = np.float32([
            [0, 0], [0, h], [w, h], [w, 0]
        ]).reshape(-1, 1, 2)

        warped = cv2.warpPerspective(frame, mat, (h, w))
        dst = cv2.perspectiveTransform(pts, mat)

        img_h, img_w = image.shape[:2]
        frame_h, frame_w = frame.shape[:2]

        match_w = img_w + frame_w
        match_h = max(frame_h, img_h)

        match = np.zeros((match_h, match_w), dtype="uint8")

        match[0:img_h, 0:img_w] = image[:]
        match[0: frame_h, -frame_w - 1:-1] = frame[:]

        match = np.dstack([match] * 3)

        for q_pt, t_pt in zip(query_pts, train_pts):
            # start_pt = q_pt[0].astype("uint8")
            start_pt = (int(q_pt[0][0]), int(q_pt[0][1]))
            # print(start_pt)
            # print(t_pt[0])
            end_pt = (int(t_pt[0][0]) + img_w, int(t_pt[0][1]))
            # print(end_pt)
            cv2.line(match, tuple(start_pt), tuple(end_pt), (0, 0, 255), 1)


        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        cv2.imshow("Homography", homography)
    else:
        cv2.imshow("Homography", frame)
    cv2.imshow("matching", mapping_img)


    
    cv2.imshow("new_matching", match)

    key = cv2.waitKey(0)

    if key == ord('c'):
        cv2.imwrite("homo.png", homography)
        cv2.imwrite("matching.png", mapping_img)


