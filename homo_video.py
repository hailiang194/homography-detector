import cv2
import numpy as np
import sys
import time

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

    cap = cv2.VideoCapture(sys.argv[2] if len(sys.argv) == 3 else 0)

    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    # search_params = dict(checks=100)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    corners = [] 
    detect_frame_count = 1
    while cap.isOpened():
        _, frame = cap.read()

        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayframe = cv2.medianBlur(grayframe, 7)
        grayframe = cv2.GaussianBlur(grayframe, (5, 5), 0)
        kp_grayframe, desc_grayframe = orb.detectAndCompute(grayframe, None)
        kp_image, desc_image = orb.detectAndCompute(img, None)

        matches = bf.knnMatch(desc_image, desc_grayframe, k=2)
        # matches = bf.match(desc_image, desc_grayframe)
        # matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
        # good_points = sorted(matches, key=lambda val: val.distance)[:50]
        # m = matches[:]
        # print(matches[0][0])
        #distance is the difference of 2 vector
        good_points = []
        for m, n in matches:
            if m.distance < 0.75  * n.distance:
                # if True:
                good_points.append(m)
        # print("Test=" + str(matches))
        # for match in matches:
        #     if len(match) == 0:
        #         continue

        #     m, n = match if len(match) == 2 else (match[0], match[0])
        #     if m.distance < 0.7 * n.distance:
        #         good_points.append(m)

        matching_img = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe, flags=0) 
        cv2.imshow("matching", matching_img)
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
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)

        # print(np.linalg.norm((desc_image[good_points[0].queryIdx]) - (desc_grayframe[good_points[0].trainIdx])))
        # print()
        train_pts =np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        currentPts = len(query_pts)

        # for i in range(currentPts - 1):
        #     query_pts = np.append(query_pts, [(query_pts[i] + query_pts[i + 1]) / 2.0], axis=0)
        #     train_pts = np.append(train_pts, [(train_pts[i] + train_pts[i + 1]) / 2.0], axis=0)
        for i in range(1, currentPts):
            for j in range(i):
                if np.linalg.norm(train_pts[i] - train_pts[j]) > min(frame.shape[0] * 0.05, frame.shape[1] * 0.05):
                    query_pts = np.append(query_pts, [(query_pts[i] + query_pts[j]) / 2.0], axis=0)
                    train_pts = np.append(train_pts, [(train_pts[i] + train_pts[j]) / 2.0], axis=0)

        # print(query_pts.shape)
        # print(train_pts)
        if query_pts.shape[0] < 4:
            cv2.imshow("Homography", frame)
            continue
        mat, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        # mat_, mask_ = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        if mat is None:
            cv2.imshow("Homography", frame)
            continue

        real_train_pts = cv2.perspectiveTransform(query_pts, mat)
        distance_filtered_query_pts = []
        distance_filtered_train_pts = []


        # print(query_pts[0] + query_pts[1])
        for i in range(len(query_pts)):
            distance = np.linalg.norm(real_train_pts[i][0] - train_pts[i][0])
            # print(distance)
            if distance <=  5.0:
                # if True:
                # print(distance)
                distance_filtered_query_pts.append(query_pts[i])
                distance_filtered_train_pts.append(train_pts[i])

        query_pts = np.float32(distance_filtered_query_pts)
        # print(query_pts)
        train_pts = np.float32(distance_filtered_train_pts)

        print(len(good_points))
        # if len(good_points) >  10:
        # if True:
        if len(good_points) > 10 and ((query_pts.shape[0]) > 0 and len(good_points) <= (query_pts.shape[0])):
            # delta_time = time.process_time()
            # query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)

            # # print(np.linalg.norm((desc_image[good_points[0].queryIdx]) - (desc_grayframe[good_points[0].trainIdx])))
            # # print()
            # train_pts =np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            # currentPts = len(query_pts)

            # for i in range(currentPts - 1):
            #     query_pts = np.append(query_pts, [(query_pts[i] + query_pts[i + 1]) / 2.0], axis=0)
            #     train_pts = np.append(train_pts, [(train_pts[i] + train_pts[i + 1]) / 2.0], axis=0)


            # mat, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            # # mat_, mask_ = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)


            # real_train_pts = cv2.perspectiveTransform(query_pts, mat)
            # distance_filtered_query_pts = []
            # distance_filtered_train_pts = []


            # # print(query_pts[0] + query_pts[1])
            # for i in range(len(query_pts)):
            #     distance = np.linalg.norm(real_train_pts[i][0] - train_pts[i][0])
            #     # print(distance)
            #     if distance <= 10.0:
            #     # if True:
            #         # print(distance)
            #         distance_filtered_query_pts.append(query_pts[i])
            #         distance_filtered_train_pts.append(train_pts[i])

            # query_pts = np.float32(distance_filtered_query_pts)
            # # print(query_pts)
            # train_pts = np.float32(distance_filtered_train_pts)


            # print(query_pts)
            mat, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            # print(mat) 
            if mat is None:
                cv2.imshow("Homography", frame)
                continue
            matches_mask = mask.ravel().tolist()

            h, w = img.shape[:2]
            pts = np.float32([
                [0, 0], [0, h], [w, h], [w, 0]
            ]).reshape(-1, 1, 2)

            warped = cv2.warpPerspective(frame, mat, (h, w))
            dst = cv2.perspectiveTransform(pts, mat)

            # print(dst.shape)
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
            # print(query_pts)
            # print(area)
            #     cv2.imshow("Homography", frame)
            #     continue
            edge = np.float32([
                dst[1][0] - dst[0][0],
                dst[2][0] - dst[1][0],
                dst[3][0] - dst[2][0],
                dst[0][0] - dst[3][0]
            ])
            
            edge_len = np.linalg.norm(edge, axis=1)
            if abs((np.amax(edge_len) / np.amin(edge_len)) - (max(img.shape[:2]) / min(img.shape[:2]))) >= 0.4:
                cv2.imshow("Homography", frame)
                continue

            get_cos = lambda x, y: (np.dot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))

            cos = np.float32([
                get_cos(edge[0], -edge[3]),
                get_cos(-edge[0], edge[1]),
                get_cos(-edge[1], edge[2]),
                get_cos(-edge[2], edge[3])
            ])

            corner = np.arccos(cos) * 180 / np.pi
            if abs(np.sum(corner) - 360.0) > 2:
                cv2.imshow("Homography", frame)
                continue
            
            if np.min(corner) < 40:
                cv2.imshow("Homography", frame)
                continue
            elif np.max(corner) >= 110:
                cv2.imshow("Homography", frame)
                continue
            # print(corners)
            # aveger_cor = np.sum(np.float32(corners), axis=0) / len(corners) if len(corners) > 0 else corners

            corners.append(corner)

            if len(corners) > 5:
                corners.pop(0)
            
            # print(corner)
            aver_corner = (np.sum(corners, axis=0) / len(corners))

            # print(corner)
            homography = cv2.polylines(frame.copy(), [np.int32(dst)], True, (255, 0, 0), 3)
            print(corner)
            if np.max(aver_corner - corner) > 15:
                print("FAILED")
                cv2.imwrite("./frame/homo_{}.png".format(detect_frame_count), homography)
                cv2.imwrite("./frame/frame_{}.png".format(detect_frame_count), frame)
                detect_frame_count = detect_frame_count + 1

            # area = (cv2.contourArea(dst))
            # if area < 10000:
            #     cv2.imwrite("./frame/homo_{}.png".format(detect_frame_count), homography)
            #     cv2.imwrite("./frame/frame_{}.png".format(detect_frame_count), frame)
            #     detect_frame_count = detect_frame_count + 1


        img_h, img_w = img.shape[:2]
        frame_h, frame_w = frame.shape[:2]

        match_w = img_w + frame_w
        match_h = max(frame_h, img_h)

        match = np.zeros((match_h, match_w), dtype="uint8")

        match[0:img_h, 0:img_w] = img[:]
        match[0: frame_h, -frame_w - 1:-1] = grayframe[:]

        match = np.dstack([match] * 3)

        for q_pt, t_pt in zip(query_pts, train_pts):
            # start_pt = q_pt[0].astype("uint8")
            start_pt = (int(q_pt[0][0]), int(q_pt[0][1]))
            # print(start_pt)
            # print(t_pt[0])
            end_pt = (int(t_pt[0][0]) + img_w, int(t_pt[0][1]))
            # print(end_pt)
            cv2.line(match, tuple(start_pt), tuple(end_pt), (0, 0, 255), 1)

        cv2.imshow("new_matching", match)

        # delta46b3f09a39578c3da843ea5a34976da4fe2997e1_time = time.process_time() - delta_time
        # print(delta_time)
        if homography is not None:
            cv2.imshow("Homography", homography)
            # cv2.imwrite("./frame/homo_{}.png".format(detect_frame_count), homography)
            # cv2.imwrite("./frame/frame_{}.png".format(detect_frame_count), frame)
            # detect_frame_count = detect_frame_count + 1
        else:
            cv2.imshow("Homography", frame)
            # cv2.imwrite("./frame/homo_{}.png".format(detect_frame_count), homography)
            # cv2.imwrite("./frame/frame_{}.png".format(detect_frame_count), frame)
            # detect_frame_count = detect_frame_count + 1



        # cv2.imshow("Homography", homography)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
