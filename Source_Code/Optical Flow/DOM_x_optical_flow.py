import numpy as np
import cv2
import time
import csv
import sys
import tracemalloc

from RoadSurface_Extraction import extract_road_region


# orb_params = {
#     'nfeatures': 100,  # Number of keypoints to detect
#     'scaleFactor': 1.3,  # Pyramid scale factor
#     'nlevels': 8,  # Number of pyramid levels
#     'edgeThreshold': 31,  # Edge threshold
#     'firstLevel': 0,  # First level of pyramid
#     'WTA_K': 2,  # Number of points that produce each element of the BRIEF descriptor
#     'scoreType': cv2.ORB_HARRIS_SCORE,  # Score type (Harris score or FAST score)
#     'patchSize': 31,  # Size of the patch used by the oriented BRIEF descriptor
#     'fastThreshold': 20,  # FAST threshold
# }
# orb = cv2.ORB()
# orb = orb.create(**orb_params)


# Parameters for Lucas-Kanade optical flow and feature detection
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# feature_params = dict(maxCorners=20,
#                       qualityLevel=0.3,
#                       minDistance=10,
#                       blockSize=7)

feature_params = dict(maxCorners=30,
                      qualityLevel=0.3,
                      minDistance=5,
                      blockSize=7)

# feature_params = dict(maxCorners=70,
#                       qualityLevel=0.3,
#                       minDistance=5,
#                       blockSize=7)
trajectory_len = 100
detect_interval = 5
trajectories = []
frame_idx = 0
direction_trajectories = {
    'top': [],
    'down': [],
    'left': [],
    'right': [],
    'other': [],
}

final_op = {
    'top': [],
    'down': [],
    'left': [],
    'right': [],
    'other': [],
}

left_limit1 = float('-1')
right_limit1 = float('-1')
left_limit_trajectory1 = []
right_limit_trajectory1 = []

left_limit2 = float('-1')
right_limit2 = float('-1')
left_limit_trajectory2 = []
right_limit_trajectory2 = []


video_path = "../Videos/footage_4.mp4"
# total_foreground = extract_road_region(video_path)

cap = cv2.VideoCapture(video_path)
count=0

with open('mem_output_footage_4.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame Number", "Memory usage(in KB)"])
    while count < 400:
        # Start time to calculate FPS
        start = time.time()

        suc, frame = cap.read()
        count += 1
        if not suc:
            break
        height = frame.shape[0]
        width = frame.shape[1]
        # foreground_mask = trained_gmm.apply(frame)
        # foreground = cv2.bitwise_and(frame, frame, mask=foreground_mask)
        # frame_cropped = cv2.bitwise_and(frame, frame, mask=total_foreground)
        # cv2.imshow("Total Foreground", frame_cropped)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Cropped Frame", frame_gray)
        img = frame.copy()

        # print(left_limit_trajectory1)
        # print(right_limit_trajectory2)
        # cv2.circle(img, (1076, 704), 10, 0, -1)
        tracemalloc.start()
        # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
        if len(trajectories) > 0:
            img0, img1 = prev_gray, frame_gray
            # print(trajectories)
            p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
            # print(p0)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1

            new_trajectories = []
            # Get all the trajectories
            for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                if np.linalg.norm(np.array(trajectory[-1]) - np.array((x, y))) < 1.7:
                    continue

                trajectory.append((x, y))

                if len(trajectory) > trajectory_len:
                    del trajectory[0]

                if trajectory[0][1] < height - 100:
                    new_trajectories.append(trajectory)

                # Newest detected point
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

            trajectories = new_trajectories

            # Draw all the trajectories and estimate direction vectors of vehicles
            for trajectory in trajectories:
                if len(trajectory) >= 2:
                    # Calculate the direction vector from the last two points
                    x1, y1 = trajectory[-2]
                    x2, y2 = trajectory[-1]
                    direction_vector = np.array([x2 - x1, y2 - y1])
                    # Mark the road boundary based on the direction vector
                    if np.linalg.norm(direction_vector) > 0:
                        direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize
                        # Convert direction vector to angle
                        angle = np.arctan2(direction_vector[1], direction_vector[0])
                        # Convert angle to hue value (0-180)
                        hue = int((angle + np.pi) * 90 / np.pi)

                        if 17 <= hue <= 27:
                            direction_trajectories['top'].append(trajectory)
                        elif 153 <= hue <= 168:
                            direction_trajectories['down'].append(trajectory)
                        elif 85 <= hue <= 97:
                            direction_trajectories['right'].append(trajectory)
                        elif 172 <= hue <= 179:
                            direction_trajectories['left'].append(trajectory)
                        else:
                            direction_trajectories['other'].append(trajectory)

                        # print("Hue:", hue)
                        color = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]))
                        # Draw the trajectory with color
                        cv2.polylines(img, [np.int32(trajectory)], False, color, thickness=2)
            # print('top:', direction_trajectories['top'])
            # print('down:', direction_trajectories['down'])

            # Iterate through trajectories of vehicles moving towards the camera
            for trajectory in direction_trajectories['left']:
                for pt in trajectory:
                    if left_limit1 == float('-1'):
                        left_limit1 = pt[1]
                        left_limit_trajectory1 = trajectory
                    else:
                        if pt[1] <= left_limit1:
                            left_limit1 = pt[1]
                            left_limit_trajectory1 = trajectory

                    if right_limit1 == float('-1'):
                        right_limit1 = pt[1]
                        right_limit_trajectory1 = trajectory
                    else:
                        if pt[1] >= right_limit1:
                            right_limit1 = pt[1]
                            right_limit_trajectory1 = trajectory

            cv2.polylines(img, [np.int32(left_limit_trajectory1)], False, (0, 0, 0), thickness=2)
            cv2.polylines(img, [np.int32(right_limit_trajectory1)], False, (0, 0, 0), thickness=2)

            # Iterate through trajectories of vehicles moving away from the camera
            for trajectory in direction_trajectories['right']:
                for pt in trajectory:
                    if left_limit2 == float('-1'):
                        left_limit2 = pt[1]
                        left_limit_trajectory2 = trajectory
                    else:
                        if pt[1] <= left_limit2:
                            left_limit2 = pt[1]
                            left_limit_trajectory2 = trajectory

                    if right_limit2 == float('-1'):
                        right_limit2 = pt[1]
                        right_limit_trajectory2 = trajectory
                    else:
                        if pt[1] >= right_limit2:
                            right_limit2 = pt[1]
                            right_limit_trajectory2 = trajectory

            cv2.polylines(img, [np.int32(left_limit_trajectory2)], False, (0, 0, 255), thickness=2)
            cv2.polylines(img, [np.int32(right_limit_trajectory2)], False, (0, 0, 255), thickness=2)

        # Update interval - When to update and detect new features
        if frame_idx % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255

            # Latest point in the latest trajectory
            for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
                cv2.circle(mask, (x, y), 5, 0, -1)

            # Detect good features to track
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
            if p is not None:
                # If good features can be tracked - add that to the trajectories
                for x, y in np.float32(p).reshape(-1, 2):
                    trajectories.append([(x, y)])

            # kp = orb.detect(img)
            # if kp:
            #     for point in kp:
            #         x, y = point.pt
            #         trajectories.append([(x, y)])

        direction_trajectories['top'] = []
        direction_trajectories['down'] = []
        direction_trajectories['left'] = []
        direction_trajectories['right'] = []

        # Display the updated image
        cv2.imshow('Optical Flow', img)

        frame_idx += 1
        prev_gray = frame_gray

        # End time
        end = time.time()
        # Calculate the FPS for current frame detection
        # fps = 1 / (end - start)
        current, peak = tracemalloc.get_traced_memory()
        memory_usage_kb = current / 1024  # Convert to KB
        print("Current memory usage (KB):", memory_usage_kb)
        tracemalloc.stop()  # Stop tracing memory allocations

        writer.writerow([count, memory_usage_kb])

        # Show Results
        # cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Optical Flow', img)
        # cv2.imshow('Optical Flow Mask', mask)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
