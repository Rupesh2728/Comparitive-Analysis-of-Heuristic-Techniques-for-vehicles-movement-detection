import cv2
import numpy as np
import os


def extract_background(video_path, num_frames=250, components=5, var_threshold=120):
    video_frame_files = sorted(os.listdir(video_path))
    gmm = cv2.createBackgroundSubtractorMOG2(history=num_frames, varThreshold=var_threshold, detectShadows=False)
    img_path = os.path.join(frames_folder, video_frame_files[0])
    frame = cv2.imread(img_path)
    frame_height, frame_width, _ = frame.shape
    count = 0
    print(frame_height, frame_width)
    for f_file in video_frame_files:
        if count > num_frames:
            break
        img_path = os.path.join(frames_folder, f_file)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        mask = gmm.apply(frame)
        count += 1

    return gmm, frame_height, frame_width


def extract_road_region(video_path):
    trained_gmm, height, width = extract_background(video_path)
    total_foreground = np.zeros((height, width), dtype=np.uint8)
    frame_count = 0
    prev_frame = None

    frame_files = sorted(os.listdir(video_path))

    for frame_file in frame_files:
        # print(frame_count)
        if frame_count > 500:
            print("Road Mask Formed...")
            break
        frame_path = os.path.join(frames_folder, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        frame_count += 1
        img_blurred = cv2.GaussianBlur(img, (5, 5), 0)

        # Applying frame differencing
        if prev_frame is not None:
            diff_frame = cv2.absdiff(prev_frame, img_blurred)
            # cv2.imshow("MainImage", diff_frame)
            diff_gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
            _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
            # cv2.imshow("Threshold Img", diff_thresh)
            foreground_mask = trained_gmm.apply(img_blurred)
            # cv2.imshow("foreground Img mask", foreground_mask)
            foreground_mask = cv2.bitwise_and(foreground_mask, diff_thresh)  # Combine with GMM mask
            # cv2.imshow("foreground Img mask", foreground_mask)
        else:
            foreground_mask = trained_gmm.apply(img_blurred)

        if prev_frame is not None:
            total_foreground = cv2.bitwise_or(total_foreground, foreground_mask)

        prev_frame = img_blurred.copy()

        foreground = cv2.bitwise_and(img, img, mask=foreground_mask)
        background = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(foreground_mask))

        # cv2.imshow("MainImage", img_blurred)
        # cv2.imshow('foreground.png', total_foreground)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    return total_foreground


frames_folder = './TestDataVideos/MVI_39211'
total_foreground = extract_road_region(frames_folder)

frame_files = sorted(os.listdir(frames_folder))
for frame_file in frame_files:
    frame_path = os.path.join(frames_folder, frame_file)
    img = cv2.imread(frame_path)
    if img is None:
        continue

    light_red = np.full_like(img, (255, 200, 200), dtype=np.uint8)
    light_red_mask = cv2.bitwise_and(light_red, light_red, mask=total_foreground)
    result = cv2.addWeighted(img, 1, light_red_mask, 0.5, 0)

    cv2.imshow('Result', result)
    cv2.imshow('foreground.png', total_foreground)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


