import cv2
import numpy as np


def extract_background(video_path, num_frames=250, components=5, var_threshold=120):
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    height, width, _ = frame.shape
    print(height, width)

    gmm = cv2.createBackgroundSubtractorMOG2(history=num_frames, varThreshold=var_threshold, detectShadows=False)

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        mask = gmm.apply(frame)
    cap.release()

    return gmm, height, width


def chow_kaneko_threshold(image, window_size=15, k=0.2):
    """
    Apply the Chow and Kaneko adaptive thresholding algorithm to an image.

    :param image: Grayscale input image
    :param window_size: Size of the local region (must be odd)
    :param k: A constant factor to adjust the threshold
    :return: Thresholded binary image
    """

    # Convert the image to float32 to avoid overflow in the calculations
    image_float = image.astype(np.float32)

    # Calculate the local mean and local squared mean using a box filter
    local_mean = cv2.boxFilter(image_float, cv2.CV_32F, (window_size, window_size))
    local_sq_mean = cv2.boxFilter(image_float ** 2, cv2.CV_32F, (window_size, window_size))

    # Calculate the local variance and local standard deviation
    local_var = local_sq_mean - local_mean ** 2
    local_std = np.sqrt(local_var)

    # Calculate the adaptive threshold using the Chow and Kaneko formula
    T = local_mean * (1 + k * ((local_std / 128) - 1))

    # Apply the threshold to the original image
    output = np.where(image_float > T, 255, 0).astype(np.uint8)

    return output

def calculate_accuracy(ground_truth, result):
    """
    Calculate the accuracy between ground truth and result images.

    Args:
    ground_truth (numpy array): Pixel data for the ground truth image.
    result (numpy array): Pixel data for the result image.

    Returns:
    float: Accuracy percentage.
    """
    if ground_truth.shape != result.shape:
        raise ValueError("Ground truth and result images must have the same shape.")

    # Compare the pixels
    matches = np.sum(ground_truth == result)
    total_pixels = ground_truth.size

    # Calculate accuracy
    accuracy = (matches / total_pixels) * 100
    return accuracy

def extract_road_region(video_path):
    trained_gmm, height, width = extract_background(video_path)
    total_foreground = np.zeros((height, width), dtype=np.uint8)
    frame_count = 0
    cap2 = cv2.VideoCapture(video_path)
    prev_frame = None

    while True:
        # print(frame_count)
        if frame_count > 400:
            print("Road Mask Formed...")
            break

        ret2, img = cap2.read()
        if not ret2:
            break
        frame_count += 1
        img_blurred = cv2.GaussianBlur(img, (7, 7), 0)
        # cv2.imshow("Blurred Img", img_blurred)

        # Applying frame differencing
        if prev_frame is not None:
            diff_frame = cv2.absdiff(prev_frame, img_blurred)
            # cv2.imshow("MainImage", diff_frame)
            diff_gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)

            # _, diff_thresh = cv2.threshold(diff_gray, 40, 255, cv2.THRESH_BINARY)

            # Otsu Threshold
            _, diff_thresh = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # # Apply adaptive local thresholding instead of fixed thresholding
            # diff_thresh = cv2.adaptiveThreshold(diff_gray, 255,
            #                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                                     cv2.THRESH_BINARY,
            #                                     11, 2)
            # diff_thresh = cv2.bitwise_not(diff_thresh)

            # Apply Chow and Kaneko adaptive thresholding
            # diff_thresh = chow_kaneko_threshold(diff_gray, window_size=5, k=0.2)
            cv2.imshow("Threshold Img", diff_thresh)

            foreground_mask = trained_gmm.apply(img_blurred)
            foreground_mask = cv2.bitwise_and(foreground_mask, diff_thresh)
            cv2.imshow("foreground Img mask", foreground_mask)

            # cv2.imshow("Threshold Img", diff_thresh)
            # foreground_mask = trained_gmm.apply(img_blurred)
            # cv2.imshow("foreground Img mask", foreground_mask)
            # foreground_mask = cv2.bitwise_and(foreground_mask, diff_thresh)  # Combine with GMM mask

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
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap2.release()
    cv2.destroyAllWindows()

    return total_foreground


def extract_road_region2(video_path):
    trained_gmm, height, width = extract_background(video_path)
    total_foreground = np.zeros((height, width), dtype=np.uint8)
    frame_count = 0
    cap2 = cv2.VideoCapture(video_path)
    prev_frame = None

    while True:
        # print(frame_count)
        if frame_count > 400:
            print("Road Mask Formed...")
            break

        ret2, img = cap2.read()
        if not ret2:
            break
        frame_count += 1
        img_blurred = cv2.GaussianBlur(img, (7, 7), 0)
        # cv2.imshow("Blurred Img", img_blurred)

        # Applying frame differencing
        if prev_frame is not None:
            diff_frame = cv2.absdiff(prev_frame, img_blurred)
            # cv2.imshow("MainImage", diff_frame)
            diff_gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)

            _, diff_thresh = cv2.threshold(diff_gray, 40, 255, cv2.THRESH_BINARY)

            # Otsu Threshold
            # _, diff_thresh = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # # Apply adaptive local thresholding instead of fixed thresholding
            # diff_thresh = cv2.adaptiveThreshold(diff_gray, 255,
            #                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                                     cv2.THRESH_BINARY,
            #                                     11, 2)
            # diff_thresh = cv2.bitwise_not(diff_thresh)

            # Apply Chow and Kaneko adaptive thresholding
            # diff_thresh = chow_kaneko_threshold(diff_gray, window_size=5, k=0.2)
            cv2.imshow("Threshold Img", diff_thresh)

            foreground_mask = trained_gmm.apply(img_blurred)
            foreground_mask = cv2.bitwise_and(foreground_mask, diff_thresh)
            cv2.imshow("foreground Img mask", foreground_mask)

            # cv2.imshow("Threshold Img", diff_thresh)
            # foreground_mask = trained_gmm.apply(img_blurred)
            # cv2.imshow("foreground Img mask", foreground_mask)
            # foreground_mask = cv2.bitwise_and(foreground_mask, diff_thresh)  # Combine with GMM mask

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
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap2.release()
    cv2.destroyAllWindows()

    return total_foreground

video_path = '../Videos/footage_2.mp4'
total_foreground = extract_road_region(video_path)
Ground_truth = extract_road_region2(video_path)

Accuracy = calculate_accuracy(Ground_truth, total_foreground)
print('Accuracy : ', Accuracy)

cap3 = cv2.VideoCapture(video_path)

while True:
    ret3, img = cap3.read()
    if not ret3:
        break
    frame_cropped = cv2.bitwise_and(img, img, mask=total_foreground)
    cv2.imshow('Result', frame_cropped)
    cv2.imshow('foreground.png', total_foreground)
    cv2.imshow('foreground2.png', Ground_truth)
    # cv2.imshow("Main Img", img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap3.release()
cv2.destroyAllWindows()

