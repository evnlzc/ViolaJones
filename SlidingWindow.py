"""
File includes:
    Function for using a sliding window to search images for faces
    Function for suppressing weaker subwindows overlapping with a stronger subwindow
    Functions for evaluating classifier performance or detecting faces
"""

import cv2
import time
from pathlib2 import Path

import HaarLikeFeatures as HaarF
import WeakClassifier as WeakC


def sliding_window(cascade, img, sliding_window_dim=19, step_size_h=4, step_size_w=4):
    """
    Iterate scaled sliding windows over image

    :param cascade: monolithic classifier or cascade of classifiers      (list)
    :param img: image to slide window over                               (np.ndarray)
    :param sliding_window_dim: dimension of sliding window (square)      (int)
    :param step_size_h: increment to move sliding window down            (int)
    :param step_size_w: increment to move sliding window right           (int)
    :return: identified faces' bounding boxes                            (np.array)
    """
    detections, scores = [], []
    training_window_dims = (sliding_window_dim, sliding_window_dim)
    max_height, max_width = img.shape[0:2]

    # While the sliding window is smaller than the image
    while sliding_window_dim <= max_height and sliding_window_dim <= max_width:
        # For each current sliding window that fits within the image
        height_limit = max_height - sliding_window_dim + 1
        width_limit = max_width - sliding_window_dim + 1
        for h in range(0, height_limit, step_size_h):
            for w in range(0, width_limit, step_size_w):
                # Calculate integral image of normalized subwindow
                sub_img = img[h:h+sliding_window_dim, w:w+sliding_window_dim]
                sub_img = cv2.resize(sub_img, training_window_dims)
                intg_img = HaarF.to_integral_image(sub_img)
                # Evaluate cascade
                failed = False
                alpha_sum = 0
                for layer, sc in cascade.items():
                    alpha = 0
                    [wc_lst, threshold] = sc
                    alpha = WeakC.evaluate_weak_classifiers(intg_img, wc_lst)
                    if alpha < threshold:
                        failed = True
                        break
                    alpha_sum += alpha
                if not failed:
                    # Integral image is padded - save rectangle coordinates on original image
                    box = [w - 1, h - 1, w + sliding_window_dim - 1, h + sliding_window_dim - 1]
                    detections.append(box)
                    scores.append(alpha_sum)
        # Scale up the sliding window size
        sliding_window_dim = int(sliding_window_dim * 1.25)
    detections = non_maximal_suppression(detections, scores)
    return detections


def non_maximal_suppression(detections, scores):
    """
    Retain the subwindow with the greatest cumulative weight and remove other overlaps

    :param detections: subwindows top-left and bottom-right coordinates  (list)
    :param scores: subwindow sum of weights                              (list)
    :return: subwindows to keep
    """
    if len(detections) == 0:
        return []
    if len(detections) == 1:
        return list(detections)
    singular_detections = []

    # While subboxes remain to be evaluated or reevaluated
    while len(detections) != 0:
        [m_x1, m_y1, m_x2, m_y2] = detections[0]
        recycle = []
        did_overlap, retain = False, True

        # Compare the first box with all other boxes
        for idx in range(1, len(detections)):
            box_test = detections[idx]
            # Check if the boxes overlap
            in_x_range1 = m_x1 <= box_test[0] <= m_x2 or m_x1 <= box_test[2] <= m_x2
            in_y_range1 = m_y1 <= box_test[1] <= m_y2 or m_y1 <= box_test[3] <= m_y2
            in_x_range2 = box_test[0] <= m_x1 <= box_test[2] or box_test[0] <= m_x2 <= box_test[2]
            in_y_range2 = box_test[1] <= m_y1 <= box_test[3] or box_test[1] <= m_y2 <= box_test[3]
            if (in_x_range1 and in_y_range1) or (in_x_range2 and in_y_range2):
                did_overlap = True
                # If the initial box has a lower threshold, stop and don't append
                if scores[0] < scores[idx]:
                    retain = False
                    recycle.append(idx)
            else:
                recycle.append(idx)

        # Check which subwindows are to be retained for further evaluation/saving
        if not did_overlap:
            singular_detections.append(detections[0])
        elif did_overlap and retain:
            recycle.append(0)
        new_detections, new_scores = [], []
        for idx in recycle:
            new_detections.append(detections[idx])
            new_scores.append(scores[idx])
        detections, scores = new_detections, new_scores

    return singular_detections


def evaluate_classifier_performance(classifier, p_test_set, n_test_set, extension):
    """
    Test a strong classifier or cascade on sets of ground truth test data (0 or 1 face per image)

    :param classifier: monolithic or cascade classifier         (list or dict)
    :param p_test_set: face test set directory                  (str)
    :param n_test_set: nonface test set directory               (str)
    :param extension: file extension of images                  (str)
    :return: None
    """
    t0 = time.time()
    paths = [sorted(Path(p_test_set).glob("**/*." + extension)),
             sorted(Path(n_test_set).glob("**/*." + extension))]
    performance = []

    # If dealing with a monolithic classifier
    if type(classifier) == list:
        threshold = sum([wc.weight for wc in classifier]) / 2
        classifier = {0: [classifier, threshold]}

    for test_set in paths:
        counter = 0
        for img_path in test_set:
            img = cv2.imread(str(img_path))
            detections = sliding_window(classifier, img)
            counter += len(detections)
        performance.append(counter)

    print("Detections for face set: ", str(performance[0]) + " for " + str(len(paths[0])) + " images.")
    print("Detections for nonface set: ", str(performance[1]) + " for " + str(len(paths[1])) + " images.")
    print("Time to perform detections:", time.time() - t0)


def evaluate_images(classifier, image_directory, extension, save_path):
    """
    Saves images with bounding boxes where faces are detected

    :param classifier: monolithic or cascade classifier         (list or dict)
    :param image_directory: directory with images               (str)
    :param extension: extension of images to analyze            (str)
    :param save_path: directory to save images in               (str)
    :return: Non
    """
    t0 = time.time()
    paths = sorted(Path(image_directory).glob("**/*." + extension))
    print("Evaluating", len(paths), "images")
    save_path = Path(save_path)

    if type(classifier) == list:
        threshold = sum([wc.weight for wc in classifier]) / 2
        classifier = {0: [classifier, threshold]}

    img_count = 1
    for img_path in paths:
        img = cv2.imread(str(img_path))
        detections = sliding_window(classifier, img)
        for box in detections:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imwrite(str(save_path / ("result_" + str(img_count) + ".png")), img)
        img_count += 1

    print("Time to perform detections:", time.time() - t0)
