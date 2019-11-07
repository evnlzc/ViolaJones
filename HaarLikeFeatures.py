"""
File includes:
    Functions for converting an image to a grayscale and integral image
    Functions for image normalization: linear normalization and histogram equalization
    Function for calculating sub-box sums of integral image

    Class for generating Haar-like features for face detection training
        - 5 features used
        - Saves feature sums as pandas dataframes into pickle files for quick access
        - Requires positive and negative training data set
            - Provide paths to extracted images on local machine
            - Expecting equal sized training images of the same file type
"""

import cv2
import pandas as pd
import numpy as np
from pathlib2 import Path


def to_grayscale(img_array):
    """
    Averages out panels of image array if needed (BGR)

    :param img_array: image array to convert    (np.ndarray)
    :return: grayscale image                    (np.ndarray, uint8)
    """
    (height, width, depth) = img_array.shape
    if depth == 1:
        return img_array
    elif depth == 3:
        img_array = img_array.astype(np.uint16)
        gray_img = (img_array[:, :, 0] + img_array[:, :, 1] + img_array[:, :, 2]) / 3
        return gray_img.astype(np.uint8)


def linear_normalization(img_array, new_min=0, new_max=255):
    """
    Linear normalization of an image

    :param img_array: image array to convert                        (np.ndarray, uint8)
    :param new_min: minimum intensity after normalizing             (int)
    :param new_max: maximum intensity after normalizing             (int)
    :return: linearly normalized image                              (np.ndarray, uint8)
    """
    max_intensity = np.amax(img_array, axis=(0, 1))
    min_intensity = np.amin(img_array, axis=(0, 1))
    if max_intensity != min_intensity:
        scaling = (new_max - new_min) / (max_intensity - min_intensity)
        # I_new = (I_old - min) * (new_max - new_min) / (max - min) + new_min
        img_array = np.add(np.multiply(np.subtract(img_array, min_intensity), scaling), new_min)
        normalized = np.ndarray(img_array.shape, dtype=np.uint8)
        normalized[:, :] = img_array[:, :]
        return normalized
    else:
        # Monotone image
        return img_array


def histogram_equalization(img_array, max_intensity=255):
    """
    Histogram equalization of an image

    :param img_array: image array to convert                                (np.ndarray, uint8)
    :param max_intensity: maximum intensity that image array can hold       (int)
    :return: histogram equalized image                                      (np.ndarray, uint8)
    """
    total_pixels = img_array.shape[0] * img_array.shape[1]
    intensity, count = np.unique(img_array, return_counts=True)
    px_counts = dict(zip(intensity, count))
    prob_sum = 0
    # Count pixel occurrences
    for intensity, count in px_counts.items():
        prob_sum += count / total_pixels
        px_counts[intensity] = int(max_intensity * prob_sum)
    # Search the image array for the intensity values and replace with the count
    order_idx = np.argsort(list(px_counts.keys()))
    idx = np.searchsorted(list(px_counts.keys()), img_array, sorter=order_idx)
    referenced = np.asarray(list(px_counts.values()))[order_idx][idx]
    img_array = np.ndarray(referenced.shape, dtype=np.uint8)
    img_array[:, :] = referenced[:, :]
    return img_array


def to_integral_image(img_array):
    """
    Obtain integral image of provided image array
        - 0's padded for expedited calculations (corners and edges)
     ____________           ________________
    |_1_|_1_|_1_|          |_0_|_0_|_0_|_0_|
    |_2_|_2_|_2_|   -->    |_0_|_1_|_2_|_3_|
    |_3_|_3_|_3_|          |_0_|_3_|_6_|_10|
                           |_0_|_6_|_11|_18|

    :param img_array: image array to convert    (np.ndarray)
    :return: integral image array               (np.ndarray, int32)
    """

    (height, width, depth) = img_array.shape
    gray_img = histogram_equalization(linear_normalization(to_grayscale(img_array)))
    img = gray_img.astype(np.int32)

    # Calculate simple sums for leftmost column and top row
    for h in range(1, height):
        img[h, 0] = img[h - 1, 0] + img[h, 0]
    for w in range(1, width):
        img[0, w] = img[0, w - 1] + img[0, w]

    # Perform vectorized addition for larger dimension and iteratively account for remaining elements
    if height > width or height == width:
        for w in range(1, width):
            img[1:height, w] = img[1:height, w] + img[1:height, w - 1] - img[0:height - 1, w - 1]
            for h in range(1, height):
                img[h, w] += img[h - 1, w]
    else:
        for h in range(1, height):
            img[h, 1:width] = img[h, 1:width] + img[h - 1, 1:width] - img[h - 1, 0:width - 1]
            for w in range(1, width):
                img[h, w] += img[h, w - 1]

    # Pad integral image with 0's - no need for conditionals to check dimensions for feature sums
    intg_img = np.zeros(shape=(height + 1, width + 1, 1), dtype=np.int32)
    intg_img[1:height + 1, 1:width + 1, 0] = img
    return intg_img


def calc_box_sum(intg_img, start, height, width):
    """
    Calculates sum for specified sub-box of integral image
    +---+---+
    | nw| ne|       :param intg_img: integral image             (np.ndarray, int32)
    +---+---+       :param start: start pixel coordinates       (tuple, height x width)
    | sw| se|       :param height: height of sub-box            (int)
    +---+---+       :param width: width of sub-box              (int)
                    :return: sum of pixel values in sub-box     (int)
    """
    nw = start[0] - 1, start[1] - 1
    ne = start[0] - 1, start[1] + width
    sw = start[0] + height, start[1] - 1
    se = start[0] + height, start[1] + width
    return (intg_img[se] - intg_img[sw] - intg_img[ne] + intg_img[nw])[0]


def haar_horizontal_half(intg_imgs, start_px, s_dims):
    """
     Calculate difference of Haar-like feature sums - horizontal halves (2 x 1)
     +--+
     |up|       :param intg_imgs: integral images           (list)
     +--+       :param start_px: start pixel coordinates    (tuple, height x width)
     |do|       :param s_dims: dimension of sub-box (h, w)  (tuple)
     +--+       :return: difference of feature regions      (int)
     """
    h_width = s_dims[1] - 1  # Start to corner: width
    h_height = int(s_dims[0] / 2) - 1  # Start to corner: height
    down_start = start_px[0] + h_height + 1, start_px[1]  # Starting point of bottom sub-box
    difference_lst = []
    for intg_img in intg_imgs:
        [up, do] = [calc_box_sum(intg_img, st, h_height, h_width)
                    for st in [start_px, down_start]]
        difference_lst.append(do - up)
    return np.array(difference_lst, dtype=np.int16)


def haar_vertical_half(intg_imgs, start_px, s_dims):
    """
    Calculate difference of Haar-like feature sums - vertical halves (1 x 2)
    +----+-----+
    |left|right|
    +----+-----+
    :param intg_imgs: integral images           (list)
    :param start_px: start pixel coordinates    (tuple, height x width)
    :param s_dims: dimension of sub-box (h, w)  (tuple)
    :return: difference of feature regions      (int)
    """
    h_height = s_dims[0] - 1
    h_width = int(s_dims[1] / 2) - 1
    right_start = start_px[0], start_px[1] + h_width + 1
    difference_lst = []
    for intg_img in intg_imgs:
        [left, right] = [calc_box_sum(intg_img, st, h_height, h_width)
                         for st in [start_px, right_start]]
        difference_lst.append(right - left)
    return np.array(difference_lst, dtype=np.int16)


def haar_burger_proper(intg_imgs, start_px, s_dims):
    """
    Calculate difference of Haar-like feature sums - horizontal thirds (3 x 1)
    +------+
    |topbun|        :param intg_imgs: integral images           (list)
    +------+        :param start_px: start pixel coordinates    (tuple, height x width)
    |patty |        :param s_dims: dimension of sub-box (h, w)  (tuple)
    +------+        :return: difference of feature regions      (int)
    |botbun|
    +------+
    """
    h_height = int(s_dims[0] / 3) - 1
    h_width = s_dims[1] - 1
    patty_start = start_px[0] + h_height + 1, start_px[1]
    base_start = patty_start[0] + h_height + 1, start_px[1]
    difference_lst = []
    for intg_img in intg_imgs:
        [top_bun, patty, base_bun] = [calc_box_sum(intg_img, st, h_height, h_width)
                                      for st in [start_px, patty_start, base_start]]
        difference_lst.append(patty - (top_bun + base_bun))
    return np.array(difference_lst, dtype=np.int16)


def haar_burger_side(intg_imgs, start_px, s_dims):
    """
    Calculate difference of Haar-like feature sums - vertical thirds (1 x 3)
    +----+-----+-----+
    |left|patty|right|
    +----+-----+-----+
    :param intg_imgs: integral images           (list)
    :param start_px: start pixel coordinates    (tuple, height x width)
    :param s_dims: dimension of sub-box (h, w)  (tuple)
    :return: difference of feature regions      (int)
    """
    h_height = s_dims[0] - 1
    h_width = int(s_dims[1] / 3) - 1
    patty_start = start_px[0], start_px[1] + h_width + 1
    right_start = start_px[0], patty_start[1] + h_width + 1
    difference_lst = []
    for intg_img in intg_imgs:
        [left_bun, patty, right_bun] = [calc_box_sum(intg_img, st, h_height, h_width)
                                        for st in [start_px, patty_start, right_start]]
        difference_lst.append(patty - (left_bun + right_bun))
    return np.array(difference_lst, dtype=np.int16)


def haar_chess(intg_imgs, start_px, s_dims):
    """
    Calculate difference of Haar-like feature sums: diagonals (2 x 2)
    +---+---+
    | nw| ne|       :param intg_imgs: integral images           (list)
    +---+---+       :param start_px: start pixel coordinates    (tuple, height x width)
    | sw| se|       :param s_dims: dimensions of sub-box        (tuple)
    +---+---+       :return: difference of feature regions      (int)
    """
    h_height = int(s_dims[0] / 2) - 1
    h_width = int(s_dims[1] / 2) - 1
    ne_start = start_px[0], start_px[1] + h_width + 1
    sw_start = start_px[0] + h_height + 1, start_px[1]
    se_start = start_px[0] + h_height + 1, start_px[1] + h_width + 1
    difference_lst = []
    for intg_img in intg_imgs:
        [nw, ne, sw, se] = [calc_box_sum(intg_img, st, h_height, h_width)
                            for st in [start_px, ne_start, sw_start, se_start]]
        difference_lst.append((ne + sw) - (nw + se))
    return np.array(difference_lst, dtype=np.int16)


def calc_haar_feature_sums(intg_imgs):
    """
    Calculate integral sums for all possible features for face/non-face integral images
    Example: 24 x 24 image --> 162,336 possible features

    :param intg_imgs: integral images                               (list)
    :return: dataframes of possible features and sums               (pd.DataFrame)
    """
    height, width = intg_imgs[0].shape[0:2]
    features_lst = []
    features_sum_lst = []
    # For each non-padded height (row) and width (column) integral image pixel
    for h in range(1, height):
        for w in range(1, width):
            h_inc = 0
            # While feature dimension don't exceed integral image height and image width
            while h + h_inc <= height - 1:
                w_inc = 0
                while w + w_inc <= width - 1:
                    s_dims = h_inc + 1, w_inc + 1  # Dimensions of Haar-like feature (height, width)
                    if s_dims[0] % 2 == 0 and h + h_inc > h:
                        features_lst.append([h, w, 2, 1, s_dims[0], s_dims[1]])
                        features_sum_lst.append(haar_horizontal_half(intg_imgs, (h, w), s_dims))
                    if s_dims[1] % 2 == 0 and w + w_inc > w:
                        features_lst.append([h, w, 1, 2, s_dims[0], s_dims[1]])
                        features_sum_lst.append(haar_vertical_half(intg_imgs, (h, w), s_dims))
                    if s_dims[0] % 3 == 0 and h + h_inc > h:
                        features_lst.append([h, w, 3, 1, s_dims[0], s_dims[1]])
                        features_sum_lst.append(haar_burger_proper(intg_imgs, (h, w), s_dims))
                    if s_dims[1] % 3 == 0 and w + w_inc > w:
                        features_lst.append([h, w, 1, 3, s_dims[0], s_dims[1]])
                        features_sum_lst.append(haar_burger_side(intg_imgs, (h, w), s_dims))
                    if s_dims[0] % 2 == s_dims[1] % 2 == 0:
                        features_lst.append([h, w, 2, 2, s_dims[0], s_dims[1]])
                        features_sum_lst.append(haar_chess(intg_imgs, (h, w), s_dims))
                    w_inc += 1
                h_inc += 1

    # Save the possible features and corresponding sums (feature_df rows -> unsorted_df columns)
    feature_df_columns = ["Start_h", "Start_w", "Feature_h", "Feature_w", "Dims_h", "Dims_w"]
    features_df = pd.DataFrame(features_lst, columns=feature_df_columns, dtype=np.uint8)
    unsorted_df = pd.DataFrame(features_sum_lst, dtype=np.int16)
    return features_df, unsorted_df.transpose()


def paths_to_integral_images(data_path, extension):
    """
    Calculate integral images from images in a directory

    :param data_path: path where images are located         (string)
    :param extension: extension of image file               (string)
    :return: integral images                                (list)
    """
    paths = sorted(Path(data_path).glob("**/*." + extension))
    intg_imgs = []
    for p in paths:
        intg_img = to_integral_image(cv2.imread(str(p)))
        intg_imgs.append(intg_img)
    return intg_imgs


def sum_single_feature(weak_classifier, intg_img):
    """
    Calculate a weak classifier feature sum for evaluation

    :param weak_classifier: weak classifier                 (object)
    :param intg_img: integral image                         (np.ndarray, int32)
    :param haar_start: feature start coordinates            (tuple)
    :param haar_feature: feature base dimensions            (tuple)
    :param haar_dims: feature dimensions                    (tuple)
    :return: calculated sum                                 (int)
    """
    calc_sum = None
    haar_start = weak_classifier.feature_info["Start_h"], weak_classifier.feature_info["Start_w"]
    haar_dims = weak_classifier.feature_info["Dims_h"], weak_classifier.feature_info["Dims_w"]
    haar_feature = weak_classifier.feature_info["Feature_h"], weak_classifier.feature_info["Feature_w"]
    if haar_feature == (2, 1):
        calc_sum = haar_horizontal_half([intg_img], haar_start, haar_dims)
    elif haar_feature == (1, 2):
        calc_sum = haar_vertical_half([intg_img], haar_start, haar_dims)
    elif haar_feature == (3, 1):
        calc_sum = haar_burger_proper([intg_img], haar_start, haar_dims)
    elif haar_feature == (1, 3):
        calc_sum = haar_burger_side([intg_img], haar_start, haar_dims)
    elif haar_feature == (2, 2):
        calc_sum = haar_chess([intg_img], haar_start, haar_dims)
    return calc_sum


class HaarLikeFeatures:
    def __init__(self, positive_set, negative_set, extension, out_path):
        """
        Initialize dataframes and weights used for training weak classifiers in a monolithic or cascade classifier

        NOTE: algorithm geared for training images with same dimensions and file type

        :param positive_set: path to directory with training positive data      (str)
        :param negative_set: path to directory with training negative data      (str)
        :param extension: extension of training files - homogeneous             (str)
        :param out_path: directory to save associated files                     (str)
        """
        # Check inputted paths for images of provided extension
        [positives, negatives] = [sorted(Path(data_set).glob("**/*." + extension))
                                  for data_set in [positive_set, negative_set]]
        n_positives, n_negatives = len(positives), len(negatives)
        if n_positives == 0 or n_negatives == 0:
            raise Exception("Empty positive/negative data set: positives: " + str(n_positives) +
                            " negatives: " + str(n_negatives))
        print("Positive set:", n_positives, "\nNegative set:", n_negatives)

        # Calculate initial weights
        [self.face_weights, self.non_face_weights] = self.calculate_weights(n_positives, n_negatives)

        # Set paths for reading/saving dataframes
        outpath = Path(out_path)
        feature_path = outpath / "features_df.pkl"    # all possible features for a given training image (row = feature)
        training_path = outpath / "training_df.pkl"   # sorted feature sums for all training images (column = feature)
        type_path = outpath / "type_df.pkl"           # booleans - (face or nonface) corresponding to the sorted sums
        image_path = outpath / "image_df.pkl"         # image number corresponding to the sorted sums
        if not outpath.exists():
            outpath.mkdir()

        # Read saved data if it exists
        if False not in [p.exists() for p in [feature_path, training_path, type_path, image_path]]:
            self.features = pd.read_pickle(feature_path)
            self.training_df = pd.read_pickle(training_path)
            self.type_df = pd.read_pickle(type_path)
            self.image_df = pd.read_pickle(image_path)
            print("Loaded saved training data")
        else:
            # Calculate integral images and possible features
            [intg_p, intg_n] = self.gather_integral_images(positives, negatives)
            print("Integral images calculated")
            training_data = self.calculate_possible_features(intg_p, intg_n)
            print("Haar-like feature sums calculated")
            self.features, self.training_df, self.type_df, self.image_df = training_data

            # Save possible features and corresponding sum dataframes to pickle
            self.features.to_pickle(str(feature_path))
            self.training_df.to_pickle(str(training_path))
            self.type_df.to_pickle(str(type_path))
            self.image_df.to_pickle((str(image_path)))

    def calculate_possible_features(self, intg_p, intg_n):
        """
        Calculate all possible Haar-like features - i.e. 4x4 image, 5 features --> 136 possible features

        :param intg_p: list of face integral images                                     (list)
        :param intg_n: list of nonface integral images                                  (list)
        :return: dataframes for sums, image type (face/nonface), and image number       (pd.DataFrame)
        """
        all_intg_imgs = intg_p + intg_n
        print("Calculating feature sums...")
        features_df, unsorted_df = calc_haar_feature_sums(all_intg_imgs)
        print("Calculated feature sums")

        # Sort data alongside labels
        p_count, n_count = len(intg_p), len(intg_n)
        o_type_order = pd.DataFrame([True] * p_count + [False] * n_count, dtype=np.bool)
        o_image_order = pd.DataFrame(list(range(0, p_count)) + list(range(0, n_count)), dtype=np.uint16)
        
        # Initialize training dataframes
        training_df, type_df, image_no_df = self.sort_feature_data(0, unsorted_df, o_type_order, o_image_order)
        print("Organizing and sorting training data:")
        
        # Sort sums/corresponding labels and obtain new dataframes for each remaining feature
        for row in range(1, features_df.shape[0]):
            if row % 5000 == 0:
                print(str(row) + " / " + str(features_df.shape[0]))
            trn, typ, imn = self.sort_feature_data(row, unsorted_df, o_type_order, o_image_order)
            training_df[row] = trn
            type_df[row] = typ
            image_no_df[row] = imn
        return features_df, training_df, type_df, image_no_df

    def cascade_remove_negative_features(self, im_no_to_remove):
        """
        Cascade training function, removes data corresponding to image numbers that were detected true negatives

        :param im_no_to_remove: cumulative list of image numbers to remove      (list)
        :returns: only positive feature training data                           (pd.DataFrame)
        """
        # Zero the weights of detected nonface images
        new_weights = self.calculate_weights(len(self.face_weights), len(self.non_face_weights))
        self.face_weights = new_weights[0]
        new_non_face_weight = 1 / (2 * (len(self.non_face_weights) - len(im_no_to_remove)))
        for idx in range(0, len(self.non_face_weights)):
            if idx not in im_no_to_remove:
                self.non_face_weights[idx] = new_non_face_weight
            else:
                self.non_face_weights[idx] = 0

        # Remove the image's data from each training dataframe and resort
        if len(im_no_to_remove) > 0:
            new_num_rows = len(self.face_weights) + (len(self.non_face_weights) - len(im_no_to_remove))
            # Initialize templace dataframes
            new_training_df = pd.DataFrame(dtype=np.int16)
            new_type_df = pd.DataFrame(dtype=np.bool)
            new_imno_df = pd.DataFrame(dtype=np.uint16)
            # Remove discovered image numbers and resort based on sums
            for column in range(0, self.training_df.shape[1]):
                sum_col, type_col, image_col = self.training_df[column], self.type_df[column], self.image_df[column]
                sum_lst, type_lst, image_lst = [], [], []
                for t_sum, lbl, im_no in zip(sum_col, type_col, image_col):
                    if lbl or im_no not in im_no_to_remove:
                        sum_lst.append(t_sum)
                        type_lst.append(lbl)
                        image_lst.append(im_no)
                    if len(sum_lst) == new_num_rows:
                        break
                new_training_df[column] = np.asarray(sum_lst, dtype=np.int16)
                new_type_df[column] = np.asarray(type_lst, dtype=np.bool)
                new_imno_df[column] = np.asarray(image_lst, dtype=np.uint16)
                if column % 5000 == 0:
                    print("Finished modifying", column, "columns out of", self.training_df.shape[1])
            self.training_df, self.type_df, self.image_df = new_training_df, new_type_df, new_imno_df

    @staticmethod
    def calculate_weights(n_positives, n_negatives):
        """
        Calculate the initial weights: 1 / (2 * number of images for label)

        :param n_positives: number of face images                   (int)
        :param n_negatives: number of nonface images                (int)
        :return: weights for positive and negative images           (list)
        """
        weights = []
        for count in [n_positives, n_negatives]:
            weight_scale = [1 for img in range(0, count)]
            weight = 1 / (2 * count)
            weights.append([weight * scale for scale in weight_scale])
        return weights

    @staticmethod
    def gather_integral_images(positives, negatives):
        """
        Read images and convert to integral image

        :param positives: path to face images                   (string)
        :param negatives: path to nonface images                (string)
        :return: integral images for face/nonface sets          (list)
        """
        intg_imgs = []
        for labeled_set in [positives, negatives]:
            imgs = [cv2.imread(str(path)) for path in labeled_set]
            intg_imgs.append([to_integral_image(img) for img in imgs])
        return intg_imgs

    @staticmethod
    def sort_feature_data(column, training_df, type_label, image_no):
        """
        Sort the training data for a given feature sum - each column in training_df contains all training case feature 
        sums for a given feature

        :param column: column to sort in the unsorted sum training dataframe    (pd.DataFrame)
        :param training_df: unsorted feature sums dataframe                     (pd.DataFrame)
        :param type_label: unsorted image label dataframe                       (pd.DataFrame)
        :param image_no: unsorted image number dataframe                        (pd.DataFrame)
        :return: sorted sum, image label/number columns                         (pd.DataFrame)
        """
        # Arrange column elements into tuples - will sort by the training sum
        training_sums = training_df[column]
        type_label = type_label[0]
        image_no = image_no[0]
        unsorted_data = []
        for idx in range(0, training_sums.shape[0]):
            unsorted_data.append((training_sums[idx], type_label[idx], image_no[idx]))

        # Sort tuples and re-extract sorted sums, labels, and numbers into columns
        sorted_data = sorted(unsorted_data)
        sum_data, type_data, image_data = [], [], []
        for dat in sorted_data:
            sum_data.append(dat[0])
            type_data.append(dat[1])
            image_data.append(dat[2])
        data_df = pd.DataFrame(sum_data, columns=[column], dtype=np.int16)
        type_df = pd.DataFrame(type_data, columns=[column], dtype=bool)
        imno_df = pd.DataFrame(image_data, columns=[column], dtype=np.uint16)
        return data_df, type_df, imno_df
