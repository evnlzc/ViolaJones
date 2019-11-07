"""
File includes:
    Function for generating a monolithic strong classifier and evaluating weak classifiers

    Class for generating a weak classifier with AdaBoost - weak classifiers combine to form a strong classifier
        - Generates classifier stats: error, threshold, direction of threshold, corresponding feature
        - Saves an alpha value for final weight of weak classifier
        - Requires Haar-like features sums training object

    Caution: small training (< ~100) sets are prone to encountering infinite weak classifier thresholds
             and "perfect splits" (no error - WC completely separates face/nonface data)
"""

import math
import pickle
from pathlib2 import Path

import HaarLikeFeatures as HaarF


def monolithic_adaboost(haar_feature_sums, num_wc, save_path="c:/", save_list=False):
    """
    Generate a monolithic classifier using AdaBoost

    :param haar_feature_sums: feature sums dataframes               (object)
    :param num_wc: number of weak classifiers to have               (int)
    :param save_path: directory where progress is saved             (str)
    :param save_list: save weak classifier list between steps       (bool)
    :return: list of weak classifiers (a strong classifier)         (list)
    """
    save_path = Path(save_path) / "wc_lst_progress.pkl"
    if save_path.exists():
        print("Loading prior progress for monolithic classifier")
        with open(str(save_path), 'rb') as f:
            [wc_lst, weights, start] = pickle.load(f)
            haar_feature_sums.face_weights, haar_feature_sums.non_face_weights = weights
    else:
        wc_lst = []
        start = 0
    for it in range(start, num_wc):
        wc = WeakClassifier(haar_feature_sums)
        # Current weak classifier cannot be generated (prior wc was a perfect split)
        if wc.stop_flag:
            break
        wc_lst.append(wc)
        if save_list:
            with open(save_path, 'wb') as f:
                weights = haar_feature_sums.face_weights, haar_feature_sums.non_face_weights
                progress = [wc_lst, weights, it + 1]
                pickle.dump(progress, f)
        print("\tCalculated weak classifier: " + str(it + 1) + " / " + str(num_wc))
    if save_path.exists():
        save_path.unlink()
    return wc_lst


def evaluate_weak_classifiers(intg_img, wc_lst):
    """
    Get the score for a weak classifier

    :param intg_img: integral image     (nd.array, int32)
    :param wc_lst: weak classifiers     (list)
    :return: alpha value (score)        (float)
    """
    alpha = 0
    for wc in wc_lst:
        calc_sum = HaarF.sum_single_feature(wc, intg_img)
        if wc.direction == "up" and calc_sum > wc.threshold:
            alpha += wc.weight
        if wc.direction == "down" and calc_sum < wc.threshold:
            alpha += wc.weight
    return alpha


class WeakClassifier:
    def __init__(self, hf):
        """
        Weak classifier object contains overall classifier weight, error, threshold, and comparison direction
        Using AdaBoost to find an optimal separation between faces and nonfaces
        Updates the weights of the dataframe used for training

        :param hf: Haar-like feature sums object        (object)
        - hf.features: each row is a feature
        - hf.training_df: each column contains a single feature's sums
        - hf.type_df: each column corresponds to a column in training_df (face or nonface)
        - hf.image_df: each column corresponds to a column in training df (image number)
        """
        self.stop_flag = False

        # Calculate sum of weights and normalize - escape if prior classifier had no error
        hf_fw, hf_nfw = hf.face_weights, hf.non_face_weights
        p_sum, n_sum = sum(hf_fw), sum(hf_nfw)
        normalization = p_sum + n_sum

        if normalization == 0:
            # Typically will occur with small training sets (< ~100 cases)
            print("\tAll samples previously classified correctly")
            print("\tNo more weak classifiers can be generated with current weights\n")
            self.stop_flag = True
        else:
            face_weights = [fw / normalization for fw in hf_fw]
            non_face_weights = [nw / normalization for nw in hf_nfw]

            # Calculate sums of normalized weights
            p_sum, n_sum = sum(face_weights), sum(non_face_weights)

            # For each feature, iterate through the sorted sums to find the threshold with minimal error
            classifier_stats = [float("inf")]
            for row in range(0, hf.training_df.shape[1]):
                training_data = self.extract_feature_column(row, hf.training_df, hf.type_df, hf.image_df)
                classifier_stats = self.generate_classifier_stats(classifier_stats, p_sum, n_sum, face_weights,
                                                                  non_face_weights, training_data)
            self.error, self.threshold, self.direction, row = classifier_stats

            # Get weight of weak classifier and scaled weights for next iteration
            feature_sums, feature_labels, feature_images, row = self.extract_feature_column(row, hf.training_df,
                                                                                            hf.type_df, hf.image_df)
            self.weight, hf_fw, hf_nfw = self.update_weights(feature_sums, feature_labels, feature_images,
                                                             face_weights, non_face_weights)

            # Get position and dimensions where feature is applied
            self.feature_info = hf.features.iloc[row]

        # Update weights for next round
        hf.face_weights, hf.non_face_weights = hf_fw, hf_nfw

    def update_weights(self, feature_sums, feature_labels, feature_images, o_f_weights, o_nf_weights):
        """
        Update the face/nonface weights based on the weak classifier's performance

        :param feature_sums: training data sums                         (pd.DataFrame)
        :param feature_labels: training data types                      (pd.DataFrame)
        :param feature_images: training data image numbers              (pd.DataFrame)
        :param o_f_weights: face weights                                (list)
        :param o_nf_weights: nonface weights                            (list)
        :return: weight of weak classifier, updated weights             (float, lists)
        """
        # classified_idxs will store the image numbers of cases that were properly classified
        classified_p_idx, classified_n_idx = [], []

        # Set evaluation direction
        if self.direction == "up":
            iterative_range1 = range(0, feature_sums.shape[0])
        else:
            iterative_range1 = range(feature_sums.shape[0] - 1, 0, -1)

        # Find images where a proper classification occurred
        flip_label = 0
        for idx in iterative_range1:
            if feature_sums[idx] == self.threshold:
                # Threshold is noninclusive
                if not feature_labels[idx]:
                    classified_n_idx.append(feature_images[idx])
                flip_label = 1
                continue
            if not flip_label and not feature_labels[idx]:
                classified_n_idx.append(feature_images[idx])
            if flip_label and feature_labels[idx]:
                classified_p_idx.append(feature_images[idx])

        # Apply new weights based on weak classifier error and calculate overall weight of weak classifier (alpha)
        beta = self.error / (1 - self.error)
        try:
            alpha = math.log10(1 / beta)
        except (ValueError, ZeroDivisionError):
            # Can't have infinite alpha or beta too close to/at 0, settle for using minimum error weight
            beta = min(min(o_f_weights), min(o_nf_weights))
            alpha = beta / (1 - beta)
            print("\tLayer error too close to 0 - setting layer weight to:", alpha)
            beta = 0
        for idx in classified_p_idx:
            o_f_weights[idx] *= beta
        for idx in classified_n_idx:
            o_nf_weights[idx] *= beta
        return alpha, o_f_weights, o_nf_weights

    @staticmethod
    def extract_feature_column(row, training_df, label_df, im_no_df):
        """
        Extract relevant values for feature (row) being analyzed

        :param row: features dataframe row                          (int)
        :param training_df: training sums dataframe                 (pd.DataFrame)
        :param label_df: training labels dataframe                  (pd.DataFrame)
        :param im_no_df: training image numbers dataframe           (pd.DataFrame)
        :return: columns corresponding to the feature of interest   (pd.DataFrames)
        """
        feature_sums = training_df[row]
        feature_labels = label_df[row]
        feature_images = im_no_df[row]
        return feature_sums, feature_labels, feature_images, row

    @staticmethod
    def generate_classifier_stats(classifier_stats, p_sum, n_sum, fw, nfw, training_data):
        """
        Save the threshold with the minimum weighted error if performance is better than best prior classifier

        :param classifier_stats: classifier to beat (error, threshold, direction, feature row)          (list)
        :param p_sum: sum of face feature weights                                                       (int)
        :param n_sum: sum of nonface feature weights                                                    (int)
        :param fw: list of face weights                                                                 (list)
        :param nfw: list of nonface weights                                                             (list)
        :param training_data: column data for feature being investigated                                (pd.DataFrame)
        :return: classifier stats of best performing classifier                                         (list)
        """
        encountered_f, encountered_nf = 0, 0
        feature_sums, feature_labels, feature_images, row = training_data

        # Calculate the weighted error from setting the threshold for each sum and its respective label and image number
        for img_sum, img_lbl, img_no in zip(feature_sums, feature_labels, feature_images):
            # Track encountered faces and nonfaces in respective variables
            if img_lbl:
                encountered_f = encountered_f + fw[img_no]
            else:
                encountered_nf = encountered_nf + nfw[img_no]

            # faces_above_t - current threshold has more faces above the threshold, vice-versa for faces_below_t
            # Effectively act as the weighted errors per direction of the current threshold
            faces_below_t = encountered_nf + (p_sum - encountered_f)
            faces_above_t = encountered_f + (n_sum - encountered_nf)

            # If the current feature has a lower weighted error than the previous "best" feature
            if faces_above_t < classifier_stats[0] or faces_below_t < classifier_stats[0]:
                threshold = img_sum
                if faces_above_t <= faces_below_t:
                    # Face features have sums greater than the threshold
                    error = faces_above_t
                    direction = "up"
                else:
                    # Face features have sums lower than the threshold
                    error = faces_below_t
                    direction = "down"
                classifier_stats = [error, threshold, direction, row]
        return classifier_stats
