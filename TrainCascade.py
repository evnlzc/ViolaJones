"""
File includes:
    Function for adding a strong classifier to the end of a cascade

    Class for generating a cascade of strong classifiers
        - Generate or load existing training dataframes
        - Validate on a separate set of face/nonface images
        - Iteratively remove images from training and validation to simulate the cascade
"""

import pickle
from pathlib2 import Path

import WeakClassifier as WeakC
import HaarLikeFeatures as HaarF


def combine_monolithic_cascade(hf_path):
    """
    Append a strong classifier to the cascade

    :param hf_path: path to cascade and strong classifier   (str)
    :return: cascade with appended classifier               (AdaBoostCascade Object)
    """
    if hf_path[-1] != "/":
        hf_path += "/"
    with open(hf_path + "weak_classifiers.pkl", 'rb') as f:
        monolith = pickle.load(f)
    monolith_threshold = sum([wc.weight for wc in monolith]) / 2
    strong_classifiers = AdaBoostCascade(hf_path, 'rb')
    strong_classifiers.cascade[len(strong_classifiers.cascade) + 1] = [monolith, monolith_threshold]
    with open(hf_path + "extended_cascade.pkl", 'wb') as f:
        pickle.dump(strong_classifiers.cascade, f)
    return strong_classifiers.cascade


class AdaBoostCascade:
    def __init__(self, hf_data, hf_path, neg_set="", pos_val_set="", neg_val_set="", extension="",
                 max_layer_fpr=1, min_layer_tpr=0, target_fpr=1):
        """
        Generate a cascade of classifiers

        :param hf_data: Haar-like features training dataframes                          (object)
        :param hf_path: path to pckl files of Haar-like feature for first cascade       (string)
        :param neg_set: path to nonface training data                                   (string)
        :param pos_val_set: path to face validation set                                 (string)
        :param neg_val_set: path to nonface validation set                              (string)
        :param extension: extension of training data                                    (string)
        :param max_layer_fpr: maximum allowable false positive rate per layer           (float)
        :param min_layer_tpr: minimum allowable true positive rate per layer            (float)
        :param target_fpr: desired cumulative product of layer FPRs                     (float)
        """
        cascade_path = Path(hf_path) / "cascade.pkl"
        extended_cascade_path = Path(hf_path) / "extended_cascade.pkl"
        # Load existing classifier or generate/continue generating cascade
        if cascade_path.exists():
            with open(str(cascade_path), 'rb') as f:
                self.cascade = pickle.load(f)
        elif extended_cascade_path.exists():
            with open(str(extended_cascade_path), 'rb') as f:
                self.cascade = pickle.load(f)
        else:
            self.progress_path = Path(hf_path) / "cascade_progress.pkl"

            # Load progress later if it exists
            if not self.progress_path.exists():
                # Convert positive and negative validation sets to integral images
                self.pos_validation, self.neg_validation, self.cascade_neg_test = [
                    self.set_test_intg_img_with_position(path, extension) for path in
                    [pos_val_set, neg_val_set, neg_set]]
                if 0 in [len(self.pos_validation), len(self.neg_validation), len(self.cascade_neg_test)]:
                    pos_check = "Positive validation set has: " + str(len(self.pos_validation)) + " images\n"
                    neg_check = "Negative validation set has: " + str(len(self.neg_validation)) + " images\n"
                    train_check = "Negative training set has: " + str(len(self.cascade_neg_test)) + " images"
                    raise Exception(pos_check + neg_check + train_check)
                print("Positive validation set:", len(self.pos_validation))
                print("Negative validation set:", len(self.neg_validation))
                # Initialize training dataframes and cascade
                self.hf = hf_data
                self.cascade = {}

            # Train cascade
            self.train_cascade_layer(max_layer_fpr, target_fpr, min_layer_tpr)
            if self.progress_path.exists():
                self.progress_path.unlink()
            with open(str(cascade_path), 'wb') as f:
                pickle.dump(self.cascade, f)

    def train_cascade_layer(self, max_layer_fpr, target_fpr, min_layer_tpr):
        """
        Train a cascade of strong classifiers

        :param max_layer_fpr:
        :param target_fpr:
        :param min_layer_tpr:
        :return:
        """
        wc_lst, pos_removals, neg_removals = [], [], []
        threshold = float('inf')
        if not self.progress_path.exists():
            all_true_negatives = []
            no_cascade = 1
            fpr_product = 1.0
        else:
            with open(str(self.progress_path), 'rb') as f:
                print("Loading cascade progress")
                [no_cascade, fpr_product, all_true_negatives, self.hf, self.cascade,
                 self.pos_validation, self.neg_validation, self.cascade_neg_test] = pickle.load(f)
        # While the product of layer FPRs is above the desired holistic FPR product
        while fpr_product > target_fpr:
            fpr_layer = 1.0
            print("\nTraining layer", no_cascade)

            # While the current layer's FPR is higher than the maximum acceptable FPR
            while fpr_layer > max_layer_fpr:
                # Add a single weak classifier per iteration - weights are updated per iteration
                print("\tStrong classifier might have", len(wc_lst) + 1, "weak classifier(s)")
                wc = WeakC.monolithic_adaboost(self.hf, 1)
                if len(wc) == 0:
                    break
                wc_lst.append(wc[0])

                # Evaluate threshold on positive and negative validation sets
                threshold, pos_removals = self.evaluate_positive_validation_set(wc_lst, min_layer_tpr)
                fpr_layer, neg_removals = self.evaluate_negative_validation(threshold, wc_lst)

            # Remove false negatives and true negatives
            print("Removing", len(pos_removals), "false negatives from the positive validation set")
            for idx in pos_removals:
                self.pos_validation.pop(idx)
            print("Removing", len(neg_removals), "true negatives from the negative validation set")
            for idx in neg_removals:
                self.neg_validation.pop(idx)

            # Update cumulative product and update cascade/training set if needed
            fpr_product *= fpr_layer
            self.cascade[no_cascade] = [wc_lst, threshold]
            if fpr_product <= target_fpr:
                print("Target false positive rate reached - saving strong classifier")
                break
            elif len(self.pos_validation.keys()) == 0 or len(self.neg_validation.keys()) == 0:
                print("No more positive or negative validation images - saving strong classifier")
                break
            else:
                o_len_neg_test = len(self.cascade_neg_test.keys())
                df_imno_to_remove = self.determine_training_nonface_images(threshold, wc_lst)
                # New strong classifier classified negative training set completely correctly
                if len(df_imno_to_remove) == 0:
                    print("No false positives from nonface training set")
                    break
                # Reset weights if no nonface training images are to be removed, remove true negatives otherwise
                all_true_negatives += [df_imno_to_remove]
                self.hf.cascade_remove_negative_features(all_true_negatives)
                no_cascade += 1
                wc_lst = []

                print("Saving cascade layer")
                with open(str(self.progress_path), 'wb') as f:
                    progress = [no_cascade, fpr_product, all_true_negatives, self.hf, self.cascade,
                                self.pos_validation, self.neg_validation, self.cascade_neg_test]
                    pickle.dump(progress, f)

    def evaluate_positive_validation_set(self, wc_lst, min_layer_tpr):
        """
        Mock evaluation of cascade - iteratively removing false negatives:
            - Evaluate an inital TPR with the default threshold and collect false negatives' thresholds
            - Decrease threshold to meet minimum acceptable layer TPR if needed
            - Identify which images are false positives with the determined threshold
                - If the threshold's FPR is acceptable - false negatives will be removed from subsequent
                  evaluations to replicate a cascade

        :param wc_lst: weak classifiers accumulated for layer                                           (list)
        :param min_layer_tpr: minimum acceptable true positive rate                                     (float)
        :return: threshold that meets minimum acceptable TPR and false negatives to remove              (float, list)
        """
        pos_val_len = len(self.pos_validation.keys())
        threshold = sum([wc.weight for wc in wc_lst]) / 2
        tpr_counter, failed_alphas, idx_removals = self.evaluate_cascade_layer(self.pos_validation, threshold, wc_lst)
        layer_tpr = tpr_counter / pos_val_len
        print("\tInitial layer TPR:", tpr_counter, "/", pos_val_len, "=", layer_tpr)

        # If TPR is lower than desired, lower the threshold to meet the minimum
        if tpr_counter / pos_val_len < min_layer_tpr:
            min_pos_hits = round(pos_val_len * min_layer_tpr)
            num_to_include = min_pos_hits - tpr_counter
            threshold = failed_alphas[-1 * num_to_include]
            idx_removals = idx_removals[0:-1*num_to_include]

        return threshold, idx_removals

    def evaluate_negative_validation(self, threshold, wc_lst):
        """
        Evaluate threshold on negative validation set to get false positive of layer and extract true negatives

        :param threshold: threshold to test against                             (float)
        :param wc_lst: strong classifier                                        (list)
        :return: false positive rate and true negatives to remove               (float)
        """
        neg_val_len = len(self.neg_validation)
        fpr_counter, _, idx_removals = self.evaluate_cascade_layer(self.neg_validation, threshold, wc_lst)
        layer_fpr = fpr_counter / neg_val_len
        print("\tLayer FPR:", fpr_counter, "/", neg_val_len, "=", layer_fpr, "\n")
        return layer_fpr, idx_removals

    def determine_training_nonface_images(self, threshold, wc_lst):
        """
        Evaluate threshold on negative validation set to get false positive of layer and extract true negatives

        :param threshold: threshold to test against                             (float)
        :param wc_lst: strong classifier                                        (list)
        :return: false positive rate                                            (float)
        """
        _, _, idx_removals = self.evaluate_cascade_layer(self.cascade_neg_test, threshold, wc_lst)
        print("Removing", len(idx_removals), "nonface images from training set\n")
        for idx in idx_removals:
            self.cascade_neg_test.pop(idx)
        df_imno_to_remove = list(self.cascade_neg_test.keys())
        print(len(df_imno_to_remove), "images to make up nonface training set")
        return df_imno_to_remove

    @staticmethod
    def set_test_intg_img_with_position(img_set, extension):
        """
        Initialize image test set for determining subsequent nonface training data:
            - Included images are associated with their original image numbers
            - Correlations are for expedited removal from training dataframes and mock evaluation of cascade

        :param img_set: path to nonface training set                                (string)
        :param extension: image file extension                                      (string)
        :return: integral image and corresponding reference image number            (dict)
        """
        test_dict = {}
        neg_test_intg_imgs = HaarF.paths_to_integral_images(img_set, extension)
        for idx in range(0, len(neg_test_intg_imgs)):
            test_dict[idx] = neg_test_intg_imgs[idx]
        return test_dict

    @staticmethod
    def evaluate_cascade_layer(validation_intg_imgs, threshold, classifiers):
        """
        Evaluate how many face detections occur with the layer's weak classifiers

        :param validation_intg_imgs: validation integral images                                   (dict)
        :param validation_idxs: validation image indexes                                          (list)
        :param threshold: threshold to compare against                                            (float)
        :param classifiers: strong classifier                                                     (list)
        :return: number of positives, thresholds of negatives                                     (int, list)
        """
        low_alphas = []
        counter = 0
        # If the cumulative threshold is insufficient, the corresponding image's threshold and index are saved
        for im_no, intg_img in validation_intg_imgs.items():
            cumulative_alpha = WeakC.evaluate_weak_classifiers(intg_img, classifiers)
            if cumulative_alpha >= threshold:
                counter += 1
            else:
                low_alphas.append((cumulative_alpha, im_no))
        low_alphas = sorted(low_alphas)
        idx_removals = [pair[1] for pair in low_alphas]
        low_alphas = [pair[0] for pair in low_alphas]
        return counter, low_alphas, idx_removals
