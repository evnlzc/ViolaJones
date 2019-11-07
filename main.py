"""
Personal project to build Viola-Jones facial detection method.
Implemented with Python 3.7.4 on Windows 10
"""

import time
import pickle
from pathlib2 import Path

import HaarLikeFeatures as HaarF
import SlidingWindow as SlideW
import WeakClassifier as WeakC
import TrainCascade as TrainC

"""
Choose classifier type: monolithic or cascade
    - Monolithic: all training data used to train sequential weak classifiers for a single strong classifier
    - Cascade: training and verification sets used to train a cascade of strong classifiers 
"""
classifier_type = "cascade"

# Training data
face_training = "c:/Git/Data/face_training"
non_face_training = "c:/Git/Data/nonface_training"
extension = "pgm"
output_path = "c:/Git/ViolaJones"

# Monolithic parameters
num_weak_classifiers = 200

# Cascade parameters
face_validation = "c:/Git/Data/face_validation"
non_face_validation = "c:/Git/Data/nonface_validation"

# Test set
positive_test_set = "C:/git/Data/face_test"
negative_test_set = "C:/git/Data/nonface_test"


if __name__ == "__main__":

    classifiers = []
    # Generate or load dataframes containing Haar-like feature sums for training
    t0 = time.time()
    hf = HaarF.HaarLikeFeatures(face_training, non_face_training, extension, output_path)
    print("Time to generate/load training data:", time.time() - t0)

    # Generate or load a monolithic or cascade of strong classifier(s)
    if classifier_type == "monolithic":
        t0 = time.time()
        weak_classifier_path = Path(output_path) / "weak_classifiers.pkl"
        if not weak_classifier_path.exists():
            print("\nGenerating monolithic strong classifier:")
            classifiers = WeakC.monolithic_adaboost(hf, num_weak_classifiers,
                                                    save_path=output_path, save_list=True)
            with open(str(weak_classifier_path), 'wb') as wc_f:
                pickle.dump(classifiers, wc_f)
        else:
            with open(str(weak_classifier_path), 'rb') as f:
                classifiers = pickle.load(f)
        print("Time to generate/load monolithic classifier:", time.time() - t0)

    elif classifier_type == "cascade":
        t0 = time.time()
        cascade = TrainC.AdaBoostCascade(hf, output_path, non_face_training, face_validation, non_face_validation,
                                         extension, max_layer_fpr=0.8, min_layer_tpr=0.9, target_fpr=0.5**3)
        classifiers = cascade.cascade

        print("Time to generate/load cascade:", time.time() - t0)

    # Evaluations
    with open('c:/git/violajones/extended_cascade.pkl', 'rb') as f:
        classifiers = pickle.load(f)

    SlideW.evaluate_classifier_performance(classifiers, positive_test_set, negative_test_set, "pgm")
    SlideW.evaluate_images(classifiers, 'c:/git/data/big_faces/', 'jpg', 'c:/git/data/big_face_results')
