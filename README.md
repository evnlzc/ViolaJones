# Viola Jones Face Detection

#### A personal ML/CV project implementing Viola and Jones' algorithm for face detection using Python
* https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf


### Requirements
* Built using Python 3.7.4 and additional modules (see requirments.txt)
* Built and tested on Windows 10

### Implementation Background
1. Generating Haar-like feature sums
    * Normalize and equalize image intensities to reduce deviations due to lighting
    * Calculate integral image: allows for rapid comparison of regional pixel intensities
    * Accumulate Haar-like feature sums for 5 features applied at allowable scales within the image
    * Example: 24 x 24 image will have 162,336 possible features
    <br/>

2. Using AdaBoost to train weak classifiers
    * Trying to find the best feature and a threshold that best separates training faces and nonfaces
    * Training images are weighted: goal of weak classifier is to minimize weighted error
    * With each iteration, a Haar-like feature and corresponding dimensions, position, and sum threshold are obtained
    * Correctly classified images have their weights decreased so future classifiers are geared to correct prior oversights
    * Errors of weak classifiers are used to generate intermediary thresholds
    * Weak classifiers and their individual thresholds are combined to make a strong classifier
    <br/>
    
3. Build a cascade of strong classifiers
    * A single strong classifier, a "monolith", is not efficient for searching a large image for faces
    * With a cascade, sub-images without obvious facial features are quickly disregarded so other regions are analyzed
    * Train using sets of face and nonface training/validation images
    * A cascade aggregates strong classifiers with each "layer" acting as a filter
    * Each layer is trained based on a maximum allowable false positive rate and minimum false negative rate
    * The TPR and FPR are obtained by implementing the current cascade on the validation sets
    * Once an allowable FPR and TPR are obtained, the cascade is applied to the training data
    * Training the next layer uses all the original face images and any misclassified nonface images
    * Example: A 5 layer cascade <br/>
               Each layer has a max FPR of 0.5 and a min TPR of 0.99 <br/>
               Overall classifier will have a FPR of 0.5 ^ 5 and a TPR of 0.99 ^ 5
    <br/>
    
4. Analyze images using a sliding window and non-maxima suppression
    * Resize each subwindow obtained to the size of the training images and calculate an integral image
    * Apply either a monolithic classifier or cascade to the integral image and save any positive detections and respective thresholds
    * Use non-maxima suppression of overlapping subwindows based on the classifier threshold
    
### Usage
* Check main.py for examples of training and application of sliding window
* Training and test images were obtained from http://www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz
* Test images for sliding window were obtained from http://vis-www.cs.umass.edu/lfw/
* (Optional) flipped images horizontally to augment data quantity (i.e. for testing)

### Results
* Trained using 2,000 face images and 2,000 nonface images, cascade used 700 and 3,000 respective validation images <br/>

* Monolithic classifier: 200 weak classifiers
    * Test on direct front face images: 2348 / 2429 (TPR of 0.97)
    * Test on challenging face images (tilted/lighting): 102 / 472 (TPR: 0.22)
    * Test on nonface images: 84 / 28121 (FPR: 0.003)
    * Took ~9 hours to calculate sums and train
    * Tested on ~30,000 images in 537 seconds <br/>

* Cascade: Layer minimum TPR (1.0), Layer maximum FPR (0.5), Target FPR (0.5 ^ 10)
    * Test on direct front face images: 2308 / 2429 (TPR of 0.95)
    * Test on challenging face images: 121 / 472 (TPR: 0.26)
    * Test on nonface images: 188 / 28121 (FPR: 0.007)
    * Took ~11 hours to calculate sums and train
    * About 144 seconds to test on all images <br/>
    
* Training the cascade did not reach the desired 10 layers (all nonface training images were depleted)
    * 7 layers: <br/>
      Layer 1:  5 weak classifiers 0.472 (FPR) <br/>
      Layer 2: 15 weak classifiers 0.411 <br/>
      Layer 3: 23 weak classifiers 0.469 <br/>
      Layer 4: 29 weak classifiers 0.445 <br/>
      Layer 5: 42 weak classifiers 0.487 <br/>
      Layer 6: 79 weak classifiers 0.436 <br/>
      Layer 7: 70 weak classifiers 0.458 <br/>
      
* Failure to evaluate the full cascade led to appending the monolithic classifier to the cascade (out of curiosity)
    * Calling this an "extended classifier"
    * Test on direct front face images: 2284 / 2429 (TPR of 0.94)
    * Test on challenging face images: 83 / 472 (TPR: 0.18)
    * Test on nonface images: 57 / 28121 (FPR: 0.002)
      
* Sliding window samples evaluated with the extended classifier: <br/>
![alt text](https://github.com/evnlzc/ViolaJones/blob/master/images/conan.jpg) &nbsp;
![alt text](https://github.com/evnlzc/ViolaJones/blob/master/images/conan.png) <br/>
![alt text](https://github.com/evnlzc/ViolaJones/blob/master/images/alias.jpg) &nbsp;
![alt text](https://github.com/evnlzc/ViolaJones/blob/master/images/alias_result.jpg) <br/>
![alt text](https://github.com/evnlzc/ViolaJones/blob/master/images/neo.jpg) &nbsp;
![alt text](https://github.com/evnlzc/ViolaJones/blob/master/images/neo.png) <br/>
![alt text](https://github.com/evnlzc/ViolaJones/blob/master/images/dim.JPG) &nbsp;
![alt text](https://github.com/evnlzc/ViolaJones/blob/master/images/dim_result.jpg) <br/>
![alt text](https://github.com/evnlzc/ViolaJones/blob/master/images/volc.jpg) &nbsp;
![alt text](https://github.com/evnlzc/ViolaJones/blob/master/images/volc.png) <br/>


### Conclusions
* Approach was valid but overfitted to direct frontal faces
    * Subset of the CBCL test data proved very challenging for both classifiers ( < 0.3 TPR)
    * Evalulate training images more thoroughly, add in some challenging cases to training/validation sets
    * Lowering the minimum TPR per layer may yield a looser fit (0.99 ^ 10 = ~0.9)
    * Viola and Jones implemented methods to shuffle and inject training/validation data (could help with finishing cascade)
    * Early training implementations showed more negative samples decrease the FPR (use above method)

* Sliding window is quite slow regardless of the cascade ( > 10 minutes for images larger than 1080 x 720)
    * Ultimately scaled images down to detector size rather than vice-versa - getting highly inaccurate classifications
    * Works quite well with resized images - possibly rescale upwards with bounded boxes
    * Investigate the possibility of more image processing (blurring to remove noise, sharpen edges)
    * More investigation needed into false positives of seemingly homogenous regions
      
      
      
      
      
      
      
      
      
      
      
      
      

    
