# MSc-Thesis-Project
Python - Machine Learning Models


Business Problem

The main goal of this research is to estimate the coordinates of a location in a under
LOS (Line of Sight) and NLOS (Non-Line of Sight) conditions and to minimise the estimation
error by training deep neural network. The proposed approach based on deep
learning is benchmarked against a state-of-the-art algorithm developed by engineers.

15 locations and signals were received at 5 different locations (referred to Nanobees)
which are distributed throughout the environment. The rover moves to each location
and transmits signals for 10-15 minutes. These signal's data are collected for each
location and nanobee. Some of these location points are in direct line of sight (LOS)
while some are in non-line of light (NLOS). One of the main challenges is to estimate
the position of these NLOS points. This is non-trivial as the data are noisy due to the
presence of multi-path signals.

Using some hardware apparatus, the accuracy thus calculated was found 2 meters
approximately with 90% confidence interval in 2-dimensional space.
The main objective for us was to bring this error down to less than 1 meter.

Results

I have developed several machine learning algorithms for location estimation and compared and contrasted
my results with respect to existing benchmark.

One of the key achievements of my work is a novel triangulation model to predict
location coordinates using the TOAs. The model is a multi-layer perceptron and is
trained using simulated coordinates. The location estimation error in this model is less
than 1 meter with a 90% confidence interval. 

Overal accuracy of the final model was 1.2m with 90% confidense interval.
