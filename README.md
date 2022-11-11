# EKF_SLAM

Implemented Extended Kalman Filter based monocular SLAM for the Dinosaur 
dataset (http://www.robots.ox.ac.uk/~vgg/data/mview/).

Computed simultaneously the map and the localization of 5 
frames of the dataset selected. Used the following paper as a reference 
(https://docs.google.com/viewer?url=http%3A%2F%2Fwww.doc.ic.ac.uk%2F~ajd%2FPublications%2Fdavison_etal_pami2007.pdf ) 

The output of the algorithm is a point cloud of the map and a set of 3D locations 
associated to the camera center. 
