import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

from camera import Camera
import structure
import processor
import features

import numpy as np
 
np.set_printoptions(precision=2,suppress=True)

def getBMatrix(yaw, deltak):
    B = np.array([  [np.cos(yaw)*deltak, 0],
                                    [np.sin(yaw)*deltak, 0],
                                    [0, deltak]])
    return B
 
def ekfilter(z_k_observation_vector, stateEstimate_k_minus_1, 
        Umatrix_k_minus_1, P_k_minus_1, dk):
    state_estimate_k = Amatrix_k_minus_1 @ (
            stateEstimate_k_minus_1) + (
            getBMatrix(stateEstimate_k_minus_1[2],dk)) @ (
            Umatrix_k_minus_1) + (
            ProcessNoise_k_minus_1)
             
    print(f'State Estimate Before EKF={state_estimate_k}')

    P_k = Amatrix_k_minus_1 @ P_k_minus_1 @ Amatrix_k_minus_1.T + (
            Qmatrix_k)

    measurement_residual_y_k = z_k_observation_vector - (
            (Hmatrix_k @ state_estimate_k) + (
            SensorNoise_w_k))
 
    print(f'Observation={z_k_observation_vector}')

    S_k = Hmatrix_k @ P_k @ Hmatrix_k.T + Rmatrix_k

    K_k = P_k @ Hmatrix_k.T @ np.linalg.pinv(S_k)

    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)

    P_k = P_k - (K_k @ Hmatrix_k @ P_k)

    print(f'State Estimate After EKF={state_estimate_k}')

    return state_estimate_k, P_k

def dinowadham(i1,i2):
    image1 = cv2.imread(i1)
    image2 = cv2.imread(i2)
    pts1, pts2 = features.find_correspondence_points(image1, image2)
    pts1 = processor.cart2hom(pts1)
    pts2 = processor.cart2hom(pts2)

    height, width, ch = image1.shape
    intrinsic = np.array([  # for wadham images the focal length is set as 36mm
        [3600, 0, width / 2],
        [0, 3600, height / 2],
        [0, 0, 1]])

    return pts1, pts2, intrinsic

def tripts(p1,p2,intrin):
    pts1n = np.dot(np.linalg.inv(intrin), p1)
    pts2n = np.dot(np.linalg.inv(intrin), p2)
    E = structure.compute_essential_normalized(pts1n, pts2n)
    E = -E / E[0][1]
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2s = structure.compute_P_from_essential(E)

    ind = -1
    for i, P2 in enumerate(P2s):
        d1 = structure.reconstruct_one_point(
            pts1n[:, 0], pts2n[:, 0], P1, P2)

        P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_homogenous[:3, :4], d1)

        if d1[2] > 0 and d2[2] > 0:
            ind = i

    P2 = np.linalg.inv(np.vstack([P2s[ind], [0, 0, 0, 1]]))[:3, :4]
    tripts3d = structure.reconstruct_points(pts1n, pts2n, P1, P2)
    return tripts3d, P2, E

Amatrix_k_minus_1 = np.array([[1.0,  0,   0],
                        [  0,1.0,   0],
                        [  0,  0, 1.0]])

ProcessNoise_k_minus_1 = np.array([0.005,0.01,0.005])

Qmatrix_k = np.array([[1.0,   0,   0],
                 [  0, 1.0,   0],
                 [  0,   0, 1.0]])

Hmatrix_k = np.array([[1.0,  0,   0],
                 [  0,1.0,   0],
                 [  0,  0, 1.0]])

Rmatrix_k = np.array([[1.0,   0,    0],
                  [  0, 1.0,    0],
                  [  0,    0, 1.0]])  

SensorNoise_w_k = np.array([0.08,0.06,0.05])

image1 = 'images/001.ppm'
image2 = 'images/002.ppm'
image3 = 'images/003.ppm'
image4 = 'images/004.ppm'
image5 = 'images/005.ppm'

pts1, pts2, intrinsic = dinowadham(image1,image2)
pts3, pts4, intrinsic1 = dinowadham(image2,image3)
pts5, pts6, intrinsic2 = dinowadham(image3,image4)
pts7, pts8, intrinsic3 = dinowadham(image4,image5)


t1,prev_1,E1=tripts(pts1,pts2,intrinsic)
t2,prev_2,E2=tripts(pts3,pts4,intrinsic1)
t3,prev_3,E3=tripts(pts5,pts6,intrinsic2)
t4,prev_4,E4=tripts(pts7,pts8,intrinsic3)

origin = [[0],[0],[0]]

r1,r2,translation1=cv2.decomposeEssentialMat(E1)
r3,r4,translation2=cv2.decomposeEssentialMat(E2)
r5,r6,translation3=cv2.decomposeEssentialMat(E3)
r7,r8,translation4=cv2.decomposeEssentialMat(E4)

newCam1= origin+translation1
newCam1=np.matmul(r2,newCam1)
newCam2= origin+translation2
newCam2=np.matmul(r4,newCam2)
newCam3= origin+translation3
newCam3=np.matmul(r6,newCam3)
newCam4= origin+translation4
newCam4=np.matmul(r8,newCam4)

print(newCam1[0][0])

def main():
    k = 1
    new_point=[]

    dk = 1

    z_k = np.array([[newCam1[0][0], newCam1[1][0], newCam1[2][0]], # k=1
                    [newCam2[0][0], newCam2[1][0], newCam2[2][0]], # k=2
                    [newCam3[0][0], newCam3[1][0], newCam3[2][0]],# k=3
                    [newCam4[0][0], newCam4[1][0], newCam4[2][0]]])# k=5

    stateEstimate_k_minus_1 = np.array([0.0,0.0,0.0])

    Umatrix_k_minus_1 = np.array([0.1,0.0])

    P_k_minus_1 = np.array([[0.1,  0,   0],
                             [  0,0.1,   0],
                             [  0,  0, 0.1]])

    for k, obs_vector_z_k in enumerate(z_k,start=1):

        print(f'Timestep k={k}')  

        optimal_state_estimate_k, covariance_estimate_k = ekfilter(
            obs_vector_z_k, # Most recent sensor measurement
            stateEstimate_k_minus_1, # Our most recent estimate of the state
            Umatrix_k_minus_1, # Our most recent control input U_k-1
            P_k_minus_1, # Our most recent state covariance matrix
            dk) # Time interval
         
        stateEstimate_k_minus_1 = optimal_state_estimate_k
        P_k_minus_1 = covariance_estimate_k

        new_point.append(optimal_state_estimate_k) 
        print()

    print(new_point)

    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=10)
    ax = fig.gca(projection='3d')
    ax.plot(t1[0], t1[1], t1[2], 'b.')
    ax.plot(t2[0], t2[1], t2[2], 'r.')
    ax.plot(t3[0], t3[1], t3[2], 'g.')
    ax.plot(t4[0], t4[1], t4[2], 'y.')
    ax.plot(0, 0, 0, 'k.', marker="1",markersize=9)

    ax.plot(new_point[0][0], new_point[0][1], new_point[0][2], 'k.', marker="1",markersize=15)
    ax.plot(new_point[1][0], new_point[1][1], new_point[1][2], 'k.', marker="1",markersize=15)
    ax.plot(new_point[2][0], new_point[2][1], new_point[2][2], 'k.', marker="1",markersize=15)
    ax.plot(new_point[3][0], new_point[3][1], new_point[3][2], 'k.', marker="1",markersize=15)

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    plt.show()
 
# Program starts running here with the main method  
main()


