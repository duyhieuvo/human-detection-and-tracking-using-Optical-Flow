import numpy as np
from pykalman import KalmanFilter

#Function to apply the Kalman filter on the stored trajectory file
def Filter(Measured):
    while True:
        if Measured.shape[0]==0: #If the recorded trajectory is empty, return an empty array
            return np.array([])
        if Measured[0, 0] == -1.:
            Measured = np.delete(Measured, 0, 0)
        else:
            break

    #Determine positions (-1,-1) on the trajectory with missing measurement
    MarkedMeasure = np.ma.masked_less(Measured, 0)


    #Define the state transition matrix A and the transformation matrix H
    Transition_Matrix = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
    Observation_Matrix = [[1, 0, 0, 0], [0, 1, 0, 0]]

    #Define the initial state of the process
    x_i = MarkedMeasure[0, 0]
    y_i = MarkedMeasure[0, 1]
    vx_i = MarkedMeasure[1, 0] - MarkedMeasure[0, 0]
    vy_i = MarkedMeasure[1, 1] - MarkedMeasure[0, 1]
    initstate = [x_i, y_i, vx_i, vy_i]
    initcovariance = 1.0e-3 * np.eye(4) #variance of the initial state
    transistionCov = 1.0e-4 * np.eye(4) #variance of the state transition
    observationCov = 0.5 * np.eye(2) #variance of the measurement

    #Initialize the Kalman filter
    kf = KalmanFilter(transition_matrices=Transition_Matrix,
                      observation_matrices=Observation_Matrix,
                      initial_state_mean=initstate,
                      initial_state_covariance=initcovariance,
                      transition_covariance=transistionCov,
                      observation_covariance=observationCov)

    (filtered_state_means, filtered_state_covariances) = kf.filter(MarkedMeasure)
    output = np.hstack((filtered_state_means[:,0].reshape(-1,1),filtered_state_means[:,1].reshape(-1,1)))
    return output
