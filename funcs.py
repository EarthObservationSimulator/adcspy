import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.sparse.linalg import expm
from scipy.linalg import block_diag
from scipy import integrate
from mpl_toolkits import mplot3d


""" FUNCTIONS """
    
def bodydyn(w,M,Parameters):
    """ Euler's equation for rigid body dynamics """
    I = Parameters['Body']['I']
    H    = I @ w
    wdot = np.linalg.inv(I) @ ( M - np.cross(w,H) )
    return wdot


def quatdyn(q,w,Parameters):
    """ Quaternion kinematics """
    wx = w[0]
    wy = w[1]
    wz = w[2]
    OMEGA = np.array( [ [0,+wz,-wy,+wx] , [-wz,0,+wx,+wy] , [+wy,-wx,0,+wz] , [-wx,-wy,-wz,0] ] )
    qdot = 0.5 * OMEGA @ q
    return qdot


def rotordyn(wr,w,Mc,Parameters):
    """ Rotor dynamics """
    Irmat = Parameters['Actuators']['Rotors']['Irmat']
    A_rot = Parameters['Actuators']['Rotors']['A_rot']
    wrdot = np.linalg.inv(Irmat) @ ( np.linalg.pinv(A_rot) @ ( -Mc - np.cross( w , A_rot @ Irmat @ wr ) )  )
    return wrdot


def sensorsuite(w,q,Parameters):
    """ Measurements by onboard sensors """
    w_actual = w
    q_actual = q
    Q_actual = utils.quat_to_rot(q_actual)
    bias     = np.array([0.5,-0.1,0.2]) #np.zeros(3)
    
    # Measurement of reference directions by Sun Sensor
    rNSunSensor    = Parameters['Sensors']['SunSensor']['rNSunSensor']
    COV_suns       = Parameters['Sensors']['SunSensor']['COV_suns']
    rBSunSensor    = Q_actual.T @ rNSunSensor + np.linalg.cholesky(COV_suns) @ np.random.randn(3)     # rotation from inertial to body!!!!
    rBSunSensor    = rBSunSensor/np.linalg.norm(rBSunSensor)

    # Measurement of reference directions by Magnetometer
    rNMagnetometer = Parameters['Sensors']['Magnetometer']['rNMagnetometer']
    COV_magn       = Parameters['Sensors']['Magnetometer']['COV_magn']
    rBMagnetometer = Q_actual.T @ rNMagnetometer + np.linalg.cholesky(COV_magn) @ np.random.randn(3)  # rotation from inertial to body!!!!
    rBMagnetometer = rBMagnetometer/np.linalg.norm(rBMagnetometer)

    # Measurement of attitude by Startracker
    COV_star     = Parameters['Sensors']['Startracker']['COV_star']
    phi_noise    = np.linalg.cholesky(COV_star) @ np.random.randn(3)   # noise introduced via phi parameterization
    quat_noise   = utils.phi_to_quat(phi_noise)
    quat_noise   = quat_noise/np.linalg.norm(quat_noise)
    qStarTracker = utils.qmult(q_actual,quat_noise)

    # Measurement by Gyroscope
    COV_rnd = Parameters['Sensors']['Gyroscope']['COV_rnd']
    COV_arw = Parameters['Sensors']['Gyroscope']['COV_arw']
    bias    = bias + np.linalg.cholesky(COV_arw) @ np.random.randn(3)
    wGyro   = w_actual + bias + np.linalg.cholesky(COV_rnd) @ np.random.randn(3)

    return rBSunSensor, rBMagnetometer, qStarTracker, wGyro, bias


def prediction(xk,w,dt):
    """ Prediction step for MEKF """
    q = xk[0:4]
    b = xk[4:7]
    
    biaserror = np.linalg.norm(w-b)
    if biaserror != 0:
        theta = np.linalg.norm(w-b) * dt
        r = (w-b)/np.linalg.norm(w-b)
    else:
        theta = 0
        r = (w-b)
        
    xn_qv = r*np.sin(theta/2)
    xn_qs = np.cos(theta/2)
    xn_q = utils.qmult( q, np.concatenate((xn_qv,xn_qs[np.newaxis]),axis=0) )
    xn_b = b
    xn = np.concatenate((xn_q,xn_b),axis=0)
    
    R = expm( utils.hat(-w+b)*dt )
    
    A = np.block([[R , -dt*np.identity(3)], [np.zeros((3,3)) , np.identity(3)]])
    return xn, A


def measurement(q,rN):
    """ Measurement step for MEKF """
    Q  = utils.quat_to_rot(q)
    rB = np.kron(np.eye(rN.shape[0]), Q.T) @ rN.reshape(rN.size,1)
    rBarray = rB.reshape(rN.shape[0],3)
    
    C = np.block( [np.identity(3) , np.zeros((3,3)) ])
    
    for ii in np.arange(0,rN.shape[0]):
       C = np.vstack((C, np.block( [utils.hat(rBarray[ii,:]) , np.zeros((3,3))]) ))
    
    y = np.concatenate((q,rB.flatten()),axis=0)
   
    return y,C


def mekf(x0,P0,whist,yhist,Parameters):
    """ MEKF algorithm (quaternion and body rate estimation) """
    
    dt       = Parameters['Simulation']['dt']
    rN1      = Parameters['Sensors']['SunSensor']['rNSunSensor']
    rN2      = Parameters['Sensors']['Magnetometer']['rNMagnetometer']
    COV_rnd  = Parameters['Sensors']['Gyroscope']['COV_rnd']
    COV_arw  = Parameters['Sensors']['Gyroscope']['COV_arw']
    COV_star = Parameters['Sensors']['Startracker']['COV_star']
    COV_suns = Parameters['Sensors']['SunSensor']['COV_suns']
    COV_magn = Parameters['Sensors']['Magnetometer']['COV_magn']
    
    rN = np.vstack((rN1[np.newaxis,:],rN2[np.newaxis,:]))
    
    W = block_diag(COV_rnd,COV_arw)
    V = block_diag(COV_star,COV_suns,COV_magn)
    
    
    # Memory allocation
    if yhist.ndim == 1:
        max_iter = 1
        xhist = np.zeros((2,7))
        Phist = np.zeros((2,6,6))
        whist = np.reshape(whist,(1,-1))
        yhist = np.reshape(yhist,(1,-1))
    else: 
        max_iter = yhist.shape[0]-1
        xhist = np.zeros((yhist.shape[0],7))
        Phist = np.zeros((yhist.shape[0],6,6))
          
    xhist[0,:] = x0
    Phist[0,:,:] = P0
    
    for k in np.arange(0,max_iter):
        x_p, A = prediction(xhist[k,:],whist[k,:],dt)
        P_p = A @ Phist[k,:,:] @ A.T + W
        y_p, C = measurement(x_p[0:4],rN)
        
        # Innovation
        z_q = utils.quat_to_phi( utils.qmult( utils.qconj( x_p[0:4] ), yhist[k,0:4]) ) 
        z_r =  yhist[k,4:] - y_p[4:]
        z = np.concatenate((z_q,z_r),axis=0)
        S = C @ P_p @ C.T + V
        
        # Kalman Gain
        L = P_p @ C.T @ np.linalg.inv(S)
        
        # Update
        dx = L @ z
        phi = dx[0:3]
        dq = utils.phi_to_quat(phi)
        dq = dq/np.linalg.norm(dq)
        
        # Quaternion update
        xhist[k+1,0:4] = utils.qmult( x_p[0:4] , dq )
        # Bias update
        xhist[k+1,4:7] = x_p[4:7] + dx[3:6]
        # Covariance update
        tmp = (np.identity(6) - L @ C)
        Phist[k+1,:,:] = tmp @ P_p @ tmp.T + L @ V @ L.T      
          
    return xhist, Phist


def controllaw(k,w,q,wr,Parameters):
    """ Control law """
    A_rot = Parameters['Actuators']['Rotors']['A_rot']
    Mc_max = Parameters['Actuators']['Rotors']['Mc_max']
    Kp = Parameters['Controllers']['Kp']
    Kd = Parameters['Controllers']['Kd']
    qdes = Parameters['Controllers']['q_tf']
    
    # Error quaternion
    q_err = utils.qmult( utils.qconj(qdes) , q )
    
    # Body rate error 
    w_des = np.zeros(3)
    w_err = w - w_des
    
    # "PD" control law
    u_c = -Kp*q_err[0:3] - Kd*w_err
    
    # Torque command for each rotor
    tau_c = np.linalg.pinv(A_rot) @ u_c
    sigma = np.amax(np.absolute(tau_c))

    # If torque command larger than max value, saturate command
    if np.absolute(sigma) > Mc_max:
        Mc = Mc_max*A_rot @ tau_c/sigma
    else:
       Mc = u_c
    
    return Mc


def fulldyn(k,x,x_est,P_est,Parameters):
    """ Closed loop control of system (body + rotors) using estimated state feedback """
    Mext   = Parameters['ExternalMoments']['Mext']
    w      = x[0:3]
    qold   = x[3:7]
    wr     = x[7:11]
    q      = qold/np.linalg.norm(qold); 
    qdot   = quatdyn(q,w,Parameters)
    
    rBSunSensor, rBMagnetometer, qStarTracker, wGyro, bias = sensorsuite(w,q,Parameters)
    yhist = np.concatenate((qStarTracker,rBSunSensor,rBMagnetometer),axis=0)
    whist = wGyro

    xhist, Phist = mekf(x_est,P_est,whist,yhist,Parameters)    
    x_est = xhist[-1,:]
    P_est = Phist[-1,:,:]
    
    q_filt = x_est[0:4]    # estimator is q then w. But dynamics is w then q
    w_filt = w    # estimator is q then w. But dynamics is w then q
    Mc     = controllaw(k,w_filt,q,wr,Parameters)
    wrdot  = rotordyn(wr,w,Mc,Parameters)
    wdot   = bodydyn(w,Mext+Mc,Parameters)
    xdot   = np.concatenate((wdot,qdot,wrdot),axis=0)
    return xdot, Mc, Mext, w, w_filt, q, q_filt, wr, wdot, qdot, wrdot, x_est, P_est, rBSunSensor, rBMagnetometer, qStarTracker, wGyro, bias


def simulate(Parameters):
    """ Integration of equations of motion using Forward Euler method """
    rNSunSensor    = Parameters['Sensors']['SunSensor']['rNSunSensor']
    rNMagnetometer = Parameters['Sensors']['Magnetometer']['rNMagnetometer']
    x0             = Parameters['Simulation']['x0']
    dt             = Parameters['Simulation']['dt']
    tSpan          = Parameters['Simulation']['tSpan']
    tSim           = tSpan
    
    A_rot          = Parameters['Actuators']['Rotors']['A_rot']
    Irmat          = Parameters['Actuators']['Rotors']['Irmat']
    
    # Initialize
    xSim = np.zeros((tSim.shape[0],x0.shape[0]))
    McSim = np.zeros((tSim.shape[0],3))
    RotorTorqueSim = np.zeros((tSim.shape[0],4)) 
    MextSim = np.zeros((tSim.shape[0],3))    
    wSim = np.zeros((tSim.shape[0],3))  
    w_filtSim = np.zeros((tSim.shape[0],3))  
    qSim = np.zeros((tSim.shape[0],4))  
    q_filtSim = np.zeros((tSim.shape[0],4)) 
    wrSim = np.zeros((tSim.shape[0],4))  
    wdotSim = np.zeros((tSim.shape[0],3))  
    qdotSim = np.zeros((tSim.shape[0],4))  
    wrdotSim = np.zeros((tSim.shape[0],4))  
        
    x_estSim = np.zeros((tSim.shape[0],7)) 
    P_estSim = np.zeros((tSim.shape[0],6,6)) 
    
    rBSunSensor_act     = np.zeros((tSim.shape[0],3))
    rBMagnetometer_act  = np.zeros((tSim.shape[0],3))   
    rBSunSensor_meas    = np.zeros((tSim.shape[0],3))
    rBMagnetometer_meas = np.zeros((tSim.shape[0],3))
    qStarTracker_meas   = np.zeros((tSim.shape[0],4))
    wGyro_meas          = np.zeros((tSim.shape[0],3))
    bias_act            = np.zeros((tSim.shape[0],3))  
    
    # Initial conditions
    q_t0_est = Parameters['Estimator']['q_t0_est']
    b_t0_est = Parameters['Estimator']['b_t0_est']
    P_t0_est = Parameters['Estimator']['P_t0_est']

    xSim[0,:] = x0 
        
    x_estSim[0,:]   = np.concatenate((q_t0_est,b_t0_est),axis=0)
    P_estSim[0,:,:] = P_t0_est
        
    # Numerical integration
    for k in np.arange(0,tSim.shape[0]-1):    
        dxdt, Mc, Mext, w, w_filt, q, q_filt, wr, wdot, qdot, wrdot, x_est, P_est, rBSunSensorNoisy, rBMagnetometerNoisy, qStarTrackerNoisy, wGyroNoisy, biasNoisy = fulldyn(k,xSim[k,:],x_estSim[k,:],P_estSim[k,:,:],Parameters)
        xSim[k+1,:] = xSim[k,:] + dxdt*dt
        McSim[k,:] = Mc
        MextSim[k,:] = Mext
        wSim[k,:] = w
        w_filtSim[k,:] = w_filt
        qSim[k,:] = q
        q_filtSim[k,:] = q_filt
        wrSim[k,:] = wr
        wdotSim[k,:] = wdot
        qdotSim[k,:] = qdot
        wrdotSim[k,:] = wrdot
        x_estSim[k+1,:] = x_est
        P_estSim[k+1,:,:] = P_est
        
        Q = utils.quat_to_rot(q)
        rBSunSensor_act[k,:]     = Q.T @ rNSunSensor
        rBMagnetometer_act[k,:]  = Q.T @ rNMagnetometer
        

        rBSunSensor_meas[k,:]    = rBSunSensorNoisy
        rBMagnetometer_meas[k,:] = rBMagnetometerNoisy
        
        qStarTracker_meas[k,:]   = qStarTrackerNoisy
        wGyro_meas[k,:]          = wGyroNoisy
        
        bias_act[k,:] = biasNoisy
    
    # Below not necessary, but getting estimation of final states and final derivatives
    kf = int(tSpan[-1]/dt)
    _, Mc_tf, Mext_tf, w_tf, w_filt_tf, q_tf, q_filt_tf, wr_tf, wdot_tf, qdot_tf, wrdot_tf, x_est_tf, P_est_tf, rBSunSensorNoisy_tf, rBMagnetometerNoisy_tf, qStarTrackerNoisy_tf, wGyroNoisy_tf, biasNoisy_tf = fulldyn(kf,xSim[-1,:],x_estSim[-1,:],P_estSim[-1,:,:],Parameters) # calculate signals of interest at the horizon
    McSim[-1,:] = Mc_tf
    MextSim[-1,:] = Mext_tf
    wSim[-1,:] = w_tf
    w_filtSim[-1,:] = w_filt_tf
    qSim[-1,:] = q_tf
    q_filtSim[-1,:] = q_filt_tf
    wrSim[-1,:] = wr_tf
    wdotSim[-1,:] = wdot_tf
    qdotSim[-1,:] = qdot_tf
    wrdotSim[-1,:] = wrdot_tf 
    Q_tf = utils.quat_to_rot(q_tf)
    rBSunSensor_act[-1,:]     = Q_tf.T @ rNSunSensor
    rBMagnetometer_act[-1,:]  = Q_tf.T @ rNMagnetometer

    rBSunSensor_meas[-1,:]    = rBSunSensorNoisy_tf
    rBMagnetometer_meas[-1,:] = rBMagnetometerNoisy_tf

    qStarTracker_meas[-1,:]   = qStarTrackerNoisy_tf
    wGyro_meas[-1,:]          = wGyroNoisy_tf

    bias_act[k,:]             = biasNoisy_tf
    
    RotorTorqueSim[:,:] = (A_rot.T @ McSim.T).T       # torque from each rotor
    
    # Performance metrics
    PowerSim = np.zeros((tSim.shape[0],4))  
    TotalPowerSim = np.zeros((tSim.shape[0]))  
    CumEnergySim = np.zeros((tSim.shape[0],4)) 
    TotalCumEnergySim = np.zeros((tSim.shape[0]))
    PowerSim[:,:]       = abs(np.multiply( RotorTorqueSim ,  wrSim )) 
    TotalPowerSim[:]    = np.sum(PowerSim,axis=1)
    tempCumEnergySim    = integrate.cumtrapz(PowerSim, x=None, dx=dt, axis=0, initial=None)  # check this abs(power)
    CumEnergySim[:,:]   = np.insert(tempCumEnergySim,0,np.zeros((1,4)),axis=0)
    TotalCumEnergySim[:]= np.sum(CumEnergySim,axis=1) 


    eachRotorMaxAbsInstaPower = np.amax(abs(PowerSim),axis=0)
    allRotorsMaxAbsInstaPower = np.amax(eachRotorMaxAbsInstaPower)

    eachRotorTotalEnergy = CumEnergySim[-1,:]
    allRotorsTotaEnergy  = sum(eachRotorTotalEnergy)
    
    eachRotorAbsChangeInRotorMomentum = abs(Irmat @ (wrSim[0,:] - wrSim[-1,:]).T)
    
    allRotorsMaxAbsChangeInRotorMomentum = np.amax(eachRotorAbsChangeInRotorMomentum)
    
    scalarQuantities = {
              'eachRotorMaxAbsInstaPower' : eachRotorMaxAbsInstaPower, 'allRotorsMaxAbsInstaPower' : allRotorsMaxAbsInstaPower,
              'eachRotorTotalEnergy' : eachRotorTotalEnergy, 'allRotorsTotaEnergy' : allRotorsTotaEnergy,
              'eachRotorAbsChangeInRotorMomentum' : eachRotorAbsChangeInRotorMomentum, 'allRotorsMaxAbsChangeInRotorMomentum' : allRotorsMaxAbsChangeInRotorMomentum}
    
    varyingTimeQuantities = {'PowerSim' : PowerSim, 'TotalPowerSim' : TotalPowerSim, 'CumEnergySim' : CumEnergySim, 'TotalCumEnergySim' : TotalCumEnergySim}
    
        
    """ Results dictionary """
    Rigid_Body_States = {'wSim' : wSim, 'qSim' : qSim, 'wdotSim' : wdotSim, 'qdotSim' : qdotSim}
    Rigid_Body_Estimated_States = {'w_filtSim' : w_filtSim, 'q_filtSim' : q_filtSim}
    SunSensor_meas    =  {'rBSunSensor_act' : rBSunSensor_act, 'rBSunSensor_meas' : rBSunSensor_meas}
    Magnetometer_meas =  {'rBMagnetometer_act' : rBMagnetometer_act, 'rBMagnetometer_meas' : rBMagnetometer_meas}
    StarTracker_meas  = {'qStarTracker_meas' : qStarTracker_meas}
    Gyro_meas         = {'wGyro_meas' : wGyro_meas}
    Sensor_Measurements = {'SunSensor_meas' : SunSensor_meas, 'Magnetometer_meas' : Magnetometer_meas, 'StarTracker_meas' : StarTracker_meas, 'Gyro_meas' : Gyro_meas }
    Estimator_States = {'x_estSim' : x_estSim, 'P_estSim' : P_estSim}
    Actuator_States = {'wrSim' : wrSim, 'wrdotSim' : wrdotSim}
    Inputs = {'McSim' : McSim, 'MextSim' : MextSim, 'RotorTorqueSim' : RotorTorqueSim}
    Performance = {'scalarQuantities' : scalarQuantities, 'varyingTimeQuantities' : varyingTimeQuantities}
    Results = {'tSim' : tSim, 'Rigid_Body_States' : Rigid_Body_States, 'Rigid_Body_Estimated_States' : Rigid_Body_Estimated_States, 'Sensor_Measurements' : Sensor_Measurements, 'Actuator_States' : Actuator_States, 'Estimator_States' : Estimator_States, 'Inputs' : Inputs, 'Performance' : Performance}
    
    return Results


def plot(Results,Parameters):
    """ Plotting functions """
    
    I = Parameters['Body']['I']
    Ixx = I[0,0]
    A_int = Parameters['Controllers']['A_t0']
    A_des = Parameters['Controllers']['A_tf']
    
    tSim                = Results['tSim']
    wSim                = Results['Rigid_Body_States']['wSim']
    qSim                = Results['Rigid_Body_States']['qSim']
    wdotSim             = Results['Rigid_Body_States']['wdotSim']
    qdotSim             = Results['Rigid_Body_States']['qdotSim']
    w_filtSim           = Results['Rigid_Body_Estimated_States']['w_filtSim']
    q_filtSim           = Results['Rigid_Body_Estimated_States']['q_filtSim']
    rBSunSensor_act     = Results['Sensor_Measurements']['SunSensor_meas']['rBSunSensor_act']
    rBSunSensor_meas    = Results['Sensor_Measurements']['SunSensor_meas']['rBSunSensor_meas']
    rBMagnetometer_act  = Results['Sensor_Measurements']['Magnetometer_meas']['rBMagnetometer_act']
    rBMagnetometer_meas = Results['Sensor_Measurements']['Magnetometer_meas']['rBMagnetometer_meas']
    qStarTracker_meas   = Results['Sensor_Measurements']['StarTracker_meas']['qStarTracker_meas']
    wGyro_meas          = Results['Sensor_Measurements']['Gyro_meas']['wGyro_meas']
    wrSim               = Results['Actuator_States']['wrSim']
    wrdotSim            = Results['Actuator_States']['wrdotSim']
    McSim               = Results['Inputs']['McSim']
    MextSim             = Results['Inputs']['MextSim']
    RotorTorqueSim      = Results['Inputs']['RotorTorqueSim']
    
    PowerSim            = Results['Performance']['varyingTimeQuantities']['PowerSim']
    TotalPowerSim       = Results['Performance']['varyingTimeQuantities']['TotalPowerSim']
    CumEnergySim        = Results['Performance']['varyingTimeQuantities']['CumEnergySim']
    TotalCumEnergySim   = Results['Performance']['varyingTimeQuantities']['TotalCumEnergySim']
    
    thetadotmax = Parameters['Actuators']['Rotors']['thetadotmax']
    torquemax = Parameters['Actuators']['Rotors']['torquemax']
    powermax = Parameters['Actuators']['Rotors']['powermax']
    energymax = Parameters['Actuators']['Rotors']['energymax']

    
    plt.plot(tSim, qSim[:, 0], 'b', label='$q_1$')
    plt.plot(tSim, qSim[:, 1], 'r', label='$q_2$')
    plt.plot(tSim, qSim[:, 2], 'g', label='$q_3$')
    plt.plot(tSim, qSim[:, 3], 'k', label='$q_4$')
    plt.title('Attitude of Body w.r.t. Inertial (actual)')
    plt.legend(loc='best')
    plt.ylabel('[ ]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()   
    
    plt.plot(tSim, thetadotmax*np.ones(tSim.shape), 'k--', label='$\omega_{r,max}$')
    plt.plot(tSim, -thetadotmax*np.ones(tSim.shape), 'k--')
    plt.plot(tSim, wrSim[:, 0], 'b', label='$\omega_{r_1}$')
    plt.plot(tSim, wrSim[:, 1], 'r', label='$\omega_{r_2}$')
    plt.plot(tSim, wrSim[:, 2], 'g', label='$\omega_{r_3}$')
    plt.plot(tSim, wrSim[:, 3], 'k', label='$\omega_{r_4}$')
    plt.title('Rotor Angular Velocities about axes')
    plt.legend(loc='best')
    plt.ylabel('[rad/s]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()
    
    plt.plot(tSim, torquemax*np.ones(tSim.shape), 'k--', label='$\\tau_{r,max}$')
    plt.plot(tSim, -torquemax*np.ones(tSim.shape), 'k--')
    plt.plot(tSim, RotorTorqueSim[:, 0], 'b', label='$\\tau_{r_1}$')
    plt.plot(tSim, RotorTorqueSim[:, 1], 'r', label='$\\tau_{r_2}$')
    plt.plot(tSim, RotorTorqueSim[:, 2], 'g', label='$\\tau_{r_3}}$')
    plt.plot(tSim, RotorTorqueSim[:, 3], 'k', label='$\\tau_{r_4}$')
    plt.title('Rotor Torques about axes')
    plt.legend(loc='best')
    plt.ylabel('[Nm]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()

    plt.plot(tSim, powermax*np.ones(tSim.shape), 'k--', label='$P_{max}$')
    plt.plot(tSim, TotalPowerSim, 'm', label='$P_{tot}$')
    plt.plot(tSim, PowerSim[:, 0], 'b', label='$P_{r_1}$')
    plt.plot(tSim, PowerSim[:, 1], 'r', label='$P_{r_2}$')
    plt.plot(tSim, PowerSim[:, 2], 'g', label='$P_{r_3}$')
    plt.plot(tSim, PowerSim[:, 3], 'k', label='$P_{r_4}$')
    plt.title('Instantaneous Power Drawn by Rotors')
    plt.legend(loc='best')
    plt.ylabel('[W]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()

    plt.plot(tSim, energymax*np.ones(tSim.shape), 'k--', label='$E_{max}$')
    plt.plot(tSim, TotalCumEnergySim, 'm', label='$E_{tot}$')
    plt.plot(tSim, CumEnergySim[:, 0], 'b', label='$E_{r_1}$')
    plt.plot(tSim, CumEnergySim[:, 1], 'r', label='$E_{r_2}$')
    plt.plot(tSim, CumEnergySim[:, 2], 'g', label='$E_{r_3}$')
    plt.plot(tSim, CumEnergySim[:, 3], 'k', label='$E_{r_4}$')
    plt.title('Cumulative Energy Consumption by Rotors')
    plt.legend(loc='best')
    plt.ylabel('[ J ]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()
    
    plt.plot(tSim, qSim[:, 0], 'b', label='$q_1$')
    plt.plot(tSim, qSim[:, 1], 'r', label='$q_2$')
    plt.plot(tSim, qSim[:, 2], 'g', label='$q_3$')
    plt.plot(tSim, qSim[:, 3], 'k', label='$q_4$')
    plt.title('Attitude of Body w.r.t. Inertial (actual, reference)')
    plt.legend(loc='best')
    plt.ylabel('[ ]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()   
    
    plt.plot(tSim, qSim[:, 0], 'b', label='$q_1$')
    plt.plot(tSim, qSim[:, 1], 'r', label='$q_2$')
    plt.plot(tSim, qSim[:, 2], 'g', label='$q_3$')
    plt.plot(tSim, qSim[:, 3], 'k', label='$q_4$')
    plt.plot(tSim, qStarTracker_meas[:, 0], 'b:', alpha=0.5, label='$q_{1_m}$')
    plt.plot(tSim, qStarTracker_meas[:, 1], 'r:', alpha=0.5, label='$q_{2_m}$')
    plt.plot(tSim, qStarTracker_meas[:, 2], 'g:', alpha=0.5, label='$q_{3_m}$')
    plt.plot(tSim, qStarTracker_meas[:, 3], 'k:', alpha=0.5, label='$q_{4_m}$')
    plt.plot(tSim, q_filtSim[:, 0], 'b--', alpha=0.5, label='$\hat{q}_1$')
    plt.plot(tSim, q_filtSim[:, 1], 'r--', alpha=0.5, label='$\hat{q}_2$')
    plt.plot(tSim, q_filtSim[:, 2], 'g--', alpha=0.5, label='$\hat{q}_3$')
    plt.plot(tSim, q_filtSim[:, 3], 'k--', alpha=0.5, label='$\hat{q}_4$')
    plt.title('Attitude of Body w.r.t. Inertial (act, meas, est)')
    plt.legend(loc='best')
    plt.ylabel('[ ]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()    
    
    plt.plot(tSim, qdotSim[:, 0], 'b', label='$\dot{q}_1$')
    plt.plot(tSim, qdotSim[:, 1], 'r', label='$\dot{q}_2$')
    plt.plot(tSim, qdotSim[:, 2], 'g', label='$\dot{q}_3$')
    plt.plot(tSim, qdotSim[:, 3], 'k', label='$\dot{q}_4$')
    plt.title('Quaternion Rate of Change')
    plt.legend(loc='best')
    plt.ylabel('[ /s]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()
    
    plt.plot(tSim, wSim[:, 0], 'b', label='$\omega_{x}$')
    plt.plot(tSim, wSim[:, 1], 'r', label='$\omega_{y}$')
    plt.plot(tSim, wSim[:, 2], 'g', label='$\omega_{z}$')
    plt.plot(tSim, wGyro_meas[:, 0], 'b:', alpha=0.5, label='$\omega_{x_m}$')
    plt.plot(tSim, wGyro_meas[:, 1], 'r:', alpha=0.5, label='$\omega_{y_m}$')
    plt.plot(tSim, wGyro_meas[:, 2], 'g:', alpha=0.5, label='$\omega_{z_m}$')
    plt.plot(tSim, w_filtSim[:, 0], 'b--', alpha=0.5, label='$\hat{\omega}_{x}$')
    plt.plot(tSim, w_filtSim[:, 1], 'r--', alpha=0.5, label='$\hat{\omega}_{y}$')
    plt.plot(tSim, w_filtSim[:, 2], 'g--', alpha=0.5, label='$\hat{\omega}_{z}$')
    plt.title('Angular Velocity of Body w.r.t. Inertial (act, meas, est)')
    plt.legend(loc='best')
    plt.ylabel('[rad/s]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()
    
    plt.plot(tSim, wdotSim[:, 0], 'b', label='$\dot{\omega}_1$')
    plt.plot(tSim, wdotSim[:, 1], 'r', label='$\dot{\omega}_2$')
    plt.plot(tSim, wdotSim[:, 2], 'g', label='$\dot{\omega}_3$')
    plt.title('Angular Acceleration of Body w.r.t. Inertial')
    plt.legend(loc='best')
    plt.ylabel('[rad/$s^2$]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()

    plt.plot(tSim, rBSunSensor_act[:, 0], 'b', label='$r_{x}$')
    plt.plot(tSim, rBSunSensor_act[:, 1], 'r', label='$r_{y}$')
    plt.plot(tSim, rBSunSensor_act[:, 2], 'g', label='$r_{z}$')
    plt.plot(tSim, rBSunSensor_meas[:, 0], 'b:', alpha=0.5, label='$r_{x_m}$')
    plt.plot(tSim, rBSunSensor_meas[:, 1], 'r:', alpha=0.5, label='$r_{y_m}$')
    plt.plot(tSim, rBSunSensor_meas[:, 2], 'g:', alpha=0.5, label='$r_{z_m}$')
    plt.title('Reference vector $r$ (actual, measured)')
    plt.legend(loc='best')
    plt.ylabel('[ ]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()
    
    plt.plot(tSim, rBMagnetometer_act[:, 0], 'b', label='$s_{x}$')
    plt.plot(tSim, rBMagnetometer_act[:, 1], 'r', label='$s_{y}$')
    plt.plot(tSim, rBMagnetometer_act[:, 2], 'g', label='$s_{z}$')
    plt.plot(tSim, rBMagnetometer_meas[:, 0], 'b:', alpha=0.5, label='$s_{x_m}$')
    plt.plot(tSim, rBMagnetometer_meas[:, 1], 'r:', alpha=0.5, label='$s_{y_m}$')
    plt.plot(tSim, rBMagnetometer_meas[:, 2], 'g:', alpha=0.5, label='$s_{z_m}$')
    plt.title('Reference vector $s$ (actual, measured)')
    plt.legend(loc='best')
    plt.ylabel('[ ]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()
    
    plt.plot(tSim, wrSim[:, 0], 'b', label='$\omega_{r_1}$')
    plt.plot(tSim, wrSim[:, 1], 'r', label='$\omega_{r_2}$')
    plt.plot(tSim, wrSim[:, 2], 'g', label='$\omega_{r_3}$')
    plt.plot(tSim, wrSim[:, 3], 'k', label='$\omega_{r_4}$')
    plt.title('Rotor Angular Velocities w.r.t Body')
    plt.legend(loc='best')
    plt.ylabel('[rad/s]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()

    plt.plot(tSim, wrdotSim[:, 0], 'b', label='$\dot{\omega}_{r_1}$')
    plt.plot(tSim, wrdotSim[:, 1], 'r', label='$\dot{\omega}_{r_2}$')
    plt.plot(tSim, wrdotSim[:, 2], 'g', label='$\dot{\omega}_{r_3}$')
    plt.plot(tSim, wrdotSim[:, 3], 'k', label='$\dot{\omega}_{r_4}$')
    plt.title('Rotor Accelerations w.r.t Body')
    plt.legend(loc='best')
    plt.ylabel('[rad/$s^2$]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()
        
    plt.plot(tSim, McSim[:, 0], 'b', label='$Mc_1$')
    plt.plot(tSim, McSim[:, 1], 'r', label='$Mc_2$')
    plt.plot(tSim, McSim[:, 2], 'g', label='$Mc_3$')
    plt.title('Specific Control Moments applied about Body Axes')
    plt.legend(loc='best')
    plt.ylabel('[rad/$s^2$]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()
    
    plt.plot(tSim, MextSim[:, 0], 'b', label='$Mext_1$')
    plt.plot(tSim, MextSim[:, 1], 'r', label='$Mext_2$')
    plt.plot(tSim, MextSim[:, 2], 'g', label='$Mext_3$')
    plt.title('Specific External Moments applied about Body Axes')
    plt.legend(loc='best')
    plt.ylabel('[rad/$s^2$]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()
    
    # Folloeing phase plot has artifacts from the plot used in the
    # "bang-bang + PD" implementation. Note that we do not use the 
    # bang-bang law in this implementation. Found that the combination of
    # bang-bang + PD didn't always have predictable behavior
    angle = np.zeros((tSim.shape[0]))
    angle2 = np.zeros((tSim.shape[0]))
    vec_x = np.zeros((tSim.shape[0],3))
    vec_y = np.zeros((tSim.shape[0],3))
    vec_z = np.zeros((tSim.shape[0],3))
    vec_start = np.zeros(3)
    v_store = np.zeros((tSim.shape[0],3))
    ang_vel = np.zeros((tSim.shape[0]))

    for k in np.arange(0,tSim.shape[0]):
        qSim_k = qSim[k,:]
        A_sim_k = utils.quat_to_rot(qSim_k)
        
        A_err = A_sim_k @ np.transpose(A_des)
          
        vec_x[k,:] = np.transpose(A_sim_k) @ np.array([1,0,0])
        vec_y[k,:] = np.transpose(A_sim_k) @ np.array([0,1,0])
        vec_z[k,:] = np.transpose(A_sim_k) @ np.array([0,0,1])
        
        angle[k] = np.arccos( 0.5*( np.trace(A_err)-1 ) )

        wmag_k = np.linalg.norm(wSim[k,:])
        
        if wmag_k != 0:
            w_norm_k = wSim[k,:]/wmag_k
        else:
            w_norm_k = wSim[k,:]
        
        eig_v = np.zeros(3)
        
        eig_v[0] = (A_err[1,2]-A_err[2,1])/(2*np.sin(angle[k]))
        eig_v[1] = (A_err[2,0]-A_err[0,2])/(2*np.sin(angle[k]))
        eig_v[2] = (A_err[0,1]-A_err[1,0])/(2*np.sin(angle[k]))
        
        
        eig_v = eig_v/np.linalg.norm(eig_v)
        
        v_store[k,:] = eig_v
        
        if k == 1:
            vec_start = eig_v
        
        if np.dot(eig_v,vec_start) < 0:
            angle2[k] = -np.arccos( 0.5*( np.trace(A_err)-1 ) )
        else:
            angle2[k] = +np.arccos( 0.5*( np.trace(A_err)-1 ) )
        
        if np.dot(w_norm_k,vec_start) > 0:
            ang_vel[k] = wmag_k
        else:
            ang_vel[k] = -wmag_k
        
    Umax = np.amax(np.absolute(McSim))/Ixx

    xtemp1 = np.arange(-3,0+0.01,0.01)
    xtemp2 = np.arange(0,3+0.01,0.01)
    xtemp3 = np.arange(3,0-0.01,-0.01)
    ytemp1 = np.sqrt(2*Umax)*np.sqrt(-xtemp1)
    ytemp2 = -np.sqrt(2*Umax)*np.sqrt(xtemp2)

    plt.plot(angle2, ang_vel, 'b', label='Satellite phase')
    plt.plot(xtemp3, ytemp1, 'r--', label='Bang-bang switching line')
    plt.plot(np.array([0.1745,0.1745]) , np.array([-1.5,+1.5]), 'k--', label='PD Control Area')
    plt.plot(np.array([-0.1745,-0.1745]) , np.array([-1.5,+1.5]), 'k--')
    plt.plot(xtemp2, ytemp2, 'r--')
    
    plt.title('Phase Plot of Euler Axis Angle Error')
    plt.legend(loc='best')
    plt.ylabel('$\omega$ [rad/s]')
    plt.xlabel(r'$\theta$ [rad]')
    plt.grid()
    plt.show()
    
    plt.plot(tSim, angle*180/np.pi, 'b', label='AxisAngleError')
    plt.title('Attitude Error represented by Euler Axis Angle')
    plt.ylabel('[deg]')
    plt.xlabel('[sec]')
    plt.grid()
    plt.show()
    
    vec_start = np.transpose(A_int) @ np.identity(3)
    vec_end   = np.transpose(A_des) @ np.identity(3)
    ax = plt.axes(projection='3d')
    ax.quiver(np.zeros(3),np.zeros(3),np.zeros(3),vec_start[0,:],vec_start[1,:],vec_start[2,:],color='b',linewidth=0.5,label='Initial orientation')
    ax.quiver(np.zeros(3),np.zeros(3),np.zeros(3),vec_end[0,:],vec_end[1,:],vec_end[2,:],color='r',linewidth=0.5,label='Final orientation')
    ax.quiver(0,0,0,vec_start[0,-1],vec_start[1,-1],vec_start[2,-1],color='b',linewidth=2,label='Initial pointing')
    ax.quiver(0,0,0,vec_end[0,-1],vec_end[1,-1],vec_end[2,-1],color='r',linewidth=2,label='Final pointing')
    ax.plot3D(vec_z[:,0],vec_z[:,1],vec_z[:,2],'g',label='Orientation over time')
    ax.plot3D(vec_y[:,0],vec_y[:,1],vec_y[:,2],'g')
    ax.plot3D(vec_x[:,0],vec_x[:,1],vec_x[:,2],'g')
    ax.plot3D([-1,+1],[0,0],[0,0],'k')
    ax.plot3D([0,0],[-1,+1],[0,0],'k')
    ax.plot3D([0,0],[0,0],[-1,+1],'k')
    ax.set_xlim([-1,+1])
    ax.set_ylim([-1,+1])
    ax.set_zlim([-1,+1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(30.0,190)
    ax.grid()
    ax.legend(loc='best')
    
    return 0


if __name__ == '__main__':
    print('utils.py run as a main file')
