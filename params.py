import numpy as np
import utils


""" PARAMETERS """
def get_params():
    """ Earth Parameters """
    mu_E = 398600.4418                           # [km^3/s^2] standard gravitational parameter
    r_E  = 6378.137                              # [km] mean equatorial radius
    w_E  = 0.0000729211585530                    # [rad/s] rotation rate of earth
    
    """ Satellite Body Parameters """
    m   = 20                                     # [kg] total mass
    I   = np.diag([0.2456, 0.2456, 0.2456])      # [kg m^2] MOI
    
    """ Solar Panel Parameters """
    A      = 0.21                                # [m^2] solar panel area
    AM0    = 1353.                               # [W/m^2] solar irradiance
    eff    = 0.28                                # [ ] efficiency of solar panels
    
    """ Rotor Parameters (RWP500 from BCT) """
    A_rot = np.array([[1,0,0,1/np.sqrt(3)],      # rotor orientation (4 rotors)
                      [0,1,0,1/np.sqrt(3)], 
                      [0,0,1,1/np.sqrt(3)]])   
    Ir = 5e-4                     
    Irmat = np.identity(4)*Ir                    # [kgm^2] moments of inertia of rotors
    Mc_max      = 0.025                          # [Nm] maximum moment
    L_max       = 0.500                          # [Nms] maximum momentum
    thetadotmax = 10 #L_max/Ir                   # [rad/s] maximum angular velocity 
    torquemax   = 0.01                           # [Nm] maximum rotor torque 
    powermax    = 0.50                           # [Nm] maximum power to ACS
    energymax   = 5.00                           # [Nm] maximum power to ACS
    
    """ Controller Parameters """
    fn   = 0.5                                   # natural frequency of 2nd order response
    zeta = 1.5                                   # damping ratio of 2nd order response
    Kd = 2*I[0,0]*fn*zeta                        # proportional gain of PD regulator
    Kp = Kd**2/(4*I[0,0])                        # derivative gain of PD regulator
    smallangle = 0.2*np.pi/180                   # [rad] angle at which PD takes over and no longer bang-bang
    
    """ Sensor Parameters """

    """ Reference vector 1 (Sun Sensor) """
    rN1  = 2*np.random.rand(3) - 1               # random vector with elements between -1 and +1
    rN1  = rN1/np.linalg.norm(rN1)               # normalize random vector
    rNSunSensor = rN1                            # reference direction for Sun Sensor
    sigma_suns = 5.0*np.pi/180                   # [rad] error in sun sensor accuracy, 5 degrees converted to radians
    COV_suns = sigma_suns**2 * np.identity(3)    # covariance matrix for Sun Sensor   
    
    """ Reference vector 2 (Magnetometer) """
    rtmp = 2*np.random.rand(3) - 1               # random vector with elements between -1 and +1
    rN2  = np.cross(rN1,rtmp)                    # ensure rN2 is orthogonal to rN1
    rN2  = rN2/np.linalg.norm(rN2)               # normalize random vector
    rNMagnetometer = rN2                         # reference direction fro Magnetometer
    sigma_magn = 3.0*np.pi/180                   # [rad] error in Magnetometer accuracy, 3 degrees converted to radians
    COV_magn = sigma_magn**2 * np.identity(3)    # covariance matrix for magnetometer   
    
    """ Reference attitude (Startracker) """
    sigma_star = 3.0*np.pi/180                   # [rad] error in startracker accuracy, (purposely made much worse than typical spec.)
    COV_star = sigma_star**2 * np.identity(3)    # covariance matrix for startracker             
    
    """ Rate gyro """
    sigma_rnd = 0.1*np.pi/180                    # [rad/s] rate noise density, converted from [deg/s/sqrt(Hz)]
    sigma_arw = 0.1*np.pi/180                    # [rad]   angle random walk, converted from [deg/sqrt(Hz)] 
    COV_rnd   = sigma_rnd**2*np.identity(3)      # covariance matrix for noise
    COV_arw   = sigma_arw**2*np.identity(3)      # covariance matrix for bias
    
    """ Initial and desired final orientation """
    q_t0 = np.array([0,0,0,1])                   # initial quaternion (e.g., the identity quaternion)
                                                 # scalar last, represents rotation from body frame to inertial frame
    A_t0 = utils.quat_to_rot(q_t0)               # rotation matrix corresponding to the initial quaternion
    
    zaxis = np.array([0,0,1])                    # axis of rotation (e.g., the body z-axis)
    phi_tf = (60*np.pi/180) * zaxis              # 45-degree rotation about the z-axis
    
    q_tf = utils.phi_to_quat(phi_tf)             # desired, final quaternion
    A_tf = utils.quat_to_rot(q_tf)               # rotation matrix corresponding to the desired, final quaternion
    
    """ Initial angular velocity of body """
    w_t0 = np.array([0,0,0])                     # [rad/s] initial angular velocity about body x-,y-,z-axis
    
    """ Initial angular velocity of rotors """
    wr_t0 = np.array([0,0,0,0])                  # [rad/s] initial angular velocities of rotors 1, 2, 3, 4
    
    """ Estimator Parameters and Initial Condition """
    b_t0_est  = np.zeros(3)                      # initial bias estimate
    w_t0_est  = w_t0                             # [rad/s] initial angular velocity estimate
    q_t0_est  = q_t0                             # initial quaternion estimate
    P_t0_est  = (10*np.pi/180)**2*np.identity(6) # initial Riccati matrix estimate    
    Pw_t0_est = COV_rnd                          # initial rate gyro riccati matrix?
    
    """ External Moments """
    Mext = np.array([0,0,0])                     # [Nm] zero external moments
    
    """ Initial conditions """
    x0          = np.concatenate((w_t0,q_t0,wr_t0),axis=0)

    """ ODE solve settings and initial condition """
    dt          = 0.1                            # [s] sampling time of simulation
    tEnd        = 30.0                           # [s] simulation end time
    tSpan       = np.arange(0,tEnd+dt,dt)        # [s] seqiemce of time steps
    
    """ Parameter dictionary """
    Earth           = {'mu_E' : mu_E, 'r_E' : r_E, 'w_E' : w_E}
    Body            = {'m' : m, 'I' : I}
    SolarPanels     = {'A' : A, 'AM0' : AM0, 'eff' : eff}
    Power           = {'SolarPanels' : SolarPanels}
    Rotors          = {'A_rot' : A_rot, 'Irmat' : Irmat, 'Mc_max' : Mc_max, 'L_max' : L_max, 'thetadotmax' : thetadotmax, 'torquemax' : torquemax, 'powermax' : powermax, 'energymax' : energymax}
    Magnetorquer    = []
    Actuators       = {'Rotors' : Rotors, 'Magnetorquer' : Magnetorquer}
    Controllers     = {'Kp' : Kp, 'Kd' : Kd, 'smallangle' : smallangle, 'q_t0' : q_t0, 'q_tf' : q_tf, 'A_t0' : A_t0, 'A_tf' : A_tf}
    SunSensor       = {'rNSunSensor' : rNSunSensor, 'COV_suns' : COV_suns}
    Magnetometer    = {'rNMagnetometer' : rNMagnetometer, 'COV_magn' : COV_magn}
    Startracker     = {'COV_star' : COV_star}
    Gyroscope       = {'COV_rnd' : COV_rnd, 'COV_arw' : COV_arw}
    Sensors         = {'SunSensor' : SunSensor, 'Magnetometer' : Magnetometer, 'Startracker' : Startracker, 'Gyroscope' : Gyroscope}
    Estimator       = {'b_t0_est' : b_t0_est, 'w_t0_est' : w_t0_est, 'q_t0_est' : q_t0_est, 'P_t0_est' : P_t0_est, 'Pw_t0_est' : Pw_t0_est}
    ExternalMoments = {'Mext' : Mext}
    Simulation      = {'dt' : dt, 'tEnd' : tEnd, 'tSpan' : tSpan, 'x0': x0}
    Parameters      = {'Earth' : Earth, 'Body' : Body, 'Power' : Power, 'Actuators' : Actuators, 'Controllers' : Controllers, 'Sensors' : Sensors, 'Estimator' : Estimator, 'ExternalMoments' : ExternalMoments, 'Simulation' : Simulation}
    return Parameters            
            
            
if __name__ == '__main__':
    Parameters = get_params()
    print('params.py run as a main file')