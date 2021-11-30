import numpy as np
from scipy.linalg import logm


""" UTILITY FUNCTIONS """

def hat(w):
    """ Function takes in a vector of size 3 and returns
    its corresponding skew-symmetric matrix """    
    w1 = w[0]
    w2 = w[1]
    w3 = w[2]
    what  = np.array( [ [0,-w3,w2], [w3,0,-w1], [-w2,w1,0] ] )
    return what

def unhat(what):
    """ Function takes in a skew-symmetric matrix and returns
    its corresponding vector """    
    w1 = what[2,1]
    w2 = what[0,2]
    w3 = what[1,0]
    w  = np.array( (w1,w2,w3) )
    return w

def qmult(q1,q2):
    """ Function takes in quaternions q1 and q2, and performs 
    quaternion multiplication:  q3 = q1*q2 """ 
    v1 = q1[0:3]
    s1 = q1[3]
    q3 = np.block([ [s1*np.identity(3) + hat(v1), v1[:,np.newaxis] ], [-v1, s1] ]) @ q2
    return q3

def qconj(q):
    """ Function takes in a quaternion and returns its conjugate """ 
    v = q[0:3]
    v = -v
    qplus = np.concatenate((v,q[3,np.newaxis]),axis=0)
    return qplus

def phi_to_quat(phi):
    """ Function takes in a rotation parameterized by
    Euler Axis & Angle and returns its corresponding quaternion """
    if np.linalg.norm(phi) > 10*np.pi/180:
        theta = np.linalg.norm(phi)
        r     = phi/theta
        qvec  = r*np.sin(theta/2)
        qsca  = np.array(np.cos(theta/2))
        q     = np.hstack((qvec,qsca))
    else:
        qvec  = phi/2
        qsca  = np.array(1-1/8*np.dot(phi,phi))
        q     = np.hstack((qvec,qsca))
    return q

def quat_to_phi(q):
    """ Function takes in a rotation parameterized by
    a quaternion and returns its corresponding Euler Axis & Angle """  
    Q = quat_to_rot(q)
    phi = unhat(logm(Q))
    return phi

def quat_to_rot(q):
    """ Function takes in a rotation parameterized by
    a quaternion and returns its corresponding rotation matrix """      
    v = q[0:3]
    s  = q[3]    
    A = np.identity(3) + 2*hat(v) @ (s*np.identity(3) + hat(v))
    #A = np.transpose(A)
    return A

""" Below is another way to convert from quaternion to rotation matrix
def quat_to_rot(q): 
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    Q  = np.array( [ [0,-q3,+q2], [+q3,0,-q1], [-q2,q1,0] ] )
    A  = (q4**2 - (q1**2+q2**2+q3**2))*np.identity(3) + 2*np.outer(np.array([q1,q2,q3]), np.array([q1,q2,q3])) - 2*q4 * Q
    return A
"""