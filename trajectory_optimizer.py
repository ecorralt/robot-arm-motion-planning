import casadi as ca
import numpy as np
from Kinematics import newton_raphson
SHOULDER_LIMITS = (0.0, np.pi * 4 / 9)
ELBOW_LIMITS = (-np.pi * 11 / 12, -np.pi / 6)
BASE_LIMITS = (0.0, np.pi)
MIN_TIP_HEIGHT = 0.15

N = 30
joint = 3
configuration = N+1

L2 = 1.0
L1 = 1.5
y_target = 1.0
x_target = 0.8
z_target = 0.5
target = np.array([x_target, y_target, z_target])
def optimizer(L1, L2, x_target, y_target, z_target, N, HOME_THETA):
    wp = .05
    wt = .35
    h = .4
    #time step for the trajectory.

    opti = ca.Opti()

    theta = opti.variable(joint, N + 1)
    opti.subject_to(theta[:, 0] == HOME_THETA)

    opti.subject_to(opti.bounded(BASE_LIMITS[0], theta[0, :], BASE_LIMITS[1]))
    opti.subject_to(opti.bounded(SHOULDER_LIMITS[0], theta[1, :], SHOULDER_LIMITS[1]))
    opti.subject_to(opti.bounded(ELBOW_LIMITS[0], theta[2, :], ELBOW_LIMITS[1]))

    # Seed IPOPT with the elbow-up IK branch inside the configured limits.
    # Unconstrained Newton iterations can converge to an equivalent pose on
    # the opposite side of the base or outside the joint limits.
    radial = np.hypot(x_target, y_target)
    cos_elbow = (radial**2 + z_target**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if not np.isfinite(cos_elbow) or abs(cos_elbow) > 1 + 1e-10:
        raise ValueError("Target is outside the arm's geometric reach")
    elbow_target = -np.arccos(np.clip(cos_elbow, -1, 1))
    base_target = np.arctan2(y_target, x_target) if radial > 1e-10 else HOME_THETA[0]
    shoulder_target = np.arctan2(z_target, radial) - np.arctan2(
        L2 * np.sin(elbow_target), L1 + L2 * np.cos(elbow_target))
    theta_target = np.array([base_target, shoulder_target, elbow_target])
    for name, angle, limits in zip(("base", "shoulder", "elbow"), theta_target,
                                   (BASE_LIMITS, SHOULDER_LIMITS, ELBOW_LIMITS)):
        if angle < limits[0] - 1e-8 or angle > limits[1] + 1e-8:
            raise ValueError(f"Target is outside the {name} joint limits")
    initial_trajectory = np.linspace(HOME_THETA,theta_target,N + 1).T

    opti.set_initial(theta, initial_trajectory)



    r = L1*ca.cos(theta[1,:])+ L2*ca.cos(theta[1,:] + theta[2,:])
    x  =r*ca.cos(theta[0,:])
    y = r*ca.sin(theta[0,:]) 
    z = L1*ca.sin(theta[1,:]) + L2*ca.sin(theta[1,:] + theta[2,:])

    r_start = (L1 * np.cos(HOME_THETA[1])+ L2 * np.cos(HOME_THETA[1] + HOME_THETA[2]))
    x_start = r_start*ca.cos(HOME_THETA[0])
    y_start = r_start*ca.sin(HOME_THETA[0])
    z_start = (L1 * ca.sin(HOME_THETA[1])+ L2 * ca.sin(HOME_THETA[1] + HOME_THETA[2]))

    opti.subject_to(x[N] == x_target)
    opti.subject_to(y[N] == y_target)
    opti.subject_to(z[N] == z_target)
    for i in range(N + 1):
        opti.subject_to(z[i] >= MIN_TIP_HEIGHT)
    # Ensure y is non-negative for all time steps
    
    delta_theta = theta[:, 1:] - theta[:, :-1]##this works since the : is able to iterate 

    T = opti.variable()
    opti.set_initial(T, 2.0)

    opti.subject_to(T >= 0.01)

    dt = T/N
    ##defining the angular velocity
    #############################
    omega = delta_theta/dt##this is the angular velocity, and it is defined as the change in angle over time.
    #############################
    ##through the rows of the matrix, and :-1 is able to give the same as the one before the end. same as a for loop.
        
    base_max_angv = 4 * ca.pi / 9
    shoulder_max_angv = ca.pi / 4
    elbow_max_angv = ca.pi / 6

    for j, limit in enumerate([base_max_angv, shoulder_max_angv, elbow_max_angv]):
        opti.subject_to(opti.bounded(-limit, omega[j, :], limit))
    ##assures me that the joint angles cant change more than
    ##5 degrees in either direction. 
    vertical_cost = 0
    for i in range(N+1):
        s = i / N
        z_ref = (z_start+ s * (z_target - z_start)+ 4 * h * s * (1 - s))
        vertical_cost += (z[i] - z_ref)**2
          
    opti.minimize(wt*T + wp*vertical_cost)
    opti.solver("ipopt")
    solution = opti.solve()# make the ipopt solver return the actual solution
    theta_solution = solution.value(theta)##now record the true solution. 

    print("Optimized trajectory:")
    print(np.rad2deg(theta_solution))
    print("Final time:", solution.value(T))

    print("IK result:", theta_target, flush=True)
    return theta_solution.T, float(solution.value(T))

