
import jax
import jax.numpy as jnp
import numpy as np


def forward_kinematics(theta, l1, l2):
    theta_0, theta_1, theta_2 = theta

    # Calculate endpoint coordinates
    r = l1*jnp.cos(theta_1) + l2*jnp.cos(theta_1 + theta_2)
    x = r*jnp.cos(theta_0)
    y = r*jnp.sin(theta_0)
    z = l1*jnp.sin(theta_1) + l2*jnp.sin(theta_1 + theta_2)

    return jnp.array([x, y, z])
    ##now lets iterate over to the wanted position, lets use newton_raphson for this one. 

    ##formula := x_n+1 = x_n-f(x_n)/f'(x_n)

def newton_raphson(theta_initial, target, l1, l2):
    """Find a 3D tip position from a [base, shoulder, elbow] seed (radians)."""
    target = jnp.asarray(target, dtype=float)
    theta = jnp.asarray(theta_initial, dtype=float)
    if target.shape != (3,) or theta.shape != (3,):
        raise ValueError("target and theta_initial must each contain three values")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(theta)):
        raise ValueError("target and theta_initial must be finite")

    # Initial guess for joint angles
    
    # Define the residual function
    def F(theta):
        return forward_kinematics(theta, l1, l2) - target

    # Jacobian function
    Jf = jax.jacfwd(F)

    for i in range(100):

        # Evaluate Jacobian at current joint angles
        J = Jf(theta)

        # Calculate endpoint error
        error = F(theta)

        # Check convergence
        if jnp.linalg.norm(error) < 1e-6:
            return theta

        # Newton-Raphson correction
        if not np.all(np.isfinite(J)) or np.linalg.cond(np.asarray(J)) > 1e6:
            raise RuntimeError("Newton-Raphson encountered a singular or ill-conditioned pose")
        delta_theta = jnp.linalg.solve(J, error)

        # Backtrack to avoid taking a step that increases the position error.
        for step in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = theta - step * delta_theta
            if float(jnp.linalg.norm(F(candidate))) < float(jnp.linalg.norm(error)):
                theta = candidate
                break
        else:
            raise RuntimeError("Newton-Raphson could not reduce the position error")
    if float(jnp.linalg.norm(F(theta))) < 1e-6:
        return theta
    raise RuntimeError("Newton-Raphson did not converge after 100 iterations")

# def singularity_measure(theta):
#     theta_2 = theta[1]
#     return jnp.abs(jnp.sin(theta_2))## you have to check the singularity before you solve, 
    # you could end up trying to solve a singular jacobian, and get unstable answers. 


# l1 = 1.0
# l2 = 1.0

# x_target = 1.5
# y_target = .5

# # Solve inverse kinematics
# theta_solution = newton_raphson(
#     x_target, y_target, l1, l2
# )

# print("Calculated joint angles:", theta_solution)

# # Verify using forward kinematics
# actual_position = forward_kinematics(
#     theta_solution, l1, l2
# )

# print("Target position:", [x_target, y_target])
# print("Actual position:", actual_position)
# print("Position error:",
#       jnp.linalg.norm(actual_position - jnp.array([x_target, y_target])))
