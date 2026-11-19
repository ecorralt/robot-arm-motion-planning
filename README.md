# Robot Arm Motion Planning

A Python-based 3-DOF robotic arm simulator exploring forward and inverse kinematics, constrained trajectory optimization, and motion planning.

## Project Overview

This project was developed to explore how robotic manipulators move between target positions while satisfying physical constraints.

The simulator models a robotic arm with three degrees of freedom and uses numerical optimization to generate trajectories between an initial configuration and a desired end-effector position.

The project focuses on understanding the mathematical foundations of robot motion planning and implementing them in Python.

## Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/robot-arm-motion-planning.git
cd robot-arm-motion-planning

Install dependencies:

python3 -m pip install -r requirements.txt

## Features

- **Forward Kinematics:** Calculate the end-effector position from the robot's joint angles.
- **Inverse Kinematics:** Determine joint configurations required to reach a target position.
- **Trajectory Optimization:** Generate joint trajectories using CasADi and constrained numerical optimization.
- **Joint Constraints:** Incorporate shoulder and elbow joint limits into the optimization problem.
- **3D Visualization:** Visualize the robotic arm and its motion using Matplotlib.

## Technologies

- **Python** — Main programming language
- **NumPy** — Numerical computations
- **JAX** — Automatic differentiation and kinematics
- **CasADi** — Constrained trajectory optimization
- **Matplotlib** — Interactive 3D simulation

## How It Works

1. Define the robot's initial joint configuration and target end-effector position.
2. Use forward kinematics to calculate the arm's current position.
3. Formulate a constrained optimization problem to determine a feasible trajectory.
4. Generate intermediate joint configurations.
5. Visualize the resulting motion in the 3D simulator.

## Current Limitations

- The simulator models a simplified 3-DOF manipulator rather than a complete physical robotic system.
- The optimization does not currently guarantee a globally optimal trajectory.
- Obstacle avoidance and collision detection have not been implemented.
- Motion is simulated without a full dynamic model of joint torques and actuator forces.

## Future Development

- Improve trajectory efficiency and Cartesian path tracking.
- Investigate singularities and alternative inverse kinematics solutions.
- Introduce collision avoidance.
- Extend the simulator to model an end effector for pick-and-place tasks.

## Project Status

**In development.**

This is an independent learning project focused on robot kinematics, numerical optimization, and motion planning.
