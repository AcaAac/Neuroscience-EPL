# Neuroscience Project Assignments

This repository contains two project assignments for the Mathematical Models in Neuroscience course. Each project folder includes the problem sheet, the submitted code, and the report.

## Table of Contents

- [Project 1: Stochastic Optimal Control](#project-1-neuroscience)
  - [Description](#description)
- [Project 2: EMG/Neural Data Analysis](#project-2-emg-neural-data-analysis)
  - [Description](#description)

## Project 1: Stochastic Optimal Control

### Description

This assignment focuses on implementing a stochastic optimal control system to simulate the reaching movement of a patient with a two-jointed arm model. The control is based on a linear model with muscle dynamics, and the system is governed by the following equations:

$$
M(\theta)\ddot{\theta} + C(\theta, \dot{\theta}) + B\dot{\theta} = \tau
$$

$$
\ddot{\theta} = M(\theta)^{-1} \left( \tau - C(\theta, \dot{\theta}) - B\dot{\theta} \right)
$$

Where:
- $$\tau$$ is the torque vector,
- $$\theta_1$$ and $$\theta_2$$ are the shoulder and elbow angles,
- $$M(\theta)$$, $$C(\theta, \dot{\theta})$$, and $$B$$ are system matrices representing the dynamics of the arm.

The mass matrix $$M(\theta)$$, the Coriolis and centrifugal forces $$C(\theta, \dot{\theta})$$, and the damping matrix $$B$$ are provided as follows:

$$
M(\theta) = \begin{bmatrix} a_1 + 2a_2 \cos(\theta_2) & a_3 + a_2 \cos(\theta_2) \\ a_3 + a_2 \cos(\theta_2) & a_3 \end{bmatrix}
$$

$$
C(\theta, \dot{\theta}) = \begin{bmatrix} -\dot{\theta_2}(2 \dot{\theta_1} + \dot{\theta_2}) \\ \dot{\theta_1} \end{bmatrix}
$$

Where $$a_1$$, $$a_2$$, and $$a_3$$ are specific parameters based on the mechanical properties of the arm (mass, inertia, length), and $$B$$ is a symmetric matrix of constant damping values.

The project also requires simulating the reaching movement of the arm towards different visual targets, considering perturbations and delays in the system.

### Report

The project report can be found in the following file:

```bash
cd Neuroscience-EPL/Project1/LGBIO2072__Project_1.pdf

```

## Project 2: EMG/Neural Data Analysis

### Description

This assignment involves the analysis of both neural and EMG data. The goal is to extract useful information about the kinematic movements and directional tuning of the arm. Specifically, for the neural data, we analyze the firing rate and extrapolate directional tuning based on the neural activity. For the muscle data, we correlate the EMG signals with hand kinetics data.

# Navigate to the directory containing neural data
```bash
cd Neuroscience-EPL/Project2/loadDataP2/dataNeuron
```
Here you can start working with the neural data for firing rate analysis and directional tuning
For example:
Analyze firing rate and directional tuning with the neural data in the `dataNeuron` directory

# Navigate to the directory containing muscle data
```bash
cd ../dataMuscle
```
Here you can start working with the muscle EMG data to correlate with hand kinematics
For example:
Load EMG signals and correlate them with hand movement kinematics

# If you want to run the code for Project 2:
```bash
cd Neuroscience-EPL/Project2/loadDataP2/P2.ipynb
```
# Navigate to the report for Project 2
```bash
cd ../../LGBIO2072__Project_2.pdf
```
