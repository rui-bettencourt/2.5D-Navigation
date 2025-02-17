from casadi import *
import numpy as np
import math
import timeit
import json


class MPC(object):
    def __init__(self, T, N):
        ################# Config
        self.T = T
        self.N = N
        self.nr_states_base = 8; self.nr_states_arm = 7; self.nr_states = self.nr_states_base + self.nr_states_arm
        self.nr_controls_base = 2; self.nr_controls_arm = 7; self.nr_controls = self.nr_controls_base + self.nr_controls_arm

        self.configs = {'max_Fx': 0.8,  'max_Fyaw': 1.5, 'max_vx':1.0,  'max_vyaw':2.0,  'max_vqs':  1.95,
                        'min_Fx': -0.8, 'min_Fyaw': -1.5, 'min_vx':-0.5, 'min_vyaw':-2.0, 'min_vqs': -1.95}
        with open("/home/rui/pcl_ws/src/mesh_nav/data/joints_limits.json", "r") as file:
            self.joints_limits = json.load(file)

        ############# CONSTANTS #########
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 0.1
        self.g = 9.8
        self.motor_strenght = 1
        self.state_keys = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx', 'vyaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']

        # Define symbolic variables
        self.x_sym = SX.sym('x', self.nr_states)
        self.u_sym = SX.sym('u', self.nr_controls)

        # Nonlinear dynamics function
        self.f_nl = vertcat(
            self.x_sym[6] * cos(self.x_sym[5]),
            self.x_sym[6] * sin(self.x_sym[5]),
            0, 0, 0,
            self.x_sym[7],
            self.u_sym[0] - self.g * sin(self.x_sym[4]) * (1 - self.motor_strenght),
            self.u_sym[1],
            self.u_sym[2], self.u_sym[3], self.u_sym[4], self.u_sym[5], self.u_sym[6], self.u_sym[7], self.u_sym[8]
        )

        # Linearized dynamics
        self.A_sym = jacobian(self.f_nl, self.x_sym)
        self.B_sym = jacobian(self.f_nl, self.u_sym)

        self.A_func = Function('A_func', [self.x_sym, self.u_sym], [self.A_sym])
        self.B_func = Function('B_func', [self.x_sym, self.u_sym], [self.B_sym])


    def compute_linearized_dynamics(self, x0, u0, dt):
        A_eval = self.A_func(x0, u0)
        B_eval = self.B_func(x0, u0)

        A_d = np.eye(self.nr_states) + dt * A_eval
        B_d = dt * B_eval
        # c_d = dt * (self.f_nl - A_eval @ x0 - B_eval @ u0)
        c_d = Function('c_d_func', [self.x_sym, self.u_sym], [self.f_nl - self.A_sym @ self.x_sym - self.B_sym @ self.u_sym])(x0, u0) * dt


        return A_d, B_d, c_d


    def solve_mpc(self, state, gs):
        start_time = timeit.default_timer()
        self.init_state = state # {key: state[key] for key in self.state_keys if key in state.keys()}

        opti = Opti('conic') # Optimization problem

        # ---- decision variables ---------
        self.X = opti.variable(self.nr_states, self.N+1) # state trajectory
        self.U = opti.variable(self.nr_controls, self.N)   # control trajectory (throttle)

        # ---- objective ---------
        objective = 0

        dt = self.T / self.N

        # Linearize around initial state
        x_nom = np.array([self.init_state[key] for key in self.init_state.keys()])
        u_nom = np.zeros(self.nr_controls)  # assume zero controls for linearization point

        A_d, B_d, c_d = self.compute_linearized_dynamics(x_nom, u_nom, dt)

        # ---- dynamic constraints --------
        for k in range(self.N):
            x_next = A_d @ self.X[:, k] + B_d @ self.U[:, k] + c_d
            opti.subject_to(self.X[:, k + 1] == x_next)

        # ---- cost function --------
        objective += self.alpha * (self.X[0, -1] - gs['x'])**2
        objective += self.alpha * (self.X[1, -1] - gs['y'])**2

        for k in range(1, self.N+1):
            for i in range(7):
                objective += self.beta * (self.X[8+i, k] - gs[f'q{i+1}'])**2

        opti.minimize(objective)

        # ---- constraints --------
        opti.subject_to(opti.bounded(self.configs['min_Fx'], self.U[0, :], self.configs['max_Fx']))
        opti.subject_to(opti.bounded(self.configs['min_Fyaw'], self.U[1, :], self.configs['max_Fyaw']))
        opti.subject_to(opti.bounded(self.configs['min_vx'], self.X[6,:] , self.configs['max_vx'] ))
        opti.subject_to(opti.bounded(self.configs['min_vyaw'], self.X[7,:] , self.configs['max_vyaw']))
        opti.subject_to(opti.bounded(self.joints_limits['q1'][0], self.X[8,:],      self.joints_limits['q1'][1]))
        opti.subject_to(opti.bounded(self.joints_limits['q2'][0], self.X[9,:],      self.joints_limits['q2'][1]))
        opti.subject_to(opti.bounded(self.joints_limits['q3'][0], self.X[10,:],      self.joints_limits['q3'][1]))
        opti.subject_to(opti.bounded(self.joints_limits['q4'][0], self.X[11,:],      self.joints_limits['q4'][1]))
        opti.subject_to(opti.bounded(self.joints_limits['q5'][0], self.X[12,:],      self.joints_limits['q5'][1]))
        opti.subject_to(opti.bounded(self.joints_limits['q6'][0], self.X[13,:],      self.joints_limits['q6'][1]))
        opti.subject_to(opti.bounded(self.joints_limits['q7'][0], self.X[14,:],      self.joints_limits['q7'][1]))
        opti.subject_to(opti.bounded(self.configs['min_vqs'], self.U[2, :],     self.configs['max_vqs']))
        opti.subject_to(opti.bounded(self.configs['min_vqs'], self.U[3, :],     self.configs['max_vqs']))
        opti.subject_to(opti.bounded(self.configs['min_vqs'], self.U[4, :],     self.configs['max_vqs']))
        opti.subject_to(opti.bounded(self.configs['min_vqs'], self.U[5, :],     self.configs['max_vqs']))
        opti.subject_to(opti.bounded(self.configs['min_vqs'], self.U[6, :],     self.configs['max_vqs']))
        opti.subject_to(opti.bounded(self.configs['min_vqs'], self.U[7, :],     self.configs['max_vqs']))
        opti.subject_to(opti.bounded(self.configs['min_vqs'], self.U[8, :],     self.configs['max_vqs']))

        # ---- boundary conditions --------
        for i, key in enumerate(self.state_keys):
            opti.subject_to(self.X[i, 0] == self.init_state[key])

        options = {"printLevel": 'none'} #, "ipopt": {"max_iter": 1000}, "ipopt.print_level": 0}
        # opti.solver("ipopt", options)
        solver = opti.solver('qpoases',options)

        self.dt = dt
        self.sol = None
        try:
            self.sol = opti.solve()
            converged = True
        except Exception as e:
            print("Solver failed:", e)
            converged = False

        elapsed = timeit.default_timer() - start_time
        print("To solve took {} seconds".format(elapsed))

        # with open('obj.csv','w') as w2:
        # print(opti.debug.stats())
        # obj_vals =  opti.debug.stats()['iterations']['obj']

        self.cost = 1 #obj_vals[-1]

        return converged

    def get_control(self, i=0):
        control = {}
        control['Fx'] = self.sol.value(self.U[0,i])
        control['Fyaw'] = self.sol.value(self.U[1,i])
        control['dq1'] = self.sol.value(self.U[2,i])
        control['dq2'] = self.sol.value(self.U[3,i])
        control['dq3'] = self.sol.value(self.U[4,i])
        control['dq4'] = self.sol.value(self.U[5,i])
        control['dq5'] = self.sol.value(self.U[6,i])
        control['dq6'] = self.sol.value(self.U[7,i])
        control['dq7'] = self.sol.value(self.U[8,i])
        return control

    def get_control_pose(self, i=0):
        control = {}
        control['vx'] = self.sol.value(self.X[6,i])
        control['vyaw'] = self.sol.value(self.X[7,i])
        control['q1'] = self.sol.value(self.X[8,i])
        control['q2'] = self.sol.value(self.X[9,i])
        control['q3'] = self.sol.value(self.X[10,i])
        control['q4'] = self.sol.value(self.X[11,i])
        control['q5'] = self.sol.value(self.X[12,i])
        control['q6'] = self.sol.value(self.X[13,i])
        control['q7'] = self.sol.value(self.X[14,i])
        # for j in range(0,len(self.sol.value(self.q1))):
        #     print('q1',j, self.sol.value(self.q1[j]))
        return control

    def update_state(self, start):
        state = {}
        state['x'] = self.sol.value(self.X[0,start])
        state['y'] = self.sol.value(self.X[1,start])
        state['z'] = self.sol.value(self.X[2,start])
        state['roll'] = self.sol.value(self.X[3,start])
        state['pitch'] = self.sol.value(self.X[4,start])
        state['yaw'] = self.sol.value(self.X[5,start])

        state['vx'] = self.sol.value(self.X[6,start])
        state['vyaw'] = self.sol.value(self.X[7,start])
        return state

    def interpolate(self, start, goal, N):
        """
        Interpolates N points between a start and goal position (inclusive).
        
        Parameters:
            start (float): The starting position.
            goal (float): The goal position.
            N (int): The number of points to generate (including start and goal).
            
        Returns:
            list: A list of interpolated positions.
        """
        if N <= 1:
            return [start]  # If only one sample is required, return start position.
        
        # Calculate the step size
        step_size = (goal - start) / (N - 1)
        
        # Generate N points between start and goal
        interpolated_points = [start + i * step_size for i in range(N)]
        
        return interpolated_points