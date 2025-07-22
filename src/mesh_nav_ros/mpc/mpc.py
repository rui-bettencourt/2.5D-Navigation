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

        self.configs = {'max_Fx': 1.0,  'max_Fyaw': 1.0, 'max_vx':0.3,  'max_vyaw':1.57,  'max_vqs':  1.95,
                        'min_Fx': -1.0, 'min_Fyaw': -1.0, 'min_vx':-0.05, 'min_vyaw':-1.57, 'min_vqs': -1.95}
        with open("/home/rui/pcl_ws/src/mesh_nav/data/joints_limits.json", "r") as file:
            self.joints_limits = json.load(file)

        ############# CONSTANTS #########
        # Gains
        self.alpha          = 2.0
        self.beta           = 1.0
        self.gamma          = 0.1
        self.g              = 9.8
        self.motor_strenght = 1
        self.pose_keys      = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']
        self.vels_keys      = ['vx', 'vyaw']


    def solve_mpc(self, state, gs): # gs is goal state, the reference to take
        # update position + vel of robot
        self.init_state = {key: state[key] for key in self.pose_keys if key in state.keys()}
        self.init_controls = {key: state[key] for key in self.vels_keys if key in state.keys()}

        opti = Opti() # Optimization problem

        # ---- decision variables ---------
        X = opti.variable(self.nr_states, self.N+1) # state trajectory
        U = opti.variable(self.nr_controls, self.N)   # control trajectory (throttle)

        # ---- objective          ---------
        objective = 0


        # state base
        self.pos_x      = X[0,:]
        self.pos_y      = X[1,:]
        self.pos_z      = X[2,:]
        self.pos_roll   = X[3,:]
        self.pos_pitch  = X[4,:]
        self.pos_yaw    = X[5,:]
        self.vel_x      = X[6,:]
        self.vel_yaw    = X[7,:]

        # state arm
        self.q1         = X[8,:]
        self.q2         = X[9,:]
        self.q3         = X[10,:]
        self.q4         = X[11,:]
        self.q5         = X[12,:]
        self.q6         = X[13,:]
        self.q7         = X[14,:]

        # inputs
        self.Fx         = U[0, :]
        self.Fyaw       = U[1, :]

        self.dq1        = U[2, :]
        self.dq2        = U[3, :]
        self.dq3        = U[4, :]
        self.dq4        = U[5, :]
        self.dq5        = U[6, :]
        self.dq6        = U[7, :]
        self.dq7        = U[8, :]

        # ---- dynamic constraints --------
        f = lambda x,u: vertcat(x[6]*cos(x[5]),     # dx/dt = vx*cos(yaw)
                                x[6]*sin(x[5]),     # dy/dt = vx*sin(yaw)
                                0,                  # z not controlled
                                0,                  # roll not controlled
                                0,                  # pitch not controlled
                                x[7],               # dyaw/dt = vyaw
                                u[0], #-self.g*sin(x[4])*(1-self.motor_strenght),               # dvx/dt = A
                                u[1],               # dvyaw/dt = Fyaw
                                u[2],               # dq1/dt
                                u[3],               # dq2/dt
                                u[4],               # dq3/dt
                                u[5],               # dq4/dt
                                u[6],               # dq5/dt
                                u[7],               # dq6/dt
                                u[8])               # dq7/dt

        dt = self.T/self.N # length of a control interval
        for k in range(self.N): # loop over control intervals
            # Runge-Kutta 4 integration
            k1 = f(X[0:self.nr_states,k],         U[0:self.nr_controls,k])
            k2 = f(X[0:self.nr_states,k]+dt/2*k1, U[0:self.nr_controls,k])
            k3 = f(X[0:self.nr_states,k]+dt/2*k2, U[0:self.nr_controls,k])
            k4 = f(X[0:self.nr_states,k]+dt*k3,   U[0:self.nr_controls,k])
            x_next = X[0:self.nr_states,k] + dt/6*(k1+2*k2+2*k3+k4) 
            opti.subject_to(X[0:self.nr_states,k+1]==x_next) # close the gaps

        opti.subject_to(opti.bounded(self.configs['min_Fx'],        self.Fx     , self.configs['max_Fx'])) # control is limited
        opti.subject_to(opti.bounded(self.configs['min_Fyaw'],      self.Fyaw   , self.configs['max_Fyaw'])) # control is limited
        opti.subject_to(opti.bounded(self.configs['min_vx'],        self.vel_x  , self.configs['max_vx'] ))
        opti.subject_to(opti.bounded(self.configs['min_vyaw'],      self.vel_yaw, self.configs['max_vyaw']))
        opti.subject_to(opti.bounded(self.joints_limits['q1'][0],   self.q1,      self.joints_limits['q1'][1]))
        opti.subject_to(opti.bounded(self.joints_limits['q2'][0],   self.q2,      self.joints_limits['q2'][1]))
        opti.subject_to(opti.bounded(self.joints_limits['q3'][0],   self.q3,      self.joints_limits['q3'][1]))
        opti.subject_to(opti.bounded(self.joints_limits['q4'][0],   self.q4,      self.joints_limits['q4'][1]))
        opti.subject_to(opti.bounded(self.joints_limits['q5'][0],   self.q5,      self.joints_limits['q5'][1]))
        opti.subject_to(opti.bounded(self.joints_limits['q6'][0],   self.q6,      self.joints_limits['q6'][1]))
        opti.subject_to(opti.bounded(self.joints_limits['q7'][0],   self.q7,      self.joints_limits['q7'][1]))
        opti.subject_to(opti.bounded(self.configs['min_vqs'],       self.dq1,     self.configs['max_vqs']))
        opti.subject_to(opti.bounded(self.configs['min_vqs'],       self.dq2,     self.configs['max_vqs']))
        opti.subject_to(opti.bounded(self.configs['min_vqs'],       self.dq3,     self.configs['max_vqs']))
        opti.subject_to(opti.bounded(self.configs['min_vqs'],       self.dq4,     self.configs['max_vqs']))
        opti.subject_to(opti.bounded(self.configs['min_vqs'],       self.dq5,     self.configs['max_vqs']))
        opti.subject_to(opti.bounded(self.configs['min_vqs'],       self.dq6,     self.configs['max_vqs']))
        opti.subject_to(opti.bounded(self.configs['min_vqs'],       self.dq7,     self.configs['max_vqs']))

        # ---- boundary conditions --------
        # MPC initial constraints
        opti.subject_to(self.pos_x[0]==self.init_state['x'])
        opti.subject_to(self.pos_y[0]==self.init_state['y'])
        opti.subject_to(self.pos_z[0]==self.init_state['z'])
        opti.subject_to(self.pos_roll[0]==self.init_state['roll'])
        opti.subject_to(self.pos_pitch[0]==self.init_state['pitch'])
        opti.subject_to(self.pos_yaw[0]==self.init_state['yaw'])
        opti.subject_to(self.vel_x[0]==self.init_controls['vx'])
        opti.subject_to(self.vel_yaw[0]==self.init_controls['vyaw'])
        opti.subject_to(self.q1[0]==self.init_state['q1'])
        opti.subject_to(self.q2[0]==self.init_state['q2'])
        opti.subject_to(self.q3[0]==self.init_state['q3'])
        opti.subject_to(self.q4[0]==self.init_state['q4'])
        opti.subject_to(self.q5[0]==self.init_state['q5'])
        opti.subject_to(self.q6[0]==self.init_state['q6'])
        opti.subject_to(self.q7[0]==self.init_state['q7'])


        # initialize variables x and y in the desired reference position
        init_x = self.interpolate(self.init_state['x'], gs['x'], self.N+1)
        init_y = self.interpolate(self.init_state['y'], gs['y'], self.N+1)
        init_q1 = self.interpolate(self.init_state['q1'], gs['q1'], self.N+1)
        init_q2 = self.interpolate(self.init_state['q2'], gs['q2'], self.N+1)
        init_q3 = self.interpolate(self.init_state['q3'], gs['q3'], self.N+1)
        init_q4 = self.interpolate(self.init_state['q4'], gs['q4'], self.N+1)
        init_q5 = self.interpolate(self.init_state['q5'], gs['q5'], self.N+1)
        init_q6 = self.interpolate(self.init_state['q6'], gs['q6'], self.N+1)
        init_q7 = self.interpolate(self.init_state['q7'], gs['q7'], self.N+1)
        for k in range(1,self.N+1):
            opti.set_initial(self.pos_x[k], init_x[k])
            opti.set_initial(self.pos_y[k], init_y[k])
            opti.set_initial(self.q1[k], init_q1[k])
            opti.set_initial(self.q2[k], init_q2[k])
            opti.set_initial(self.q3[k], init_q3[k])
            opti.set_initial(self.q4[k], init_q4[k])
            opti.set_initial(self.q5[k], init_q5[k])
            opti.set_initial(self.q6[k], init_q6[k])
            opti.set_initial(self.q7[k], init_q7[k])

        # define cost function
        # for k in range(0,self.N+1):
        # objective += self.alpha * (self.pos_x[self.N] - gs['x'])**2  # euclidean error in x
        # objective += self.alpha * (self.pos_y[self.N] - gs['y'])**2  # euclidean error in y
        # objective += self.beta  * (self.q1[self.N] - gs['q1'])**2
        # objective += self.beta  * (self.q2[self.N] - gs['q2'])**2
        # objective += self.beta  * (self.q3[self.N] - gs['q3'])**2
        # objective += self.beta  * (self.q4[self.N] - gs['q4'])**2
        # objective += self.beta  * (self.q5[self.N] - gs['q5'])**2
        # objective += self.beta  * (self.q6[self.N] - gs['q6'])**2
        # objective += self.beta  * (self.q7[self.N] - gs['q7'])**2
        # for k in range(1,self.N):
        #     objective -= self.gamma * (self.Fx[k])**2
        #     objective -= self.gamma * (self.Fyaw[k])**2
        #     objective -= self.gamma * (self.dq1[k])**2
        #     objective -= self.gamma * (self.dq2[k])**2
        #     objective -= self.gamma * (self.dq3[k])**2
        #     objective -= self.gamma * (self.dq4[k])**2
        #     objective -= self.gamma * (self.dq5[k])**2
        #     objective -= self.gamma * (self.dq6[k])**2
        #     objective -= self.gamma * (self.dq7[k])**2
        for k in range(1,self.N+1):
            objective += self.alpha * (self.pos_x[k] - gs['x'])**2  # euclidean error in x
            objective += self.alpha * (self.pos_y[k] - gs['y'])**2  # euclidean error in y
            objective += self.beta  * (self.q1[k] - gs['q1'])**2
            objective += self.beta  * (self.q2[k] - gs['q2'])**2
            objective += self.beta  * (self.q3[k] - gs['q3'])**2
            objective += self.beta  * (self.q4[k] - gs['q4'])**2
            objective += self.beta  * (self.q5[k] - gs['q5'])**2
            objective += self.beta  * (self.q6[k] - gs['q6'])**2
            objective += self.beta  * (self.q7[k] - gs['q7'])**2


        opti.minimize(objective)

        # ---- solve NLP              ------
        options = {"print_time": False, "ipopt": {"max_iter": 100000}, "ipopt.print_level": 0}
        opti.solver("ipopt",options) # set numerical backend
        start_time = timeit.default_timer()

        self.dt = dt
        self.sol = None
        try:    
            self.sol = opti.solve()   # actual solve
            converged = True
        except:
            print("Not converged")
            converged = False
        elapsed = timeit.default_timer() - start_time
        print("To solve took {} seconds".format(elapsed))

        # with open('obj.csv','w') as w2:
        obj_vals =  opti.debug.stats()['iterations']['obj']

        self.cost = obj_vals[-1]


        return converged

    def get_control(self, i=0):
        control = {}
        control['Fx'] = self.sol.value(self.Fx[i])
        control['Fyaw'] = self.sol.value(self.Fyaw[i])
        control['dq1'] = self.sol.value(self.dq1[i])
        control['dq2'] = self.sol.value(self.dq2[i])
        control['dq3'] = self.sol.value(self.dq3[i])
        control['dq4'] = self.sol.value(self.dq4[i])
        control['dq5'] = self.sol.value(self.dq5[i])
        control['dq6'] = self.sol.value(self.dq6[i])
        control['dq7'] = self.sol.value(self.dq7[i])
        return control

    def get_control_pose(self, i=0):
        control = {}
        control['vx'] = self.sol.value(self.vel_x[i])
        control['vyaw'] = self.sol.value(self.vel_yaw[i])
        control['q1'] = self.sol.value(self.q1[i])
        control['q2'] = self.sol.value(self.q2[i])
        control['q3'] = self.sol.value(self.q3[i])
        control['q4'] = self.sol.value(self.q4[i])
        control['q5'] = self.sol.value(self.q5[i])
        control['q6'] = self.sol.value(self.q6[i])
        control['q7'] = self.sol.value(self.q7[i])
        # for j in range(0,len(self.sol.value(self.q1))):
        #     print('q1',j, self.sol.value(self.q1[j]))
        # print(self.sol.value(self.vel_x))
        return control

    def update_state(self, start):
        state = {}
        state['x'] = self.sol.value(self.pos_x[start])
        state['y'] = self.sol.value(self.pos_y[start])
        state['z'] = self.sol.value(self.pos_z[start])
        state['roll'] = self.sol.value(self.pos_roll[start])
        state['pitch'] = self.sol.value(self.pos_pitch[start])
        state['yaw'] = self.sol.value(self.pos_yaw[start])

        state['vx'] = self.sol.value(self.vel_x[start])
        state['vyaw'] = self.sol.value(self.vel_yaw[start])
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