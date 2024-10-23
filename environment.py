import gym
from gym import spaces
import numpy as np
import math
import tensorflow as tf
import itertools
import random

class RIS_SISO(gym.Env):
    def __init__(self, num_RIS_elements=8, w_d=0.5, w_k=0.5, k=0, seed=0, num_actions = 8):
        super(RIS_SISO, self).__init__()

        self._max_episode_steps = 20

        self.N = num_RIS_elements
        self.w_d = w_d
        self.w_k = w_k
        self.K = np.load("k.npy")
        # print(self.K)
        self.action_dim = num_actions
        # self.actions = np.random.uniform(0, 2*np.pi, (num_actions, num_RIS_elements))
        self.actions = np.load("ris_8.npy")
        # print(self.actions)
        # Specify the file path
        # filename = "saved_array4.npy"
        # np.save(filename, self.actions)
        self.state_dim = 2 * 3 * self.N + 5 + num_RIS_elements
        # self.state_dim = 2 * 2* self.N + 3
        # self.state_dim = 2 * 2 * self.N + 1
        # self.state_dim = 2 * self.N + 1
        #self.state_dim = 3 * self.N + 3 + 2 * self.N
        # self.h_BE = np.random.normal(0, 1, (1, 1)) + 1j * np.random.normal(0, 1, (1, 1))


        self.H_RA = np.load('h_RA_8.npy')
        self.H_BR = np.load('h_BR_8.npy')
        self.H_RE = np.load('h_RE_8.npy')
        self.H_BE = np.load('h_BE_16.npy')
        self.H_BA = np.load('h_BA_16.npy')

        self.Phi = np.eye(self.N, dtype=complex)
        self.state = None
        self.done = None

        self.episode_t = None

        self.info = {'episode': None, 'true reward': None}
        self.action = None
        self.seed(seed)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.episode_t = 0
        self.info["r1"] = 0
        self.info["r2"] = 0

        self.k = self.K[0].item()

        self.h_RA = self.H_RA[0]
        self.h_BR = self.H_BR[0]
        self.h_RE = self.H_RE[0]
        self.h_BE = self.H_BE[0]
        self.h_BA = self.H_BA[0]

        # print(self.h_BE)
        # print(self.k,self.h_BA)
        state = []




        for _ in self.h_RA, self.h_BR, self.h_RE:
            state.extend(_.real)
            state.extend(_.imag)
        state.append(self.h_BE.real)
        state.append(self.h_BE.imag)
        state.append(self.h_BA.real)
        state.append(self.h_BA.imag)

        # for _ in self.h_RA, self.h_BR, self.h_BA:
        #     state.extend(_.real)
        #     state.extend(_.imag)

        # for _ in self.h_RA, self.h_BR:
        #     state.extend(_.real)
        #     state.extend(_.imag)

        # for _ in self.h_RA:
        #     state.extend(_.real)
        #     state.extend(_.imag)

        for i in range(self.N):
            # state.append(np.array([np.angle(self.Phi[i,i])]))
            state.append(np.angle(self.Phi[i, i]))
        # print(self.K[0].item())
        # state.append(np.array([self.K[0].item()]))
        state.append(self.K[0].item())

        # print(state)
        state = np.squeeze(state)
        # print(state)
        self.state = state
        return self.state


    def compute_reward(self, Phi):
        # print(np.diag(Phi))
        reward = 0
        h_BR = self.h_BR.reshape(1,-1)
        h_RB = self.h_BR.reshape(-1,1)
        h_RA = self.h_RA.reshape(-1,1)
        h_AR = self.h_RA.reshape(1, -1)
        h_RE = self.h_RE.reshape(-1,1)
        h_BA = self.h_BA
        h_BE = self.h_BE
        k = self.k
        v_theta = np.diag(Phi).reshape(1, -1).T
        # print(h_BA.item())
        Λ_RA = np.diag(h_RA.T)
        # print(Λ_RA)
        Λ_RE = np.diag(h_RE.T)

        # print(Λ_RA.T)
        R_A = np.mean(np.abs(Λ_RA.conjugate() @ h_BR.conjugate() @ h_BR.T @ Λ_RA.T))
        R_E = np.mean(np.abs(Λ_RE.conjugate() @ h_AR.conjugate() @ h_AR.T @ Λ_RE.T))
        R_AE = np.mean(np.abs(Λ_RE.conjugate() @ h_BR.conjugate() @ h_BR.T @ Λ_RA.T))

        R_A_D = np.abs(h_BA * h_BA.conjugate()).item()
        R_E_D = np.abs(h_BE * h_BE.conjugate()).item()
        R_AE_D = np.abs(h_BA * h_BE.conjugate()).item()
        # print(Λ_RA)
        #print(v_theta.T[0].conj().reshape(1,-1))
        # z_A = 1 + v_theta.T[0].conj().reshape(1,-1) * R_A @ v_theta + R_A_D
        # z_E = 1 + v_theta.T[0].conj().reshape(1,-1) * R_E @ v_theta + R_E_D
        # z_AE = np.abs(v_theta.conjugate().T * R_AE @ v_theta + R_AE_D).item()
        # # print(v_theta)
        # z_A = 1 + np.abs(v_theta.conjugate().T * R_A @ v_theta + R_A_D).item()
        # z_E = 1 + np.abs(v_theta.conjugate().T * R_E @ v_theta + R_E_D).item()

        z_AE = np.abs(v_theta.T * R_AE @ v_theta + R_AE_D).item()
        # print(v_theta)
        z_A = 1 + np.abs(v_theta.T * R_A @ v_theta + R_A_D).item()
        z_E = 1 + np.abs(v_theta.T * R_E @ v_theta + R_E_D).item()

        R_d = np.log(1 + np.abs(h_RA.T @ Phi @ h_RB + h_BA) ** 2) / np.log(2)
        R_k_0 = math.log(z_A ** 2 / (2 * z_A - 1), 2)

        # print(v_theta,z_A)
        # R_k_0 = z_A ** 2 / (2 * z_A - 1)
        #print(R_k_0)
        t = (z_A * z_E -  z_AE)** 2 / (z_E * ((2 * z_A - 1) * z_E - 2 * z_AE ))
        # print(z_A,z_E,z_AE)
        if  t <= 1:
            R_k_1 = 0
        else:
            R_k_1 = math.log(t, 2)
            # R_k_1 = t
            #print(R_k_1)
        if(k==0):
            r = R_k_0
        else:
            r = R_k_1
        reward = self.w_d * R_d + self.w_k * r


        # print(reward.item(), R_d, R_k_1)

        return reward.item(), R_d.item(), r

    def step(self, action):
        # print(action)
        self.episode_t += 1

        self.k = self.K[self.episode_t-1].item()

        self.h_RA = self.H_RA[self.episode_t-1]
        self.h_BR = self.H_BR[self.episode_t-1]
        self.h_RE = self.H_RE[self.episode_t-1]
        self.h_BE = self.H_BE[self.episode_t-1]
        self.h_BA = self.H_BA[self.episode_t-1]

        for i in range(self.N):
            self.Phi[i,i] = self.actions[action][i]

        done = self.episode_t >= self._max_episode_steps
        reward,self.info["r1"],self.info["r2"] = self.compute_reward(self.Phi)

        state = []


        for _ in self.h_RA, self.h_BR, self.h_RE:
            state.extend(_.real)
            state.extend(_.imag)

        state.append(self.h_BE.real)
        state.append(self.h_BE.imag)
        state.append(self.h_BA.real)
        state.append(self.h_BA.imag)

        # for _ in self.h_RA, self.h_BR, self.h_BA:
        #     state.extend(_.real)
        #     state.extend(_.imag)

        # for _ in self.h_RA, self.h_BR:
        #     state.extend(_.real)
        #     state.extend(_.imag)
        for i in range(self.N):
            # state.append(np.array([np.angle(self.Phi[i,i])]))
            state.append(np.angle(self.Phi[i, i]))

        # for _ in self.h_RA:
        #     state.extend(_.real)
        #     state.extend(_.imag)

        # state.append(np.array([self.k]))
        state.append(self.k)

        self.state = np.squeeze(state)

        return self.state, reward, done, self.info

    def step1(self, action):

        self.k = self.K[self.episode_t-1].item()

        self.h_RA = self.H_RA[self.episode_t-1]
        self.h_BR = self.H_BR[self.episode_t-1]
        self.h_RE = self.H_RE[self.episode_t-1]
        self.h_BE = self.H_BE[self.episode_t-1]
        self.h_BA = self.H_BA[self.episode_t-1]

        for i in range(self.N):
            self.Phi[i,i] = self.actions[action][i]

        done = self.episode_t >= self._max_episode_steps
        reward,self.info["r1"],self.info["r2"] = self.compute_reward(self.Phi)

        state = []


        for _ in self.h_RA, self.h_BR, self.h_RE:
            state.extend(_.real)
            state.extend(_.imag)

        state.append(self.h_BE)
        state.append(self.h_BA)
        # for _ in self.h_RA, self.h_BR, self.h_BA:
        #     state.extend(_.real)
        #     state.extend(_.imag)
        # for i in range(self.N):
        #     state.append(np.array([np.angle(self.Phi[i,i])]))
        state.append(np.array([self.k]))


        self.state = np.squeeze(state)

        return self.state, reward, done, self.info
    def close(self):
        pass
