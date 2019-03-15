import tensorflow as tf
import numpy as np

def acc_mat(masses, r, r_threshold):
  r_x = r[:, 0] # x component of positions vec
  r_y = r[:, 1] # y component of positions vec
  G = 6.67 * (10 ** 5)
  flag = 0
  num_particle = np.size(masses)
  
  outer_rx = np.subtract.outer(r_x, r_x)
  outer_ry = np.subtract.outer(r_y, r_y)
  square_outer_x = np.square(outer_rx)
  square_outer_y = np.square(outer_ry)
  dist_mat = np.sqrt(square_outer_x + square_outer_y) # separation matrix
  np.fill_diagonal(dist_mat, 1)
  # checks for threshold separation
  for i in range(num_particle):
    for j in range(num_particle):
      if dist_mat[i][j] <= r_threshold:
        flag = 1
        break
  dist_mat = np.reciprocal(dist_mat)
  dist_mat = G * np.power(dist_mat, 3)
  np.fill_diagonal(dist_mat, 0)
  dist_mat_x = np.multiply(dist_mat, outer_rx)
  dist_mat_y = np.multiply(dist_mat, outer_ry)
  acc_x = np.matmul(dist_mat_x, np.reshape(masses, (np.size(masses), 1)))
  acc_y = np.matmul(dist_mat_y, np.reshape(masses, (np.size(masses), 1)))
  acc_x = np.reshape(acc_x, (np.size(acc_x), 1))
  acc_y = np.reshape(acc_y, (np.size(acc_y), 1))
  acc_mat = np.concatenate((acc_x, acc_y), 1) # acceleration matrix
  
  return acc_mat, flag


def load_dataset():
  masses = np.load('q2_input/masses.npy')
  r = np.load('q2_input/positions.npy')
  v = np.load('q2_input/velocities.npy')
  return masses, r, v


def run_simulation(masses, init_r, init_v, time_step, r_threshold):
  r = init_r
  v = init_v
  trajectory_r = []
  trajectory_v = []
  
  trajectory_r.append(r)
  trajectory_v.append(v)
  
  flag = 0
  counter = 0
  while flag == 0:
    acc, flag = acc_mat(masses, r, r_threshold)
    r = r + time_step * v + (1 / 2) * (time_step ** 2) * acc
    v = v + time_step * acc
    trajectory_r.append(r)
    trajectory_v.append(v)
    counter += 1
    print("iter no. :", counter)
  
  return trajectory_r, trajectory_v


# main
time_step = 10 ** (-4)
r_threshold = 0.1
masses, init_r, init_v = load_dataset() # loads the dataset

import time
start_time = time.time()

trajectory_r, trajectory_v = run_simulation(masses, init_r, init_v, time_step, r_threshold) # runs simulation

print("--- %s seconds ---" % (time.time() - start_time))