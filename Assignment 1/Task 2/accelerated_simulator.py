import tensorflow as tf
import numpy as np

# loads the data from the given files
masses = tf.Variable(np.load('q2_input/masses.npy'), dtype = tf.float32) # mass vec
r = tf.Variable(np.load('q2_input/positions.npy'), dtype = tf.float32) # positions vec
v = tf.Variable(np.load('q2_input/velocities.npy'), dtype = tf.float32) # velocities vec
G = tf.Variable(6.67 * (10 ** 5)) # gravitaional constant
flag = tf.Variable(0) # stores threshold separtion condition flag

num_particle = 100
num_dim = 2
r_threshold = 0.1
time_step = 10 ** (-4)

r_x = r[:, 0] # x component of positions vec
r_y = r[:, 1] # y component of positions vec
outer_rx = tf.cast(tf.transpose([r_x]) - r_x, dtype = tf.float32)
outer_ry = tf.cast(tf.transpose([r_y]) - r_y, dtype = tf.float32)
square_outer_x = tf.cast(tf.square(outer_rx), dtype = tf.float32)
square_outer_y = tf.cast(tf.square(outer_ry), dtype = tf.float32)
dist_mat = tf.sqrt(square_outer_x + square_outer_y) # separation matrix
dist_mat = tf.matrix_set_diag(dist_mat, tf.ones(num_particle))
check_threshold = tf.cond(tf.size(tf.where(tf.less_equal(dist_mat, r_threshold))) > 0, lambda: tf.assign(flag, 1), lambda: tf.assign(flag, 0))
dist_mat = tf.reciprocal(dist_mat)
dist_mat = tf.cast(G * tf.pow(dist_mat, 3), dtype = tf.float32)
dist_mat = tf.matrix_set_diag(dist_mat, tf.zeros(num_particle))
dist_mat_x = tf.multiply(dist_mat, outer_rx)
dist_mat_y = tf.multiply(dist_mat, outer_ry)
acc_x = tf.matmul(dist_mat_x, tf.reshape(masses, [tf.size(masses), 1]))
acc_y = tf.matmul(dist_mat_y, tf.reshape(masses, [tf.size(masses), 1]))
acc_x = tf.reshape(acc_x, [tf.size(acc_x), 1])
acc_y = tf.reshape(acc_y, [tf.size(acc_y), 1])
acc_mat = tf.concat([acc_x, acc_y], 1) # acceleration matrix
update_r = tf.assign(r, r + time_step * v + (1 / 2) * (time_step ** 2) * acc_mat) # update the configuration r
update_v = tf.assign(v, v + time_step * acc_mat) # update the configuration v

import time
start_time = time.time()

sess = tf.Session() # declare the sesion
sess.run(tf.global_variables_initializer()) # initialize all variables
threshold_cond = 0 # initally condition is not satisified
counter = 0

while (threshold_cond == 0):
  threshold_cond = sess.run(check_threshold)
  r_new, v_new = sess.run([update_r, update_v]) # new values of r and v
  counter += 1
  print("iter no. : %d, flag value: %d" % (counter, threshold_cond))
  
print(r_new, v_new)
print("--- %s seconds ---" % (time.time() - start_time))
