import math
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

np.random.seed(420)
matplotlib.rcParams.update({'font.size': 22})

def generate_impulses(length, min_impulse, max_impulse):
    return np.random.uniform(min_impulse, max_impulse, length)

# state transision model
F = 1.0
f_b = 0.05
# control-input model
B = 0.1
B_b = 0.02
Q = 0.0 # process noise
def transition(x, u):
    """
    Transitions the model to the next hidden state
    :param x: previous previous state
    :param u: control input
    """
    return F * x + f_b + B * x + B_b + np.random.normal(0, Q)

# observation model
H = 1
R = 0.5
def observe(x):
    """
    Observe state x
    :param x: hidden state
    :param R: process noise
    """
    return H * x + np.random.normal(0, R)

# generate data
min_impulse = -1.0
max_impulse = -1.0
length = 20
initial_state = -1
impulses = generate_impulses(length, min_impulse, max_impulse)

hidden = np.array(list(itertools.accumulate(impulses, transition, initial=initial_state)))
observed = np.array(list(map(observe, hidden)))

# kalman filter prediction model
F_k = 1.0
Q_k = 0.1
def predict_state(x_km1_km1):
    """
    a-priori state estimation
    """
    return F_k * x_km1_km1
def predict_covariance(P_km1_km1):
    """
    a-priori covariance
    """
    return F_k * P_km1_km1 * F_k + Q_k

H_k = 1.0
R_k = 0.5
def innovation_residual(z_k, x_km1_km1):
    return z_k - H_k * predict_state(x_km1_km1)
def innovation_covariance(P_km1_km1):
    return H_k * predict_covariance(P_km1_km1) * H_k + R_k
def optimal_kalman_gain(P_km1_km1):
    return predict_covariance(P_km1_km1) * H_k * (1/innovation_covariance(predict_covariance(P_km1_km1)))

def update_state(x_km1_km1, P_km1_km1, z_k):
    return predict_state(x_km1_km1) + optimal_kalman_gain(predict_covariance(P_km1_km1)) * innovation_residual(z_k, x_km1_km1)
def update_covaraince(P_km1_km1):
    return (1 - optimal_kalman_gain(predict_covariance(P_km1_km1)) * H_k) * predict_covariance(P_km1_km1)

def kalman_step(x_km1_km1__P_km1_km1, z_k):
    x_km1_km1, P_km1_km1 = x_km1_km1__P_km1_km1
    x_k_k = update_state(x_km1_km1, P_km1_km1, z_k)
    P_k_k = update_covaraince(P_km1_km1)

    return x_k_k, P_k_k


# infer true state
x_0 = observed[0]   # intial mean is inital observation
P_0 = 2             # unit variance FTW!

infered_states, infered_vars = zip(*itertools.accumulate(observed, kalman_step, initial=(x_0, P_0)))
infered_states = np.array(infered_states)
infered_vars = np.array(infered_vars)
error_bars = 2 * np.sqrt(infered_vars)

# plot

AUV_pos_x = 6
AUV_pos_y = -2.5

def calc_circle_intersect(cxy, r, xy):
    """
    :param cxy: tuple of center of circle
    :param r: radius of circle
    :param xy: point to intercept from
    """
    cx, cy = cxy
    x, y = xy
    dx = x - cx
    dy = y - cy

    theta = math.atan(dy / dx)

    px = cx + r *  math.cos(theta)
    py = cy + r *  math.sin(theta)

    return px, py

ax = plt.subplot(2, 1, 1)
true_path, = plt.plot(hidden, c='b')
observed_path, = plt.plot(observed, c='r')

# plot position 5 estimate
plt.plot(4, hidden[4], marker='+', markersize=15, color="b", markeredgewidth=3)
assumed_position, = plt.plot(4, observed[4], marker='+', markersize=15, color="m", markeredgewidth=3)
AUV, = plt.plot(AUV_pos_x, AUV_pos_y, marker='x', markersize=10, color="green", markeredgewidth=3)
plt.legend([true_path, observed_path, AUV, assumed_position],
           ['True path', 'Observed path', 'AUV', 'Safest guess'])
ax.title.set_text("Raw sensor data")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(2, 1, 2)
true_path, = plt.plot(hidden, c='b')
guessed_path, = plt.plot(infered_states, c='r')


# plot position 5 estimate
circle_x = 4.0
circle_y = infered_states[4]
radius = error_bars[4]
guessed_x, guessed_y = calc_circle_intersect((circle_x, circle_y), radius, (AUV_pos_x, AUV_pos_y))
plt.plot(4, hidden[4], marker='+', markersize=15, color="b", markeredgewidth=3)
assumed_position, = plt.plot(guessed_x, guessed_y, marker='+', markersize=15, color="m", markeredgewidth=3)

ax.add_artist(plt.Circle((4.0, infered_states[4]), radius=error_bars[4], color="red", alpha=0.5))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

safety_margin, = plt.plot(infered_states + error_bars, 'r--')
plt.plot(infered_states - error_bars, 'r--')
AUV, = plt.plot(AUV_pos_x, AUV_pos_y, marker='x', markersize=10, color="green", markeredgewidth=3)
plt.legend([true_path, guessed_path, safety_margin, AUV, assumed_position],
           ['True path', 'Guessed path', 'Safety margin', 'AUV', 'Safest guess'])

ax.title.set_text("Filtered data")
plt.show()
