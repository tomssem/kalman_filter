import itertools
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1000003)

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
def observe(x, R):
    """
    Observe state x
    :param x: hidden state
    :param R: process noise
    """
    return H * x + np.random.normal(0, R)

min_impulse = -1.0
max_impulse = -1.0
length = 20
initial_state = -1
impulses = generate_impulses(length, min_impulse, max_impulse)

hidden = np.array(list(itertools.accumulate(impulses, transition, initial=initial_state)))
observed = np.array(list(map(lambda x : observe(x, 0.5), hidden)))

plt.plot(observed, c='b')
plt.plot(hidden, c='r')
plt.plot(hidden + 1, c='r')
plt.plot(hidden - 1, c='r')

plt.plot(8.0, -3.0, marker='x', markersize=10, color="green")

plt.show()
