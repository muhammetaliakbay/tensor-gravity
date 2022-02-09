import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from tqdm import tqdm

@tf.function
def delta_combination(a, slice):
    expand_v = tf.expand_dims(a, 0)
    expand_h = tf.expand_dims(slice, 1)
    tile_v = tf.tile(expand_v, (slice.shape[0], 1, 1))
    tile_h = tf.tile(expand_h, (1, a.shape[0], 1))
    return tile_h - tile_v

@tf.function
def nan_to_zero(a):
    return tf.where(tf.math.is_nan(a), tf.zeros_like(a), a)

batches = 10
particles_per_batch = 250

G = 10
mass = 0.00001

time_delta = 0.05
time_factor = 0.5

location_variables = [
    tf.Variable(tf.random.uniform((particles_per_batch, 3), -1, 1, seed = i))
    for i in range(batches)
]
velocity_variables = [
    tf.Variable(tf.random.uniform((particles_per_batch, 3), -0.1, 0.1, seed = i))
    for i in range(batches)
]
acceleration_variables = [
    tf.Variable(tf.zeros((particles_per_batch, 3)))
    for i in range(batches)
]
color_variables = [
    tf.Variable(tf.zeros((particles_per_batch,)))
    for i in range(batches)
]

@tf.function
def update():
    for i in range(batches):
        location_variables[i].assign_add(velocity_variables[i]*time_delta*time_factor)
        velocity_variables[i].assign_add(acceleration_variables[i]*time_delta*time_factor)

@tf.function
def calculate(location_variable, velocity_variable, acceleration_variable, color_variable):
    locations = tf.concat(location_variables, 0)

    distances = delta_combination(locations, location_variable)

    abs_distances = tf.sqrt(tf.reduce_sum(tf.square(distances), axis=-1))

    forces = ( G * ( (mass * mass) / abs_distances ) )

    peer_accelerations = (tf.expand_dims(forces / mass, -1) * distances) / tf.expand_dims(abs_distances, -1)
    peer_accelerations = nan_to_zero(peer_accelerations)

    acceleration_variable.assign(-tf.reduce_sum(peer_accelerations, -2))

    new_colors = tf.sqrt(tf.reduce_sum(tf.square(velocity_variable), axis=-1))
    color_variable.assign(new_colors)

def animate(frame):
    update()
    calculate(
        location_variables[frame % batches],
        velocity_variables[frame % batches],
        acceleration_variables[frame % batches],
        color_variables[frame % batches],
    )

    ax.view_init(30, frame * 0.25)

    locations = tf.concat(location_variables, 0)

    graph._offsets3d=(
        locations[:,0].numpy(),
        locations[:,1].numpy(),
        locations[:,2].numpy(),
    )
    graph.set_color(cm.hot(tf.concat(color_variables, 0).numpy()))
    title.set_text('3D Test, time={}'.format(frame))
    return title, graph, 

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
title = ax.set_title('Gravity')

ax.set_facecolor('black')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0))

graph = ax.scatter(
    [],
    [],
    [],
    s=0.5
)

frames = int(60/time_delta)

ani = FuncAnimation(
    fig,
    animate,
    # frames=frames,
    interval=1.0/time_delta,
    blit=False,
)

# with tqdm(total=frames) as pbar:
#     ani.save(
#         "gravity.mp4",
#         fps=1.0/time_delta,
#         progress_callback=lambda i, n: pbar.update(i)
#     )

plt.show()
