import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from tqdm import tqdm

@tf.function
def delta_combination(a, start, end):
    expand_v = tf.expand_dims(a, 0)
    expand_h = tf.expand_dims(a[start:end], 1)
    tile_v = tf.tile(expand_v, (end-start, 1, 1))
    tile_h = tf.tile(expand_h, (1, a.shape[0], 1))
    return tile_h - tile_v

@tf.function
def nan_to_zero(a):
    return tf.where(tf.math.is_nan(a), tf.zeros_like(a), a)

particles = 10000
G = 10
mass = 0.0001

time_delta = 0.05
time_factor = 0.5

locations = tf.Variable(tf.random.uniform((particles, 3), -1, 1, seed = 1))
velocities = tf.Variable(tf.random.uniform((particles, 3), -0.1, 0.1, seed = 2))
colors = tf.Variable(tf.zeros((particles,)))

@tf.function
def calculate(start, end):
    distances = delta_combination(locations, start, end)

    abs_distances = tf.sqrt(tf.reduce_sum(tf.square(distances), axis=-1))

    forces = ( G * ( (mass * mass) / abs_distances ) )

    accelerations = forces / mass

    peer_accelerations = (tf.expand_dims(accelerations, -1) * distances) / tf.expand_dims(abs_distances, -1)
    peer_accelerations = nan_to_zero(peer_accelerations)

    accelerations = -tf.reduce_sum(peer_accelerations, -2)

    new_velocities = velocities[start:end] + accelerations*time_delta*time_factor
    
    velocities.assign(tf.concat([velocities[:start], new_velocities, velocities[end:]], 0))

    new_locations = locations + velocities*time_delta*time_factor
    locations.assign(new_locations)

    new_colors = tf.sqrt(tf.reduce_sum(tf.square(new_velocities), axis=-1))
    colors.assign(tf.concat([colors[:start], new_colors, colors[end:]], 0))


def animate(frame):
    length = 100
    start = (frame * length) % particles
    end = start + length
    calculate(start, end)

    ax.view_init(30, frame * 0.25)

    graph._offsets3d=(
        locations[:,0].numpy(),
        locations[:,1].numpy(),
        locations[:,2].numpy(),
    )
    graph.set_color(cm.hot(colors.numpy()))
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
    locations[:,0].numpy(),
    locations[:,1].numpy(),
    locations[:,2].numpy(),
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
