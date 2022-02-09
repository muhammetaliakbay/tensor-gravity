from numpy import Infinity
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from tqdm import tqdm

@tf.function
def delta_combination(a):
    expand = tf.expand_dims(a, 0)
    tile = tf.tile(expand, (a.shape[0], 1, 1))
    transpose = tf.transpose(tile, perm=[1,0,2])
    return transpose - tile

@tf.function
def nan_to_zero(a):
    return tf.where(tf.math.is_nan(a), tf.zeros_like(a), a)

particles = 10000
G = 10
mass = 0.0001

time_delta = 0.05
time_factor = 0.5

locations = tf.Variable(tf.random.uniform((particles, 3), -1, 1, seed = 1))
velocities = tf.Variable(tf.random.uniform((particles, 3), -1, 1, seed = 1))
colors = tf.Variable(tf.zeros((particles,)))

@tf.function
def calculate():
    distances = delta_combination(locations)

    abs_distances = tf.sqrt(tf.reduce_sum(tf.square(distances), axis=-1))

    forces = ( G * ( (mass * mass) / abs_distances ) )

    accelerations = forces / mass

    peer_accelerations = (tf.expand_dims(accelerations, -1) * distances) / tf.expand_dims(abs_distances, -1)
    peer_accelerations = nan_to_zero(peer_accelerations)

    accelerations = -tf.reduce_sum(peer_accelerations, -2)

    new_velocities = velocities + accelerations*time_delta*time_factor

    new_locations = locations + new_velocities*time_delta*time_factor
    
    locations.assign(new_locations)
    velocities.assign(new_velocities)

    colors.assign(tf.sqrt(tf.reduce_sum(tf.square(new_velocities), axis=-1)))

def animate(frame):
    calculate()

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
    frames=frames,
    interval=1.0/time_delta,
    blit=False,
)

with tqdm(total=frames) as pbar:
    ani.save(
        "gravity.mp4",
        fps=1.0/time_delta,
        progress_callback=lambda i, n: pbar.update(i)
    )

# plt.show()
