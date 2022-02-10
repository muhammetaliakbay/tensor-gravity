import os
from queue import Queue
import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from tqdm import tqdm

# @tf.function
def delta_combination(a, peer):
    expand_v = tf.expand_dims(a, 0)
    expand_h = tf.expand_dims(peer, 1)
    tile_v = tf.tile(expand_v, (peer.shape[0], 1, 1))
    tile_h = tf.tile(expand_h, (1, a.shape[0], 1))
    return tile_h - tile_v

batches = 10
particles_per_batch = 1000

G = 10
mass = 0.00001

time_delta = 0.05
time_factor = 0.5

location_variables = [
    tf.Variable(tf.random.uniform((particles_per_batch, 3), -1, 1, seed = i))
    for i in range(batches)
]
velocity_variables = [
    # tf.Variable(location_variables[i].numpy())
    tf.Variable(tf.random.uniform((particles_per_batch, 3), -1, 1, seed = i + 1_000_000))
    for i in range(batches)
]
acceleration_variables = [
        tf.Variable(tf.zeros((particles_per_batch, 3)))
        for _ in range(batches)
    ]
color_variables = [
    tf.Variable(tf.zeros((particles_per_batch,)))
    for i in range(batches)
]

@tf.function
def update():
    for i in range(batches):
        color_variables[i].assign(tf.sqrt(tf.reduce_sum(tf.square(velocity_variables[i]), axis=-1)))
        location_variables[i].assign_add(velocity_variables[i]*time_delta*time_factor)
        velocity_variables[i].assign_add(acceleration_variables[i]*time_delta*time_factor)

@tf.function
def calculate(location_variable, acceleration_variable):
    distances = delta_combination(tf.concat(location_variables, axis=0), location_variable)

    squared_distances = tf.reduce_sum(tf.square(distances), axis=-1)

    forces_div_mass_div_squared_distances = tf.math.divide_no_nan(G * mass, squared_distances)

    peer_accelerations = tf.expand_dims(forces_div_mass_div_squared_distances, -1) * distances

    acceleration_variable.assign(-tf.reduce_sum(peer_accelerations, -2))

queue = Queue(10)

def producer():
    with tf.device('/cpu:0'):
        while True:
            for index in range(batches):
    update()
    calculate(
                    location_variables[index],
                    acceleration_variables[index],
                )
                locations = tf.concat(location_variables, 0)
                queue.put(
                    (
                        locations[:,0].numpy(),
                        locations[:,1].numpy(),
                        locations[:,2].numpy(),
                        tf.concat(color_variables, 0).numpy(),
                    )
    )

threading.Thread(target=producer, daemon=True).start()

def animate(frame):
    # ax.view_init(30, frame * 0.25)

    locations_x, locations_y, locations_z, colors = queue.get()

    graph._offsets3d=(
        locations_x,
        locations_y,
        locations_z,
    )
    graph.set_color(cm.hot(colors))
    title.set_text('Gravity, Frame={}'.format(frame))
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
lim = 5
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])

graph = ax.scatter(
    [],
    [],
    [],
    s=0.5
)

duration = 60
frames = int(duration/time_delta)

ani = FuncAnimation(
    fig,
    animate,
    frames=frames,
    interval=time_delta*1000,
    blit=False,
)

with tqdm(total=frames) as pbar:
    ani.save(
        "gravity.mp4",
        fps=1.0/time_delta,
        progress_callback=lambda i, n: pbar.update(1)
    )

# plt.show()
