import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

@tf.function
def vector_delta_combination(a):
    expand = tf.expand_dims(a, 0)
    tile = tf.tile(expand, (a.shape[0], 1))
    transpose = tf.transpose(tile)
    return transpose - tile

@tf.function
def zero_diagonal(a):
    return tf.linalg.set_diag(
        a,
        tf.zeros(a.shape[:-1]),
    )

particles = 1000
G = 10
mass = 0.0001

time_delta = 0.05
time_factor = 0.25

locations_x = tf.Variable(tf.random.uniform((particles,), -1, 1, seed = 1))
locations_y = tf.Variable(tf.random.uniform((particles,), -1, 1, seed = 2))
locations_z = tf.Variable(tf.random.uniform((particles,), -1, 1, seed = 3))

velocities_x = tf.Variable(tf.random.uniform((particles,), -1, 1, seed = 1))
velocities_y = tf.Variable(tf.random.uniform((particles,), -1, 1, seed = 2))
velocities_z = tf.Variable(tf.random.uniform((particles,), -1, 1, seed = 3))

@tf.function
def calculate(locations_x, locations_y, locations_z, velocities_x, velocities_y, velocities_z):
    distances_x = vector_delta_combination(locations_x)
    distances_y = vector_delta_combination(locations_y)
    distances_z = vector_delta_combination(locations_z)

    distances = tf.sqrt(tf.square(distances_x) + tf.square(distances_y) + tf.square(distances_z))

    forces = ( G * ( (mass * mass) / distances ) )

    accelerations = forces / mass

    accelerations_x = -tf.reduce_sum(zero_diagonal( (accelerations * distances_x) / distances ), -1)
    accelerations_y = -tf.reduce_sum(zero_diagonal( (accelerations * distances_y) / distances ), -1)
    accelerations_z = -tf.reduce_sum(zero_diagonal( (accelerations * distances_z) / distances ), -1)

    new_velocities_x = velocities_x + accelerations_x*time_delta*time_factor
    new_velocities_y = velocities_y + accelerations_y*time_delta*time_factor
    new_velocities_z = velocities_z + accelerations_z*time_delta*time_factor

    new_locations_x = locations_x + new_velocities_x*time_delta*time_factor
    new_locations_y = locations_y + new_velocities_y*time_delta*time_factor
    new_locations_z = locations_z + new_velocities_z*time_delta*time_factor
    
    return new_locations_x, new_locations_y, new_locations_z, new_velocities_x, new_velocities_y, new_velocities_z

def animate(frame):
    (
        new_locations_x, new_locations_y, new_locations_z,
        new_velocities_x, new_velocities_y, new_velocities_z,
    ) = calculate(
        locations_x, locations_y, locations_z,
        velocities_x, velocities_y, velocities_z,
    )

    locations_x.assign(new_locations_x)
    locations_y.assign(new_locations_y)
    locations_z.assign(new_locations_z)

    velocities_x.assign(new_velocities_x)
    velocities_y.assign(new_velocities_y)
    velocities_z.assign(new_velocities_z)

    ax.view_init(30, frame * 0.25)

    graph.set_data(locations_x.numpy(), locations_y.numpy())
    graph.set_3d_properties(locations_z.numpy())
    title.set_text('3D Test, time={}'.format(frame))
    return title, graph, 

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
title = ax.set_title('Gravity')

graph, = ax.plot(locations_x.numpy(), locations_y.numpy(), locations_z.numpy(), linestyle="", marker="o", ms=0.75)

ani = FuncAnimation(fig, animate, frames=int(30/time_delta), interval=1.0/time_delta, blit=False)
# ani.save("gravity.mp4")

plt.show()
