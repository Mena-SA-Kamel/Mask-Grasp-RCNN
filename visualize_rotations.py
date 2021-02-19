import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R

def plot_vector(axis, origin, vector, colour='k'):
    test_vector_points = np.concatenate([vector.reshape(3, 1), origin], axis=1)
    x, y, z = np.split(test_vector_points, indices_or_sections=3, axis=0)
    axis.plot(x.squeeze(), y.squeeze(), z.squeeze(), color=colour)
    return axis

origin = np.zeros([3, 1])
pitch, roll, yaw = [0, 45, 0]
r = R.from_euler('xyz', [[pitch, roll, yaw]], degrees=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Columns are vx, vy, vz
rotation_matrix = r.as_matrix().squeeze()
vx, vy, vz = np.split(rotation_matrix.squeeze(), indices_or_sections=3, axis=-1)
vec_len = 0.75

p_x = origin + vx*vec_len
p_y = origin + vy*vec_len
p_z = origin + vz*vec_len

# each colum is a data point
axis_points = np.zeros([3,3]) + rotation_matrix*vec_len

colours = ['r', 'g', 'b']
for i in range(axis_points.shape[1]):
    point = axis_points[:, i]
    points = np.concatenate([point.reshape(3, 1), np.zeros((3,1))], axis=1)
    x, y, z = np.split(points, indices_or_sections=3, axis=0)
    ax.plot(x.squeeze(), y.squeeze(), z.squeeze(), color=colours[i])

test_vector = np.array([0.5, 0, -0.5])
test_vector /= np.linalg.norm(test_vector)
plot_vector(ax, origin, test_vector, 'k')

# pitch, roll, yaw = R.from_rotvec(test_vector).as_euler('xyz', degrees=True).tolist()

# Undoing the rotation
rotation = R.from_euler('xyz', [[pitch, roll, 0]], degrees=True).as_matrix()
test_vector_rotated = np.dot(test_vector, rotation.T)
plot_vector(ax, origin, test_vector_rotated, 'c')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

axis_size = 1
ax.set_xlim(-axis_size, axis_size)
ax.set_ylim(-axis_size, axis_size)
ax.set_zlim(-axis_size, axis_size)

plt.show(block=False)
import code; code.interact(local=dict(globals(), **locals()))




