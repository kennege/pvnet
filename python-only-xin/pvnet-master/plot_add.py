import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

add_list_list = np.load("/home/gerard/myPvnet/pvnet/one_iteration_3/add_list.npy")
epoch_count = len(add_list_list)
fig = plt.figure()
ax = fig.gca(projection='3d')
print('mean add: ',np.mean(add_list_list))
print('max add: ',np.max(add_list_list))
X = np.arange(0,len(add_list_list[0]),1)
Z = np.arange(0,epoch_count,1)
X,Z = np.meshgrid(X,Z)
Y = add_list_list
ax.set_xlabel("iterations")
ax.set_ylabel("add")
ax.set_zlabel('epoch')
surf = ax.plot_surface(X, Y, Z,
                linewidth=0, antialiased=False)
plt.show()
plt.savefig('add_list.png')