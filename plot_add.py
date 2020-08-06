import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import numpy as np

add_list_list = np.load("/home/gerard/GIT/pvnet/change_unit_vectors/add_list.npy")
epoch_count = len(add_list_list)
fig = plt.figure()
ax = plt.axes(projection='3d')
print('add_list: ',add_list_list)
X = np.arange(0,len(add_list_list[0]),1)
Z = np.arange(0,epoch_count,1)
Y = add_list_list
X,Z = np.meshgrid(X,Z)
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel("iterations")
ax.set_ylabel("add")
ax.set_zlabel('epoch')
plt.show()
plt.savefig('add_list.png')