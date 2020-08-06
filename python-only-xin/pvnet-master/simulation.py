import torch
import cv2
import lmdb
import numpy as np
import matplotlib.pyplot as plt

def compute_vertex(mask, points_2d):
    num_keypoints = points_2d.shape[0]
    h, w = mask.shape
    m = points_2d.shape[0]
    xy = np.argwhere(mask == 1)[:, [1, 0]]
    vertex = xy[:, None, :] * np.ones(shape=[1, num_keypoints, 1])
    vertex = points_2d[None, :, :2] - vertex
    norm = np.linalg.norm(vertex, axis=2, keepdims=True)
    norm[norm < 1e-3] += 1e-3
    vertex = vertex / norm

    vertex_out = np.zeros([h, w, m, 2], np.float32)
    vertex_out[xy[:, 1], xy[:, 0]] = vertex
    return np.reshape(vertex_out, [h, w, m * 2])

def visualize_vertex(etaList, keypointGT):

    fig, ax = plt.subplots(2,int(len(etaList)/4),figsize=(30, 12), facecolor='w', edgecolor='k')
    ax = ax.ravel()
    for i in range(int(len(etaList)/2)):
        vertex = etaList[:,:,2*i:2*i+2]
        X, Y = np.meshgrid(np.arange(0,vertex.shape[1],1), np.arange(0,vertex.shape[0],1))   
        U = vertex[:,:,0]
        V = vertex[:,:,1]
        
        ax[i].quiver(X, Y, U, V,units='xy' ,scale=2, color='red')
        ax[i].quiver(keypointGT[0][0],keypointGT[0][1],0,0,units='xy' ,scale=2, color='blue')
        title = 'iter: '+str(i)
        ax[i].set_title(title)

    plt.savefig('vectorField.png')

if __name__ == "__main__":
    imageDim = [20, 20]
    iterations = 10
    gamma = 0.1
    criterion = torch.nn.MSELoss()

    keypointGT = np.array([[np.random.uniform(0,imageDim[0]), np.random.uniform(0,imageDim[1])]])
    mask = np.ones(imageDim)
    vertexGT = compute_vertex(mask, keypointGT)
    eta = torch.tensor(vertexGT, dtype= torch.float)
    
    keypointEstInit = np.array([[np.random.uniform(0,imageDim[0]), np.random.uniform(0,imageDim[1])]])
    vertexEstInit = compute_vertex(mask, keypointEstInit)
    etaHat = torch.tensor(vertexEstInit, dtype= torch.float)
    
    LossList = []
    etaList = np.zeros((iterations*2,imageDim[0],imageDim[1]))
    for i in range(iterations):                             
        Loss = criterion(etaHat, eta)
        LossList.append(Loss)  
        etaList[:,:,2*i:2*i+2] = etaHat
        gradLoss = -2*(eta-etaHat)
        etaHat = etaHat - gamma*gradLoss      
        print(Loss)

    plt.plot(LossList)
    plt.ylabel('L2 Loss')
    plt.xlabel('iteration')
    plt.savefig('loss.png')
    
    visualize_vertex(etaList, keypointGT)