import numpy as np
from plyfile import PlyData

scene_gt = "/home/gerard/data/lm/train_pbr/000000/scene_gt.json"
camera_gt = "/home/gerard/data/lm/train_pbr/000000/scene_camera.json"
linemod = {1: "benchvise", 11: "holepuncher", 13: "lamp", 10: "glue", 2: "bowl", 6: "cup", 0: "ape",
           12: "iron", 4: "can", 9: "eggbox", 7: "driller", 14: "phone", 8: "duck", 5: "cat", 3: "cam"}
import json
with open(camera_gt, "r") as f:
    K_dict = json.load(f)
K = K_dict["0"]["cam_K"]
K = np.asarray(K, dtype=np.float)
K = K.reshape(3, 3)
img_id = str(1)
for id in range(12):
    with open(scene_gt, "r") as f:
        rt = json.load(f)
    obj_id = rt[img_id][id]["obj_id"]
    R = rt[img_id][id]["cam_R_m2c"]
    R = np.asarray(R, dtype=np.float)
    R = R.reshape(3, 3)
    T = rt[img_id][id]["cam_t_m2c"]
    T = np.asarray(T, dtype=np.float)
    T = T.reshape(3, 1)
    ply_path = "/home/gerard/data/lm/models/obj_{:06d}.ply".format(obj_id)
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    pts = np.concatenate([data["x"][..., None], data["y"][..., None], data["z"][..., None]], axis=-1)
    pts = pts[::100, :]
    rgb_path = "/home/gerard/data/lm/train_pbr/000000/rgb/{:06d}.jpg".format(int(img_id))
    import matplotlib.pyplot as plt
    img = plt.imread(rgb_path)
    p2d = np.matmul(R, pts.transpose()) + T
    p2d = np.matmul(K, p2d)
    p2d[0, :] /= p2d[2, :]
    p2d[1, :] /= p2d[2, :]
    plt.imshow(img)
    plt.plot(p2d[0, :], p2d[1, :], "r*")
    plt.savefig('/home/gerard/pbr_test/out_{}.png'.format(id))
    print(id)