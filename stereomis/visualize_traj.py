import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def load_extrinsics(pose_path):
    extrinsics = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        pose = list(map(float, line.split(",")))
        pose = torch.tensor(pose).reshape(4, 4).float().transpose(0, 1)
        extrinsic = np.linalg.inv(pose.detach().cpu().numpy())
        extrinsics.append(extrinsic)
    return extrinsics


def load_pose(file_path):
    poses = np.load(file_path)[:, :15]
    poses = poses.reshape(-1, 3, 5)[:, :3, :4]
    last_line = np.tile(np.array([[0, 0, 0, 1]])[None], (poses.shape[0], 1, 1))
    poses = np.concatenate((poses, last_line), axis=1)
    return poses


def draw_camera(ax, c2w, scale=0.03):
    """
    c2w: camera-to-world 4x4
    """
    origin = c2w[:3, 3]
    R = c2w[:3, :3]

    # simple frustum in camera coords
    pts_cam = np.array([
        [0, 0, 0],
        [-1, -1, 2],
        [ 1, -1, 2],
        [ 1,  1, 2],
        [-1,  1, 2],
    ], dtype=float) * scale

    pts_world = (R @ pts_cam.T).T + origin

    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4), (4, 1)
    ]

    for i, j in edges:
        ax.plot(
            [pts_world[i, 0], pts_world[j, 0]],
            [pts_world[i, 1], pts_world[j, 1]],
            [pts_world[i, 2], pts_world[j, 2]],
            linewidth=0.8
        )


poses = load_pose("../StereoMIS_0_0_1/P1/poses_bounds.npy")

camera_centers = []
camera_c2w = []

for i, pose in enumerate(poses):
    extrin = np.linalg.inv(pose)   # world->camera if pose was c2w, or vice versa depending on source
    c2w = np.linalg.inv(extrin)    # camera->world
    camera_c2w.append(c2w)
    camera_centers.append(c2w[:3, 3])

camera_centers = np.stack(camera_centers, axis=0)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# trajectory
ax.plot(
    camera_centers[:, 0],
    camera_centers[:, 1],
    camera_centers[:, 2],
)

# every 3rd camera
for i, c2w in enumerate(camera_c2w):
    if (i + 1) % 3 != 0:
        continue
    draw_camera(ax, c2w, scale=0.03)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Camera trajectory")

# make axes roughly equal
mins = camera_centers.min(axis=0)
maxs = camera_centers.max(axis=0)
center = (mins + maxs) / 2
radius = np.max(maxs - mins) / 2

ax.set_xlim(center[0] - radius, center[0] + radius)
ax.set_ylim(center[1] - radius, center[1] + radius)
ax.set_zlim(center[2] - radius, center[2] + radius)

plt.tight_layout()
plt.savefig("trajectory.png", dpi=200)
print("Saved trajectory.png")