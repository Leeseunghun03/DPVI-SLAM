import os
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from scipy.spatial.transform import Rotation
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory


EUROC_SCENES = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]


def euroc_stream(queue, seqdir, calib, stride):
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    intrinsics = np.array([fx, fy, cx, cy])

    frames = np.loadtxt(os.path.join(seqdir, "mav0/cam0/data.csv"), delimiter=",", dtype=str)[::stride]
    for tstamp, name in frames:
        image = cv2.imread(os.path.join(seqdir, "mav0/cam0/data", name))
        image = cv2.undistort(image, K, calib[4:])
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
        queue.put((int(tstamp) * 1e-9, image, intrinsics))

    queue.put((-1, image, intrinsics))


def load_imu(seqdir):
    """ [t, wx, wy, wz, ax, ay, az] """
    data = np.loadtxt(os.path.join(seqdir, "mav0/imu0/data.csv"), delimiter=",", comments="#")
    data[:, 0] *= 1e-9
    return data


def load_groundtruth(seqdir, T_bc=None):
    """ IMU poses, or camera poses if T_bc is given """
    data = np.loadtxt(os.path.join(seqdir, "mav0/state_groundtruth_estimate0/data.csv"), delimiter=",", comments="#")
    pos, quat = data[:, 1:4], data[:, 4:8]
    if T_bc is not None:
        R = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
        pos = pos + R.apply(T_bc[:3, 3])
        quat = (R * Rotation.from_matrix(T_bc[:3, :3])).as_quat()[:, [3, 0, 1, 2]]
    return PoseTrajectory3D(positions_xyz=pos, orientations_quat_wxyz=quat, timestamps=data[:, 0] * 1e-9)


@torch.no_grad()
def run(cfg, network, seqdir, calib, stride=1, viz=False):

    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=euroc_stream, args=(queue, seqdir, calib, stride))
    reader.start()

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)
            if cfg.ENABLE_IMU:
                slam.set_imu(load_imu(seqdir))

        slam(t, image, intrinsics)

    reader.join()

    return slam.terminate()


def ate(traj_ref, traj_est, correct_scale):
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    result = main_ape.ape(traj_ref, traj_est, PoseRelation.translation_part, align=True, correct_scale=correct_scale)
    return result.stats["rmse"]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/euroc.yaml")
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--eurocdir', default="datasets/EUROC")
    parser.add_argument('--scenes', nargs='+', default=EUROC_SCENES)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("\nRunning with config...")
    print(cfg, "\n")

    torch.manual_seed(1234)

    results = {}
    for scene in args.scenes:
        seqdir = os.path.join(args.eurocdir, scene)
        T_bc = None if cfg.ENABLE_IMU else np.array(cfg.T_BC).reshape(4, 4)
        traj_ref = load_groundtruth(seqdir, T_bc)

        scene_results = []
        for i in range(args.trials):
            poses, tstamps = run(cfg, args.network, seqdir, "calib/euroc.txt", args.stride, args.viz)

            traj_est = PoseTrajectory3D(
                positions_xyz=poses[:,:3],
                orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
                timestamps=tstamps)

            ate_se3 = ate(traj_ref, traj_est, correct_scale=False)
            ate_sim3 = ate(traj_ref, traj_est, correct_scale=True)
            print(f"{scene} trial {i+1}: ATE SE3 {ate_se3:.4f} m, Sim3 {ate_sim3:.4f} m")

            if args.plot:
                Path("trajectory_plots").mkdir(exist_ok=True)
                plot_trajectory(traj_est, traj_ref, f"EuRoC {scene} Trial #{i+1} (ATE: {ate_se3:.03f})",
                                f"trajectory_plots/EuRoC_{scene}_Trial{i+1:02d}.pdf", align=True, correct_scale=False)

            if args.save_trajectory:
                Path("saved_trajectories").mkdir(exist_ok=True)
                file_interface.write_tum_trajectory_file(f"saved_trajectories/EuRoC_{scene}_Trial{i+1:02d}.txt", traj_est)

            scene_results.append((ate_se3, ate_sim3))

        results[scene] = np.median(np.array(scene_results), axis=0)

    print()
    for scene, (se3, sim3) in results.items():
        print(f"{scene:16s} SE3 {se3:.4f}  Sim3 {sim3:.4f}")
    print(f"{'AVG':16s} SE3 {np.mean([r[0] for r in results.values()]):.4f}  Sim3 {np.mean([r[1] for r in results.values()]):.4f}")
