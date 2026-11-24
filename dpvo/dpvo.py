import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import gtsam
from gtsam.symbol_shorthand import B, V, X
from scipy.spatial.transform import Rotation

from . import altcorr, fastba, lietorch
from . import projective_ops as pops
from .imu import IMUState, gravity_aligned_rotation, linear_alignment, solve_gyro_bias, zero_bias
from .lietorch import SE3
from .net import VONet
from .patchgraph import PatchGraph
from .utils import *

mp.set_start_method('spawn', True)


autocast = torch.cuda.amp.autocast
Id = SE3.Identity(1, device="cuda")


def hessian_factor(values, H, v):
    """ Linear factor 0.5 x^T H x - v^T x on the poses in values. """
    n = H.shape[0]
    info = np.zeros((n + 1, n + 1))
    info[:n, :n] = H
    info[:n, n] = v
    keys = values.keys()
    return gtsam.LinearContainerFactor(gtsam.HessianFactor(keys, [6] * len(keys), info), values)


def pose_to_vec(T):
    """ 4x4 -> [tx ty tz qx qy qz qw] """
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    return torch.as_tensor(np.concatenate([T[:3, 3], q]), dtype=torch.float, device="cuda")


class DPVO:

    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False
        torch.set_num_threads(2)

        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        # keep track of global-BA calls
        self.ran_global_ba = np.zeros(100000, dtype=bool)

        ht = ht // RES
        wd = wd // RES

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        ### network attributes ###
        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}

        ### frame memory size ###
        self.pmem = self.mem = 36 # 32 was too small given default settings
        if self.cfg.LOOP_CLOSURE:
            self.last_global_ba = -1000 # keep track of time since last global opt
            self.pmem = self.cfg.MAX_EDGE_AGE # patch memory

        self.imap_ = torch.zeros(self.pmem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.pmem, self.M, 128, self.P, self.P, **kwargs)

        self.pg = PatchGraph(self.cfg, self.P, self.DIM, self.pmem, **kwargs)

        # classic backend
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.load_long_term_loop_closure()

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        ### inertial attributes ###
        self.imu = None
        self.imu_enabled = False    # IMU factors are in the window optimization
        if self.cfg.ENABLE_IMU:
            noise = (cfg.ACC_NOISE, cfg.GYR_NOISE, cfg.ACC_BIAS_NOISE, cfg.GYR_BIAS_NOISE)
            self.imu = IMUState(noise, cfg.GRAVITY)
            self.Tbc = gtsam.Pose3(np.array(cfg.T_BC, dtype=np.float64).reshape(4, 4))
            self.imu_data = None
            self.imu_idx = 0

            self.vi_init_t1 = -1
            self.vi_init_time = 0.0
            self.reinit = False

            self.marg_factor = None
            self.prior_factors = {}
            self.last_t0 = 0
            self.last_t1 = 0
            self.cur_result = None

            # rotation sigmas (x, y, z) followed by translation
            self.pose_prior_sigma = np.array([0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-4])
            self.bias_prior_sigma = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])

        self.viewer = None
        if viz:
            self.start_viewer()

    def load_long_term_loop_closure(self):
        try:
            from .loop_closure.long_term import LongTermLoopClosure
            self.long_term_lc = LongTermLoopClosure(self.cfg, self.pg)
        except ModuleNotFoundError as e:
            self.cfg.CLASSIC_LOOP_CLOSURE = False
            print(f"WARNING: {e}")

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v

            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

    def start_viewer(self):
        from dpviewer import Viewer

        intrinsics_ = torch.zeros(1, 4, dtype=torch.float32, device="cuda")

        self.viewer = Viewer(
            self.image_,
            self.pg.poses_,
            self.pg.points_,
            self.pg.colors_,
            intrinsics_)

    def set_imu(self, data):
        """ data: (N, 7) array of [t, wx, wy, wz, ax, ay, az] in seconds, rad/s and m/s^2 """
        self.imu_data = np.asarray(data, dtype=np.float64)

    @property
    def poses(self):
        return self.pg.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.pg.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.pg.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.pg.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.pmem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.pmem * self.M, 128, 3, 3)

    @property
    def n(self):
        return self.pg.n

    @n.setter
    def n(self, val):
        self.pg.n = val

    @property
    def m(self):
        return self.pg.m

    @m.setter
    def m(self, val):
        self.pg.m = val

    def camera_poses(self, n=None):
        # poses_ stores world-to-camera
        n = self.n if n is None else n
        return SE3(self.pg.poses_[:n]).inv().matrix().double().cpu().numpy()

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.pg.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            if self.long_term_lc.terminate(self.n):
                print(f"Loop correction applied at keyframe {self.n}")
                self.sync_imu_to_visual()

        if self.cfg.LOOP_CLOSURE:
            self.append_factors(*self.pg.edges_loop())

        for _ in range(12):
            self.ran_global_ba[self.n] = False
            self.update()

        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.pg.tstamps_[i]] = self.pg.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        if self.imu_enabled:
            # report the body (IMU) frame
            poses = SE3(pose_to_vec(self.Tbc.matrix())[None]) * poses
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=np.float64)
        if self.viewer is not None:
            self.viewer.join()

        # Poses: x y z qx qy qz qw
        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.pg.kk, self.pg.jj)
        ii1 = ii % (self.M * self.pmem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.pg.ii, self.pg.jj, self.pg.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.pg.jj = torch.cat([self.pg.jj, jj])
        self.pg.kk = torch.cat([self.pg.kk, ii])
        self.pg.ii = torch.cat([self.pg.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.pg.net = torch.cat([self.pg.net, net], dim=1)

    def remove_factors(self, m, store: bool):
        assert self.pg.ii.numel() == self.pg.weight.shape[1]
        if store:
            self.pg.ii_inac = torch.cat((self.pg.ii_inac, self.pg.ii[m]))
            self.pg.jj_inac = torch.cat((self.pg.jj_inac, self.pg.jj[m]))
            self.pg.kk_inac = torch.cat((self.pg.kk_inac, self.pg.kk[m]))
            self.pg.weight_inac = torch.cat((self.pg.weight_inac, self.pg.weight[:,m]), dim=1)
            self.pg.target_inac = torch.cat((self.pg.target_inac, self.pg.target[:,m]), dim=1)
        self.pg.weight = self.pg.weight[:,~m]
        self.pg.target = self.pg.target[:,~m]

        self.pg.ii = self.pg.ii[~m]
        self.pg.jj = self.pg.jj[~m]
        self.pg.kk = self.pg.kk[~m]
        self.pg.net = self.pg.net[:,~m]
        assert self.pg.ii.numel() == self.pg.weight.shape[1]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.pmem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.pg.ii == i) & (self.pg.jj == j)
        ii = self.pg.ii[k]
        jj = self.pg.jj[k]
        kk = self.pg.kk[k]

        flow, _ = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)

        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.pg.tstamps_[k-1]
            t1 = self.pg.tstamps_[k]

            dP = SE3(self.pg.poses_[k]) * SE3(self.pg.poses_[k-1]).inv()
            self.pg.delta[t1] = (t0, dP)

            to_remove = (self.pg.ii == k) | (self.pg.jj == k)
            self.remove_factors(to_remove, store=False)

            self.pg.kk[self.pg.ii > k] -= self.M
            self.pg.ii[self.pg.ii > k] -= 1
            self.pg.jj[self.pg.jj > k] -= 1

            for i in range(k, self.n-1):
                self.pg.tstamps_[i] = self.pg.tstamps_[i+1]
                self.pg.colors_[i] = self.pg.colors_[i+1]
                self.pg.poses_[i] = self.pg.poses_[i+1]
                self.pg.patches_[i] = self.pg.patches_[i+1]
                self.pg.intrinsics_[i] = self.pg.intrinsics_[i+1]

                self.imap_[i % self.pmem] = self.imap_[(i+1) % self.pmem]
                self.gmap_[i % self.pmem] = self.gmap_[(i+1) % self.pmem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            if self.imu is not None:
                self.imu.remove(k)

            self.n -= 1
            self.m-= self.M

            if self.cfg.CLASSIC_LOOP_CLOSURE:
                self.long_term_lc.keyframe(k)

        to_remove = self.ix[self.pg.kk] < self.n - self.cfg.REMOVAL_WINDOW # Remove edges falling outside the optimization window
        if self.cfg.LOOP_CLOSURE:
            # ...unless they are being used for loop closure
            lc_edges = ((self.pg.jj - self.pg.ii) > 30) & (self.pg.jj > (self.n - self.cfg.OPTIMIZATION_WINDOW))
            to_remove = to_remove & ~lc_edges
        self.remove_factors(to_remove, store=True)

    def __run_global_BA(self):
        """ Global bundle adjustment
         Includes both active and inactive edges """
        full_target = torch.cat((self.pg.target_inac, self.pg.target), dim=1)
        full_weight = torch.cat((self.pg.weight_inac, self.pg.weight), dim=1)
        full_ii = torch.cat((self.pg.ii_inac, self.pg.ii))
        full_jj = torch.cat((self.pg.jj_inac, self.pg.jj))
        full_kk = torch.cat((self.pg.kk_inac, self.pg.kk))

        self.pg.normalize()
        lmbda = torch.as_tensor([1e-4], device="cuda")
        t0 = self.pg.ii.min().item()
        fastba.BA(self.poses, self.patches, self.intrinsics,
            full_target, full_weight, lmbda, full_ii, full_jj, full_kk, t0, self.n, M=self.M, iterations=2, eff_impl=True)
        self.ran_global_ba[self.n] = True

    def update(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=True):
                corr = self.corr(coords)
                ctx = self.imap[:, self.pg.kk % (self.M * self.pmem)]
                self.pg.net, (delta, weight, _) = \
                    self.network.update(self.pg.net, ctx, corr, None, self.pg.ii, self.pg.jj, self.pg.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        self.pg.target = target
        self.pg.weight = weight

        with Timer("BA", enabled=self.enable_timing):
            try:
                global_ba = (self.pg.ii < self.n - self.cfg.REMOVAL_WINDOW - 1).any() and not self.ran_global_ba[self.n]
                if self.imu_enabled:
                    if global_ba:
                        edges = (torch.cat((self.pg.ii_inac, self.pg.ii)),
                                 torch.cat((self.pg.jj_inac, self.pg.jj)),
                                 torch.cat((self.pg.kk_inac, self.pg.kk)),
                                 torch.cat((self.pg.target_inac, self.pg.target), dim=1),
                                 torch.cat((self.pg.weight_inac, self.pg.weight), dim=1))
                        t0 = self.pg.ii.min().item()
                        self.ran_global_ba[self.n] = True
                    else:
                        edges = (self.pg.ii, self.pg.jj, self.pg.kk, target, weight)
                        t0 = max(self.n - self.cfg.OPTIMIZATION_WINDOW, 1)
                    self.vio_ba(edges, lmbda, t0, self.n, eff_impl=global_ba)

                elif global_ba:
                    self.__run_global_BA()
                else:
                    t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
                    t0 = max(t0, 1)
                    fastba.BA(self.poses, self.patches, self.intrinsics,
                        target, weight, lmbda, self.pg.ii, self.pg.jj, self.pg.kk, t0, self.n, M=self.M, iterations=2, eff_impl=False)
                    if self.imu is not None:
                        self.last_t0, self.last_t1 = t0, self.n
            except Exception as e:
                print(f"Warning BA failed... {e}")

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.pg.points_[:len(points)] = points[:]

    def visual_factor(self, edges, lmbda, t0, t1, eff_impl):
        ii, jj, kk, target, weight = edges
        factor = fastba.BAFactor()
        factor.init(self.poses.data, self.patches, self.intrinsics,
            target, weight, lmbda, ii, jj, kk, self.M, t0, t1, eff_impl)
        return factor

    def visual_hessian(self, factor, t0, t1):
        H = torch.zeros(6*(t1-t0), 6*(t1-t0), dtype=torch.float64)
        v = torch.zeros(6*(t1-t0), dtype=torch.float64)
        factor.hessian(H, v)
        Hv = gtsam.BA2GTSAM(H.numpy(), v.numpy(), self.Tbc)
        return Hv[:, :-1], Hv[:, -1]

    def set_prior(self, t0):
        for i in (t0, t0 + 1):
            self.prior_factors[i] = [
                gtsam.PriorFactorPose3(X(i), self.imu.wTbs[i], gtsam.noiseModel.Diagonal.Sigmas(self.pose_prior_sigma)),
                gtsam.PriorFactorConstantBias(B(i), self.imu.bs[i], gtsam.noiseModel.Diagonal.Sigmas(self.bias_prior_sigma))]

    def marginalize(self, t0):
        # Visual factors of the leaving states are dropped: their edge indices
        # may be stale after keyframe removal.
        graph = gtsam.NonlinearFactorGraph()
        keys = []
        for i in range(self.last_t0, t0):
            keys += [X(i), V(i), B(i)]
            graph.push_back(gtsam.CombinedImuFactor(X(i), V(i), X(i+1), V(i+1), B(i), B(i+1), self.imu.preintegrations[i]))

        for i in sorted(self.prior_factors):
            if i < t0:
                for f in self.prior_factors[i]:
                    graph.push_back(f)
                del self.prior_factors[i]

        if self.marg_factor is not None:
            graph.push_back(self.marg_factor)

        self.marg_factor = gtsam.marginalizeOut(graph, self.cur_result, keys)

        if self.reinit:
            # relax the bias prior once, a few seconds after initialization
            all_keys = self.marg_factor.keys()
            for i in range(len(all_keys)):
                if all_keys[i] == B(t0):
                    all_keys[i] = B(0)
            graph = gtsam.NonlinearFactorGraph()
            graph.push_back(self.marg_factor.rekey(all_keys))
            graph.push_back(gtsam.BetweenFactorConstantBias(B(0), B(t0), zero_bias(),
                gtsam.noiseModel.Diagonal.Sigmas(self.bias_prior_sigma)))
            values = self.marg_factor.linearizationPoint()
            values.insert(B(0), values.atConstantBias(B(t0)))
            self.marg_factor = gtsam.marginalizeOut(graph, values, [B(0)])
            self.reinit = False

    def vio_ba(self, edges, lmbda, t0, t1, eff_impl):
        if (self.last_t0, self.last_t1) != (t0, t1):
            if self.last_t0 >= t0:
                if not eff_impl:
                    t0 = self.last_t0
            elif self.cur_result is not None:
                self.marginalize(t0)
            if not eff_impl:
                self.last_t0 = t0
            self.last_t1 = t1

        graph = gtsam.NonlinearFactorGraph()
        for i in range(t0 + 1, t1):
            graph.add(gtsam.CombinedImuFactor(X(i-1), V(i-1), X(i), V(i), B(i-1), B(i), self.imu.preintegrations[i-1]))

        for i in sorted(self.prior_factors):
            if t0 <= i < t1:
                for f in self.prior_factors[i]:
                    graph.push_back(f)

        if not eff_impl and self.marg_factor is not None:
            graph.push_back(self.marg_factor)

        factor = self.visual_factor(edges, lmbda, t0, t1, eff_impl)
        params = gtsam.LevenbergMarquardtParams()

        for itr in range(2):
            H, v = self.visual_hessian(factor, t0, t1)

            poses = gtsam.Values()
            for i in range(t0, t1):
                poses.insert(X(i), self.imu.wTbs[i])
            graph.push_back(hessian_factor(poses, H, v))

            initial = gtsam.Values()
            for i in range(t0, t1):
                initial.insert(X(i), self.imu.wTbs[i])
                initial.insert(V(i), self.imu.vs[i])
                initial.insert(B(i), self.imu.bs[i])

            self.cur_result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()
            graph.resize(graph.size() - 1)

            dx = np.zeros(6 * (t1 - t0))
            for i in range(t0, t1):
                Ti = self.cur_result.atPose3(X(i))
                dx[6*(i-t0):6*(i-t0+1)] = gtsam.Pose3.Logmap(self.imu.wTbs[i].inverse() * Ti)
                self.imu.wTbs[i] = Ti
                self.imu.vs[i] = self.cur_result.atVector(V(i))
                self.imu.bs[i] = self.cur_result.atConstantBias(B(i))

            factor.retract(torch.from_numpy(gtsam.GTSAM2BA(dx, self.Tbc)))

    def sync_imu_to_visual(self):
        if not self.imu_enabled:
            return
        Twc = self.camera_poses()
        Tcb = self.Tbc.inverse().matrix()
        for i in range(self.n):
            R_old = self.imu.wTbs[i].rotation().matrix()
            self.imu.wTbs[i] = gtsam.Pose3(Twc[i] @ Tcb)
            self.imu.vs[i] = self.imu.wTbs[i].rotation().matrix() @ R_old.T @ self.imu.vs[i]

        t0 = max(self.n - self.cfg.OPTIMIZATION_WINDOW, 1)
        self.marg_factor = None
        self.cur_result = None
        self.prior_factors.clear()
        self.set_prior(t0)
        self.last_t0, self.last_t1 = t0, self.n

        points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
        points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
        self.pg.points_[:len(points)] = points[:]

    def propagate_imu(self, t):
        data = self.imu_data
        while self.imu_idx < len(data) - 1 and data[self.imu_idx, 0] < t:
            self.imu.append_imu(data[self.imu_idx, 0], data[self.imu_idx, 4:7], data[self.imu_idx, 1:4])
            self.imu_idx += 1
        self.imu.append_imu(t, data[self.imu_idx, 4:7], data[self.imu_idx, 1:4])
        self.imu.append_frame(t)

    def init_imu(self):
        assert self.imu_data is not None, "ENABLE_IMU is set but set_imu() was not called"
        t = self.tlist[self.pg.tstamps_[0]]
        self.imu_idx = int(np.searchsorted(self.imu_data[:, 0], t))
        self.imu.init_first_state(t)
        for i in range(1, self.n):
            self.propagate_imu(self.tlist[self.pg.tstamps_[i]])

    def align_visual_inertial(self, t0, t1, ignore_lever):
        Tcb = self.Tbc.inverse().matrix()
        Tbc = self.Tbc.matrix()
        if ignore_lever:
            Tcb[:3, 3] = 0.0
            Tbc[:3, 3] = 0.0

        wTbs = self.camera_poses(t1) @ Tcb

        bg = solve_gyro_bias(self.imu, wTbs, t0, t1)
        for i in range(len(self.imu)):
            self.imu.bs[i] = zero_bias(bg)
            self.imu.preintegrations[i] = self.imu.reintegrate(self.imu.measurements[i], self.imu.bs[i])

        vels, g, g0, s = linear_alignment(self.imu, wTbs, t0, t1, self.cfg.GRAVITY)
        if abs(np.linalg.norm(g) - self.cfg.GRAVITY) > 0.5 or s <= 0:
            return False

        wTbs[:, :3, 3] *= s
        for i in range(t0, t1):
            self.imu.vs[i] = wTbs[i, :3, :3] @ vels[i-t0]

        R0 = np.eye(4)
        R0[:3, :3] = gravity_aligned_rotation(g0)
        wTbs = R0 @ wTbs
        for i in range(t1):
            self.imu.vs[i] = R0[:3, :3] @ self.imu.vs[i]
            self.imu.wTbs[i] = gtsam.Pose3(wTbs[i])
            self.pg.poses_[i] = pose_to_vec(np.linalg.inv(wTbs[i] @ Tbc))

        self.pg.patches_[:t1, :, 2] /= s
        s = torch.tensor(s, dtype=torch.float, device="cuda")
        for t, (t_prev, dP) in self.pg.delta.items():
            self.pg.delta[t] = (t_prev, dP.scale(s))

        self.vi_init_t1 = t1
        self.vi_init_time = self.tlist[self.pg.tstamps_[t1-1]]
        return True

    def init_vio(self):
        t1 = self.n
        t0 = t1 - self.cfg.VI_INIT_WINDOW

        acc = np.array([self.imu.preintegrations[i].deltaVij() / self.imu.preintegrations[i].deltaTij()
                        for i in range(t0, t1-1)])
        if np.sqrt(np.mean(np.sum((acc - acc.mean(0))**2, axis=1))) < 0.25:
            return # not enough excitation

        for itr in range(3):
            if not self.align_visual_inertial(t0, t1, ignore_lever=(itr == 0)):
                return
            if itr < 2:
                self.update()

        self.imu_enabled = True
        self.set_prior(self.last_t0)
        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.fix_scale = True
        print(f"Visual-inertial initialized at keyframe {t1}")
        self.update()

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def __call__(self, tstamp, image, intrinsics):
        """ track new frame """

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc(image, self.n)

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--opts BUFFER_SIZE {self.N*2}"')

        if self.viewer is not None:
            self.viewer.update_image(image.contiguous())

        image = 2 * (image[None,None] / 255.0) - 0.5

        with autocast(enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME,
                    centroid_sel_strat=self.cfg.CENTROID_SEL_STRAT,
                    return_color=True)

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.pg.tstamps_[self.n] = self.counter
        self.pg.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.pg.colors_[self.n] = clr.to(torch.uint8)

        self.pg.index_[self.n + 1] = self.n + 1
        self.pg.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.pg.poses_[self.n-1])
                P2 = SE3(self.pg.poses_[self.n-2])

                # To deal with varying camera hz
                *_, a,b,c = [1]*3 + self.tlist
                fac = (c-b) / (b-a)

                xi = self.cfg.MOTION_DAMPING * fac * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.pg.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.pg.poses_[self.n] = tvec_qvec

        # TODO better depth initialization
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
        if self.is_initialized:
            s = torch.median(self.pg.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.pg.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.pmem] = imap.squeeze()
        self.gmap_[self.n % self.pmem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.pg.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M

        if self.cfg.LOOP_CLOSURE:
            if self.n - self.last_global_ba >= self.cfg.GLOBAL_OPT_FREQ:
                """ Add loop closure factors """
                lii, ljj = self.pg.edges_loop()
                if lii.numel() > 0:
                    self.last_global_ba = self.n
                    self.append_factors(lii, ljj)

        # Add forward and backward factors
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True
            if self.imu is not None:
                self.init_imu()

            for itr in range(12):
                self.update()

        elif self.is_initialized:
            if self.imu is not None:
                self.propagate_imu(tstamp)

                if self.imu_enabled:
                    if tstamp - self.vi_init_time > 5.0:
                        self.reinit = True
                        self.vi_init_time = np.inf

                    Twc = (self.imu.wTbs[-1] * self.Tbc).matrix()
                    self.pg.poses_[self.n-1] = pose_to_vec(np.linalg.inv(Twc))

            self.update()
            self.keyframe()

            if self.imu is not None and not self.imu_enabled and self.n > self.cfg.VI_WARMUP:
                self.init_vio()

        if self.cfg.CLASSIC_LOOP_CLOSURE:
            self.long_term_lc.attempt_loop_closure(self.n)
            if self.long_term_lc.lc_callback():
                print(f"Loop correction applied at keyframe {self.n}")
                self.sync_imu_to_visual()
