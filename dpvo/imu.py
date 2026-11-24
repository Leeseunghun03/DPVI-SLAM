# This file is part of DPVI-SLAM.
# Copyright (C) 2026 Seunghun Lee
# Licensed under the GNU General Public License v3. See LICENSE.

import numpy as np
import gtsam
from scipy.spatial.transform import Rotation


def zero_bias(bg=np.zeros(3)):
    return gtsam.imuBias.ConstantBias(np.zeros(3), bg)


class IMUState:
    def __init__(self, noise, gravity):
        acc, gyr, acc_rw, gyr_rw = noise
        self.params = self._make_params(acc, gyr, acc_rw, gyr_rw, gravity)
        # used across IMU dropouts
        self.params_loose = self._make_params(acc * 10, gyr * 10, acc_rw, gyr_rw, gravity)

        self.cur_t = 0.0
        self.timestamps = []
        self.wTbs = []
        self.vs = []
        self.bs = []
        self.preintegrations = []
        self.measurements = []

    @staticmethod
    def _make_params(acc, gyr, acc_rw, gyr_rw, gravity):
        params = gtsam.PreintegrationCombinedParams.MakeSharedU(gravity)
        params.setAccelerometerCovariance(np.eye(3) * acc**2)
        params.setGyroscopeCovariance(np.eye(3) * gyr**2)
        params.setIntegrationCovariance(np.zeros((3, 3)))
        params.setBiasAccCovariance(np.eye(3) * acc_rw**2)
        params.setBiasOmegaCovariance(np.eye(3) * gyr_rw**2)
        params.setBiasAccOmegaInit(np.zeros((6, 6)))
        return params

    def __len__(self):
        return len(self.wTbs)

    def init_first_state(self, t):
        self.cur_t = t
        self.timestamps.append(t)
        self.wTbs.append(gtsam.Pose3())
        self.vs.append(np.zeros(3))
        self.bs.append(zero_bias())
        self.preintegrations.append(gtsam.PreintegratedCombinedMeasurements(self.params, self.bs[-1]))
        self.measurements.append([])

    def append_imu(self, t, acc, gyr):
        dt = t - self.cur_t
        assert dt >= 0, "IMU measurements must be time-ordered"
        if dt > 0:
            if dt > 0.025:
                self.preintegrations[-1] = self.reintegrate(self.measurements[-1], self.bs[-1], self.params_loose)
            self.preintegrations[-1].integrateMeasurement(acc, gyr, dt)
        self.measurements[-1].append((acc, gyr, dt))
        self.cur_t = t

    def append_frame(self, t):
        prev = gtsam.NavState(self.wTbs[-1], self.vs[-1])
        pim = self.preintegrations[-1]
        pred = pim.predict(prev, self.bs[-1]) if pim.deltaTij() < 1.0 else prev

        self.cur_t = t
        self.timestamps.append(t)
        self.wTbs.append(pred.pose())
        self.vs.append(pred.velocity())
        self.bs.append(self.bs[-1])
        self.preintegrations.append(gtsam.PreintegratedCombinedMeasurements(self.params, self.bs[-1]))
        self.measurements.append([])

    def reintegrate(self, measurements, bias, params=None):
        pim = gtsam.PreintegratedCombinedMeasurements(params or self.params, bias)
        for acc, gyr, dt in measurements:
            if dt > 0:
                pim.integrateMeasurement(acc, gyr, dt)
        return pim

    def remove(self, k):
        for acc, gyr, dt in self.measurements[k]:
            if dt > 0:
                self.preintegrations[k-1].integrateMeasurement(acc, gyr, dt)
            self.measurements[k-1].append((acc, gyr, dt))

        for buf in (self.timestamps, self.wTbs, self.vs, self.bs, self.preintegrations, self.measurements):
            buf.pop(k)


def gravity_aligned_rotation(g):
    """ Rotation taking g to +z with zero yaw. """
    R, _ = Rotation.align_vectors([[0, 0, 1]], [g / np.linalg.norm(g)])
    R = R.as_matrix()
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return Rotation.from_euler('z', -yaw).as_matrix() @ R


def solve_gyro_bias(state, wTbs, t0, t1):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    H = [np.zeros((15, d), order='F') for d in (6, 3, 6, 3, 6, 6)]
    for i in range(t0, t1-1):
        factor = gtsam.CombinedImuFactor(0, 1, 2, 3, 4, 5, state.preintegrations[i])
        err = factor.evaluateErrorCustom(
            gtsam.Pose3(wTbs[i]), state.vs[i], gtsam.Pose3(wTbs[i+1]), state.vs[i+1],
            state.bs[i], state.bs[i+1], *H)
        J = H[4][0:3, 3:6]
        A += J.T @ J
        b += J.T @ err[0:3]
    return -np.linalg.solve(A, b)


def linear_alignment(state, wTbs, t0, t1, gravity):
    """ Returns body-frame velocities, raw and refined gravity, and scale. """
    n = t1 - t0

    def build(dim_g, g_fixed=None, basis=None):
        n_state = 3 * n + dim_g + 1
        A = np.zeros((n_state, n_state))
        b = np.zeros(n_state)
        for c, i in enumerate(range(t0, t1-1)):
            Ri, ti = wTbs[i, :3, :3], wTbs[i, :3, 3]
            Rj, tj = wTbs[i+1, :3, :3], wTbs[i+1, :3, 3]
            pim = state.preintegrations[i]
            dt = pim.deltaTij()

            Ai = np.zeros((6, 6 + dim_g + 1))
            bi = np.zeros(6)
            Ai[0:3, 0:3] = -dt * np.eye(3)
            Ai[3:6, 0:3] = -np.eye(3)
            Ai[3:6, 3:6] = Ri.T @ Rj
            Ai[0:3, -1] = Ri.T @ (tj - ti) / 100.0
            if g_fixed is None:
                Ai[0:3, 6:9] = Ri.T * dt * dt / 2
                Ai[3:6, 6:9] = Ri.T * dt
                bi[0:3] = pim.deltaPij()
                bi[3:6] = pim.deltaVij()
            else:
                Ai[0:3, 6:8] = Ri.T @ basis * dt * dt / 2
                Ai[3:6, 6:8] = Ri.T @ basis * dt
                bi[0:3] = pim.deltaPij() - Ri.T @ g_fixed * dt * dt / 2
                bi[3:6] = pim.deltaVij() - Ri.T @ g_fixed * dt

            rA = Ai.T @ Ai
            rb = Ai.T @ bi
            k = dim_g + 1
            s = slice(3*c, 3*c + 6)
            A[s, s] += rA[:6, :6]
            b[s] += rb[:6]
            A[-k:, -k:] += rA[-k:, -k:]
            b[-k:] += rb[-k:]
            A[s, -k:] += rA[:6, -k:]
            A[-k:, s] += rA[-k:, :6]
        return np.linalg.solve(A * 1000.0, b * 1000.0)

    x = build(3)
    g = x[-4:-1]
    g0 = g / np.linalg.norm(g) * gravity

    # refine gravity direction on its tangent space
    for _ in range(4):
        a = g0 / np.linalg.norm(g0)
        t = np.array([0.0, 0.0, 1.0])
        e1 = t - np.dot(a, t) * a
        e1 /= np.linalg.norm(e1)
        basis = np.stack([e1, np.cross(a, e1)], axis=1)
        x = build(2, g_fixed=g0, basis=basis)
        g0 = g0 + basis @ x[-3:-1]
        g0 = g0 / np.linalg.norm(g0) * gravity

    vels = x[:3*n].reshape(n, 3)
    return vels, g, g0, x[-1] / 100.0
