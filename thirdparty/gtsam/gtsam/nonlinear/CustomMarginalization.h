/**
* This file is part of DM-VIO.
* The code in this file is in part based on code written by Vladyslav Usenko for the paper "Direct Visual-Inertial Odometry with Stereo Cameras".
*
* Copyright (c) 2022 Lukas von Stumberg <lukas dot stumberg at tum dot de>, Vladyslav Usenko
* for more information see <http://vision.in.tum.de/dm-vio>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DM-VIO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DM-VIO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DM-VIO. If not, see <http://www.gnu.org/licenses/>.
*
* Modified from https://github.com/lukasvst/dm-vio/blob/master/src/GTSAMIntegration/Marginalization.{h,cpp}
* for DBA-Fusion (https://github.com/GREAT-WHU/DBA-Fusion) and shipped with DEIO.
* Included without changes in DPVI-SLAM (2026).
*/

#ifndef CUSTOM_MARGINALIZATION_H
#define CUSTOM_MARGINALIZATION_H

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Vector.h>

namespace gtsam {

gtsam::LinearContainerFactor
marginalizeOut(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values,
               const gtsam::FastVector<gtsam::Key>& keysToMarginalize);

// Fills newGraph with factors which are not connected and marginalizedOutGraph with all factors which will be marginalized out,
// Also fills setOfKeysToMarginalize, and connectedKeys.
void extractKeysToMarginalize(const gtsam::NonlinearFactorGraph& graph, gtsam::NonlinearFactorGraph& newGraph,
                              gtsam::NonlinearFactorGraph& marginalizedOutGraph,
                              gtsam::FastSet<gtsam::Key>& setOfKeysToMarginalize,
                              gtsam::FastSet<gtsam::Key>& connectedKeys);

// Compute the Schur complement with the given dimension of marginalized factors and other factors.
gtsam::Matrix computeSchurComplement(const gtsam::Matrix& augmentedHessian, int mSize, int aSize);

gtsam::Matrix BA2GTSAM(const gtsam::Matrix& H,const gtsam::Vector& v, const gtsam::Pose3& Tbc);
gtsam::Vector GTSAM2BA(const gtsam::Vector& x, const gtsam::Pose3& Tbc);
}
#endif //DMVIO_MARGINALIZATION_H
