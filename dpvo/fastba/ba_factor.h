// This file is part of DPVI-SLAM.
// Copyright (C) 2026 Seunghun Lee
// Licensed under the GNU General Public License v3. See LICENSE.

#pragma once

#include <torch/extension.h>
#include <memory>
#include <vector>
#include "block_e.cuh"

// Depth-marginalized pose system of the patch graph, used as a GTSAM factor.
class BAFactor
{
public:
  void init(torch::Tensor poses,
            torch::Tensor patches,
            torch::Tensor intrinsics,
            torch::Tensor target,
            torch::Tensor weight,
            torch::Tensor lmbda,
            torch::Tensor ii,
            torch::Tensor jj,
            torch::Tensor kk,
            int PPF,
            int t0, int t1, bool eff_impl);

  // H: (6N, 6N), v: (6N), CPU float64
  void hessian(torch::Tensor H, torch::Tensor v);

  // dx: (6N), CPU float64
  void retract(torch::Tensor dx);

private:
  torch::Tensor poses, patches, intrinsics;
  torch::Tensor target, weight, lmbda;
  torch::Tensor ii, jj, kk, kx, ku;
  int PPF, t0, t1;
  bool eff_impl;

  std::unique_ptr<EfficentE> blockE;
  torch::Tensor B, E, C, v, u;
};
