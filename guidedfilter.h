/**
 * This code has been adapted from:
 * https://github.com/acstacey/GLFCV
 * Copyright (c) 2017 Adam Stacey
 * to use the OpenCV CUDA API.
 *
 * It implements the guided filter by Kaiming He (http://kaiminghe.com/eccv10/)
 *
 */

/**
 * gfcuda - Using pybind11 to generate a binding that implement guided filter with opencv and cuda.
 *
 * Copyright (C) 2020 TracelessLe
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <opencv2/opencv.hpp>

class GuidedFilterImpl;

class GuidedFilter {
 public:
  GuidedFilter(const cv::cuda::GpuMat &I, int r, double eps);
  ~GuidedFilter();

  cv::cuda::GpuMat filter(const cv::cuda::GpuMat &p, int depth = -1) const;

 private:
  GuidedFilterImpl *impl_;
};

cv::Mat guidedFilter(const cv::Mat &I,
                              const cv::Mat &p, int r,
                              double eps, int depth = -1);

#endif
