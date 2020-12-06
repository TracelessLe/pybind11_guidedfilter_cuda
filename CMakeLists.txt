###############################################################################
# gfcuda - Using pybind11 to generate a binding that implement guided filter with opencv and cuda.
#
# Copyright (C) 2020 TracelessLe
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

#Change this if you need to target a specific CMake version
cmake_minimum_required(VERSION 2.8)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)

# Set up project
project(guided_filter_cuda)
add_subdirectory(pybind11)


SET(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/guidedfilter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ndarray_converter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ndarray_converter.h
)

pybind11_add_module(gfcuda ${SOURCES})

# Detect and add OpenCV
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
find_package(OpenCV REQUIRED HINTS /usr/local/share/opencv4)

find_package(CUDA 10.2 REQUIRED)

# Define sources and executable
set(EXECUTABLE_NAME "gfcuda")

# Link against OpenCV
include_directories(${OpenCV_INCLUDE_DIR})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${OpenCV_LIBS})

# Link against CUDA
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${CUDA_LIBRARIES})
