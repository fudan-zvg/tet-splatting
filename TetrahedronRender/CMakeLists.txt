#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

cmake_minimum_required(VERSION 3.20)

project(TetrahedronRender LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(TetrahedronRender STATIC
	src/render.cu
	src/ext.cpp
)

set_target_properties(TetrahedronRender PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_include_directories(TetrahedronRender PUBLIC 
	"include"
	"~/miniconda3/envs/cv/lib/python3.9/site-packages/torch/include"
	"~/miniconda3/envs/cv/include/python3.9/"
	"~/miniconda3/envs/cv/lib/python3.9/site-packages/torch/include/torch/csrc/api/include"
)

target_include_directories(TetrahedronRender PRIVATE 
	"third_party/glm" 
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)