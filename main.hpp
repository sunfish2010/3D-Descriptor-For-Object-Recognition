//
// Created by Yu Sun on 11/14/18.
//

#pragma once

#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include "util/utilityCore.hpp"
#include <cuda_runtime.h>
#include "src/detection.h"
#include "src/kdtree.hpp"
using namespace std;


bool showKeyPoints = false;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType>);
pcl::PointCloud<pcl::Normal>::Ptr model_normals(new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors(new pcl::PointCloud<pcl::SHOT352>);
pcl::PointCloud<pcl::SHOT352>::Ptr scene_descriptors(new pcl::PointCloud<pcl::SHOT352>);

// viewpoints
int vp1(0);
int vp2(0);

float model_ss(0.01f);


bool init();
void mainLoop();
void runCUDA();

void keyCallback(const pcl::visualization::KeyboardEvent & event, void *viewer_void);
void mouseCallback(const pcl::visualization::MouseEvent &event, void *viewer_void);
void display();

void detection_cpu();