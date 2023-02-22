#!/usr/bin/env bash

sudo apt update && sudo apt upgrade

# install CMake
sudo apt install cmake

# install gcc
sudo apt install build-essential

# install image lib
sudo apt install libjpeg-dev libtiff-dev libpng-dev

# install OpenCV
sudo apt install libopencv-dev

# install OpenMP
sudo apt-get install libomp-dev
