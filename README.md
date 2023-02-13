# go-onnxruntime

## Install

### Install Image Lib:

```shell
sudo apt install libjpeg-dev libtiff-dev libpng-dev
```

### Install [ONNX Runtime](https://github.com/microsoft/onnxruntime):

<https://github.com/microsoft/onnxruntime/releases>

```shell
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-x64-1.14.0.tgz

tar -zxvf onnxruntime-linux-x64-1.14.0.tgz

sudo mv onnxruntime-linux-x64-1.14.0 /opt/onnxruntime

sudo chown -R `whoami` /opt/onnxruntime
```

### Install CUDA

#### Install CUDA Toolkit:

<https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>

#### Install [cupti](https://docs.nvidia.com/cuda/cupti/index.html):

```shell
sudo apt install libcupti-dev
```

## Configure Environmental Variables

Configure the linker environmental variables since the Onnxruntime C++ library is under a non-system directory. Place the following in either your `~/.bashrc` or `~/.zshrc` file:

Linux
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/onnxruntime/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/onnxruntime/lib

```

macOS
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/onnxruntime/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/onnxruntime/lib
```
