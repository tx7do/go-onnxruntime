# go-onnxruntime

## Install

- Install Image Lib:

    ```shell
    sudo apt install libjpeg-dev libtiff-dev libpng-dev
    ```

- Install CUDA Toolkit: 

    <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>

- Install [ONNX Runtime](https://github.com/microsoft/onnxruntime):

    <https://github.com/microsoft/onnxruntime/releases>

    ```shell
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-x64-1.14.0.tgz
    
    tar -zxvf onnxruntime-linux-x64-1.14.0.tgz
    
    sudo mv onnxruntime-linux-x64-1.14.0 /opt/onnxruntime
    ```

- Install [cupti](https://docs.nvidia.com/cuda/cupti/index.html):

    ```shell
    sudo apt install libcupti-dev
    ```
