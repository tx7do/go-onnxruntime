# 获取到版本号
REPO="opencv/opencv"
latest_tag=$(curl -s https://api.github.com/repos/$REPO/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
echo "Using OpenCV version $latest_tag"

# 下载
wget https://github.com/opencv/opencv/archive/$latest_tag.zip

# 解压缩
unzip $latest_tag.zip

# 生成编译文件
cd opencv-$latest_tag && mkdir build && cd build

cmake  -D CMAKE_BUILD_TYPE=RELEASE \
       -D CMAKE_INSTALL_PREFIX=/usr/local \
       -D BUILD_EXAMPLES=OFF \
       -D INSTALL_C_EXAMPLES=OFF \
       -D INSTALL_PYTHON_EXAMPLES=OFF \
       -D OPENCV_GENERATE_PKGCONFIG=ON \
       ..

# 编译
make -j8

# 安装
make install

# 生效pkgconfig
ln -s /usr/local/lib64/pkgconfig/opencv4.pc /usr/share/pkgconfig/
ldconfig

# 查看pkgconfig版本
pkg-config --modversion opencv4

# 清理
rm -fr opencv-$latest_tag
