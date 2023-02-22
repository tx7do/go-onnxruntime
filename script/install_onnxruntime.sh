# 获取最新的版本号
REPO="microsoft/onnxruntime"
latest_tag=$(curl -s https://api.github.com/repos/$REPO/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
echo "Using OnnxRuntime version $latest_tag"

# 下载二进制程序压缩包
wget https://github.com/$REPO/releases/download/$latest_tag/onnxruntime-linux-x64-${latest_tag:1}.tgz

# 解压缩
tar -zxvf onnxruntime-linux-x64-${latest_tag:1}.tgz
mv onnxruntime-linux-x64-${latest_tag:1} /usr/local/onnxruntime
rm onnxruntime-linux-x64-${latest_tag:1}.tgz

# 写入onnxruntime.pc
filename="/usr/local/lib/pkgconfig/onnxruntime.pc"
cat>"${filename}"<<EOF
# Package Information for pkg-config

prefix=/usr/local
exec_prefix=${prefix}
libdir=${exec_prefix}/onnxruntime/lib
includedir=${prefix}/onnxruntime/include

Name: onnxruntime
Description: The ONNX Runtime library
Version: 1.14.0
Cflags: -I${includedir}
Libs: -L${libdir} -lonnxruntime
Libs.private: -lstdc++
EOF

# 生效pkgconfig
ln -s /usr/local/lib/pkgconfig/onnxruntime.pc /usr/share/pkgconfig/
ldconfig

# 查看pkgconfig版本
pkg-config --modversion onnxruntime
