package onnxruntime

// #cgo CFLAGS: -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CFLAGS: -I${SRCDIR}/lib -I/opt/onnxruntime/include/ -I/usr/local/onnxruntime/include -I/usr/local/lib/onnxruntime/include
// #cgo CXXFLAGS: -std=c++11 -g -O3 -Wno-unused-result
// #cgo CXXFLAGS: -I${SRCDIR}/lib -I/opt/onnxruntime/include/ -I/usr/local/onnxruntime/include -I/usr/local/lib/onnxruntime/include
// #cgo LDFLAGS: -lstdc++
// #cgo LDFLAGS: -L/opt/onnxruntime/lib -L/usr/local/lib/onnxruntime/lib -L/usr/local/onnxruntime/lib -lonnxruntime
import "C"
