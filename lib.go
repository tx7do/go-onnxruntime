package onnxruntime

// #cgo CFLAGS: -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CXXFLAGS: -std=c++11 -g -O3 -Wno-unused-result
// #cgo linux  pkg-config: onnxruntime
// #cgo darwin pkg-config: onnxruntime
// #cgo freebsd pkg-config: onnxruntime
import "C"
