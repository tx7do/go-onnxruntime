package onnxruntime

// #cgo CXXFLAGS: -std=c++11 -g -O3 -Wno-unused-result
// #cgo CFLAGS: -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo CFLAGS: -I${SRCDIR}/cbits -I/opt/onnxruntime/include/
// #cgo CXXFLAGS: -I${SRCDIR}/cbits -I/opt/onnxruntime/include/
// #cgo LDFLAGS: -L/opt/onnxruntime/lib/ -lstdc++ -lonnxruntime
import "C"
