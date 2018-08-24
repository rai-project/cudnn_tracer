package cudnn

// #cgo LDFLAGS: -lcudnn
// #cgo CXXFLAGS:  -std=c++14 -I${SRCDIR} -O3 -Wall -g
// #cgo CXXFLAGS: -Wno-sign-compare -Wno-unused-function
// #cgo CXXFLAGS: -I/usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib
import "C"
