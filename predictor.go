package onnxruntime

import "C"
import (
	"github.com/c3sr/dlframework/framework/options"
	"github.com/c3sr/tracer"
	"github.com/k0kubun/pp/v3"
	"github.com/opentracing/opentracing-go"
	"github.com/pkg/errors"
	"github.com/unknwon/com"
	"gorgonia.org/tensor"
	"runtime"
	"time"
	"unsafe"
)

/*
#include <stdlib.h>
#include "predictor.hpp"
*/
import "C"
import (
	"context"
)

type Predictor struct {
	ctx               C.ORT_PredictorContext
	options           *options.Options
	startingTimeSlice []int64
	endingTimeSlice   []int64
	ctxSlice          []context.Context
	predictSpanSlice  []opentracing.Span
}

func New(ctx context.Context, opts ...options.Option) (*Predictor, error) {
	defer PanicOnError()

	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_new")
	defer span.Finish()

	theOptions := options.New(opts...)
	modelFile := string(theOptions.Graph())
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}

	device := fromDevice(theOptions)
	if device == UnknownDeviceKind {
		return nil, errors.New("invalid device")
	}

	cModelFile := C.CString(modelFile)
	defer C.free(unsafe.Pointer(cModelFile))

	deviceID := theOptions.Devices()[0].ID()

	pred := &Predictor{
		ctx:     C.ORT_NewPredictor(cModelFile, C.ORT_DeviceKind(device), C.bool(theOptions.TraceLevel() >= tracer.FRAMEWORK_TRACE), C.int(deviceID)),
		options: theOptions,
	}

	runtime.SetFinalizer(pred, func(p *Predictor) {
		p.Close()
	})

	return pred, GetError()
}

func fromDevice(_ *options.Options) DeviceKind {
	device := CPUDeviceKind
	return device
}

func (p *Predictor) addInput(ten *tensor.Dense) {
	shape := make([]int64, len(ten.Shape()))
	for i, s := range ten.Shape() {
		shape[i] = int64(s)
	}
	var shapePtr *C.int64_t
	shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))

	C.ORT_AddInput(p.ctx, ten.Pointer(), shapePtr, C.int(len(shape)), fromType(ten))

	runtime.KeepAlive(shape)
}

func (p *Predictor) Predict(ctx context.Context, inputs []tensor.Tensor) error {
	defer PanicOnError()
	if len(inputs) < 1 {
		return errors.New("input nil or empty")
	}

	C.ORT_PredictorClear(p.ctx)

	for _, input := range inputs {
		dense, ok := input.(*tensor.Dense)
		if !ok {
			return errors.New("expecting a dense tensor")
		}
		p.addInput(dense)
	}

	predictSpan, ctx := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")

	if tracer.GetLevel() < tracer.FRAMEWORK_TRACE {
		defer predictSpan.Finish()
	}

	if tracer.GetLevel() >= tracer.FRAMEWORK_TRACE {
		p.predictSpanSlice = append(p.predictSpanSlice, predictSpan)
		p.ctxSlice = append(p.ctxSlice, ctx)
		p.startingTimeSlice = append(p.startingTimeSlice, time.Now().UnixNano())
	}

	C.ORT_PredictorRun(p.ctx)

	if tracer.GetLevel() >= tracer.FRAMEWORK_TRACE {
		p.endingTimeSlice = append(p.endingTimeSlice, time.Now().UnixNano())
	}

	return GetError()
}

func (p *Predictor) ReadPredictionOutput(ctx context.Context) ([]tensor.Tensor, error) {
	defer PanicOnError()

	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_predicted_output")
	defer span.Finish()

	C.ORT_PredictorConvertOutput(p.ctx)

	cNumOutputs := int(C.ORT_PredictorNumOutputs(p.ctx))

	if cNumOutputs == 0 {
		return nil, errors.New("zero number of tensors")
	}

	res := make([]tensor.Tensor, cNumOutputs)

	for i := 0; i < cNumOutputs; i++ {
		cPredictions := C.ORT_PredictorGetOutput(p.ctx, C.int(i))
		// The allocated memory will be deleted when destructor of predictor in c++ is called
		res[i] = ortValueToTensor(cPredictions)
	}

	if err := GetError(); err != nil {
		return nil, err
	}

	return res, nil
}

func (p *Predictor) Close() {
	if p == nil {
		return
	}

	if p.ctx != nil && p.options.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		C.ORT_EndProfiling(p.ctx)
		startTime := int64(C.ORT_ProfilingGetStartTime(p.ctx))

		profBuffer, err := p.ReadProfile()
		if err != nil {
			_, _ = pp.Println(err)
			return
		}

		t, err := NewTrace(profBuffer, startTime)
		if err != nil {
			//log.Panic("Predictor.Close NewTrace: ", err)
			return
		}

		tSlice, err := SplitTrace(t, p.startingTimeSlice, p.endingTimeSlice)
		if err != nil {
			log.Panic("Predictor.Close SplitTrace: ", err)
		}

		for batchNum, ctx := range p.ctxSlice {
			_ = tSlice[batchNum].Publish(ctx, tracer.FRAMEWORK_TRACE)
			p.predictSpanSlice[batchNum].FinishWithOptions(opentracing.FinishOptions{
				FinishTime: time.Unix(0, p.endingTimeSlice[batchNum]),
			})
		}

		// clear records
		p.startingTimeSlice = nil
		p.endingTimeSlice = nil
		p.ctxSlice = nil
		p.predictSpanSlice = nil
	}

	if p.ctx != nil {
		C.ORT_PredictorDelete(p.ctx)
	}
	p.ctx = nil

}
