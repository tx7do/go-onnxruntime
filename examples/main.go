package main

import (
	"context"
	"image"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/k0kubun/pp/v3"
	goTensor "gorgonia.org/tensor"

	"github.com/c3sr/config"
	"github.com/c3sr/dlframework"
	"github.com/c3sr/dlframework/framework/feature"
	"github.com/c3sr/dlframework/framework/options"
	c3srImage "github.com/c3sr/image"
	"github.com/c3sr/image/types"
	"github.com/c3sr/tracer"
	_ "github.com/c3sr/tracer/all"

	"github.com/tx7do/go-onnxruntime"
)

func normalizeImageCHW(in0 image.Image, mean []float32, scale []float32) ([]float32, error) {
	height := in0.Bounds().Dy()
	width := in0.Bounds().Dx()
	out := make([]float32, 3*height*width)
	switch in := in0.(type) {
	case *types.RGBImage:
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				offset := y*in.Stride + x*3
				rgb := in.Pix[offset : offset+3]
				r, g, b := rgb[0], rgb[1], rgb[2]
				out[0*width*height+y*width+x] = (float32(r) - mean[0]) / scale[0]
				out[1*width*height+y*width+x] = (float32(g) - mean[1]) / scale[1]
				out[2*width*height+y*width+x] = (float32(b) - mean[2]) / scale[2]
			}
		}
	case *types.BGRImage:
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				offset := y*in.Stride + x*3
				rgb := in.Pix[offset : offset+3]
				r, g, b := rgb[0], rgb[1], rgb[2]
				out[0*width*height+y*width+x] = (float32(b) - mean[0]) / scale[0]
				out[1*width*height+y*width+x] = (float32(g) - mean[1]) / scale[1]
				out[2*width*height+y*width+x] = (float32(r) - mean[2]) / scale[2]
			}
		}
	default:
		panic("unreachable")
	}
	return out, nil
}

var (
	model      = "torchvision_alexnet"
	graphFile  = "torchvision_alexnet.onnx"
	synsetFile = "synset.txt"
	imageFile  = "platypus.jpg"
	shape      = []int{1, 3, 224, 224}
	mean       = []float32{123.675, 116.280, 103.530}
	scale      = []float32{58.395, 57.120, 57.375}
)

func main() {
	defer tracer.Close()

	dir, _ := filepath.Abs("./_fixtures")
	dir = filepath.Join(dir, model)
	graph := filepath.Join(dir, graphFile)
	synset := filepath.Join(dir, synsetFile)

	imgDir, _ := filepath.Abs("./_fixtures")
	imgPath := filepath.Join(imgDir, imageFile)
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}

	batchSize := shape[0]
	height := shape[2]
	width := shape[3]

	var imgOpts []c3srImage.Option
	imgOpts = append(imgOpts, c3srImage.Mode(types.RGBMode))

	img, err := c3srImage.Read(r, imgOpts...)
	if err != nil {
		panic(err)
	}

	imgOpts = append(imgOpts, c3srImage.Resized(height, width))
	imgOpts = append(imgOpts, c3srImage.ResizeAlgorithm(types.ResizeAlgorithmLinear))
	resized, err := c3srImage.Resize(img, imgOpts...)

	imgFloats, err := normalizeImageCHW(resized, mean, scale)
	if err != nil {
		panic(err)
	}

	device := options.CPU_DEVICE

	ctx := context.Background()

	opts := options.New(options.Context(ctx),
		options.Graph([]byte(graph)),
		options.Device(device, 0),
		options.BatchSize(batchSize))

	opts.SetTraceLevel(tracer.FULL_TRACE)

	span, ctx := tracer.StartSpanFromContext(ctx, tracer.FULL_TRACE, "onnxruntime_batch")
	defer span.Finish()

	predictor, err := onnxruntime.New(
		ctx,
		options.WithOptions(opts),
	)

	if err != nil {
		panic(err)
	}

	defer predictor.Close()

	err = predictor.Predict(ctx, []goTensor.Tensor{
		goTensor.New(
			goTensor.Of(goTensor.Float32),
			goTensor.WithBacking(imgFloats),
			goTensor.WithShape(shape...),
		),
	})

	if err != nil {
		panic(err)
	}

	outputs, err := predictor.ReadPredictionOutput(ctx)
	if err != nil {
		panic(err)
	}

	output := outputs[0].Data().([]float32)

	labelsFileContent, err := os.ReadFile(synset)
	if err != nil {
		panic(err)
	}

	labels := strings.Split(string(labelsFileContent), "\n")

	featuresLen := len(output) / batchSize

	for i := 0; i < batchSize; i++ {
		rprobs := make([]*dlframework.Feature, featuresLen)
		for j := 0; j < featuresLen; j++ {
			rprobs[j] = feature.New(
				feature.ClassificationIndex(int32(j)),
				feature.ClassificationLabel(labels[j]),
				feature.Probability(output[i*featuresLen+j]),
			)
		}
		sort.Sort(dlframework.Features(rprobs))
		prediction := rprobs[0]
		_, _ = pp.Println(prediction.Probability, prediction.GetClassification().GetIndex(), prediction.GetClassification().GetLabel())
	}

}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
