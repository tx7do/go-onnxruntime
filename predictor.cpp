#include "error.hpp"
#include "predictor.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>

#include <onnxruntime_cxx_api.h>

#include "providers.hpp"

using std::string;

/* Description: The structure to handle the predictor for onnxruntime
 * Note: Call ConvertOutput before you want to read the outputs
 */
struct Predictor
{
  Predictor(const string &model_file, ORT_DeviceKind device, bool enable_trace, int device_id);
  ~Predictor();

  void Predict(void);
  void ConvertOutput(void);
  void AddOutput(Ort::Value &);
  void Clear(void);
  void *ConvertTensorToPointer(Ort::Value &, size_t);
  void EndProfiling(void);

  struct Onnxruntime_Env
  {
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    /* Description: Follow the sample given in onnxruntime to initialize the environment
     * Referenced: https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
     */
    Onnxruntime_Env(ORT_DeviceKind device, bool enable_trace, int device_id) : env_(ORT_LOGGING_LEVEL_ERROR, "ort_predict")
    {
      // Initialize environment, could use ORT_LOGGING_LEVEL_VERBOSE to get more information
      // NOTE: Only one instance of env can exist at any point in time

      // enable profiling, the argument is the prefix you want for the file
      if (enable_trace)
        session_options_.EnableProfiling("onnxruntime");

#ifdef USE_CUDA
      if (device == CUDA_DEVICE_KIND)
      {
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, device_id /* device id */);
      }
#endif

      // Sets graph optimization level
      // Available levels are
      // ORT_DISABLE_ALL -> To disable all optimizations
      // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
      // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
      // ORT_ENABLE_ALL -> To Enable All possible optimizations
      session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }
  } _ort_env;
  // Order matters when using member initializer lists
  int64_t _profile_start;
  bool _enable_trace;

  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions _allocator;

  string _profile_filename;

  std::vector<std::string> _str_input_names;
  std::vector<std::string> _str_output_names;

  std::vector<const char*> _input_names;
  std::vector<const char*> _output_names;

  std::vector<Ort::Value> _input_tensors;
  std::vector<Ort::Value> _output_tensors;

  std::vector<ORT_Value> _converted_output;
};

/* Description: Follow the sample given in onnxruntime to initialize the predictor
 * Referenced: https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
 */
Predictor::Predictor(const string &model_file, ORT_DeviceKind device, bool enable_trace, int device_id)
    : _ort_env(device, enable_trace, device_id),
      session_(_ort_env.env_, model_file.c_str(), _ort_env.session_options_),
      _enable_trace(enable_trace)
{

  // get input info
  size_t num_input_nodes = session_.GetInputCount();

  // get input node names and dimensions
  for (size_t i = 0; i < num_input_nodes; i++)
  {
    auto inputNodeName = session_.GetInputNameAllocated(i, _allocator);
     std::string inputName = inputNodeName.get();
    // std::cout << "Input Name: " << inputName << std::endl;
    _str_input_names.push_back(inputName);
    _input_names.push_back(_str_input_names.back().c_str());
  }

  // get output info
  size_t num_output_nodes = session_.GetOutputCount();

  // get output node names
  for (size_t i = 0; i < num_output_nodes; i++)
  {
    auto outputNodeName = session_.GetOutputNameAllocated(i, _allocator);
    std::string outputName = outputNodeName.get();
    // std::cout << "Output Name: " << outputName << std::endl;
    _str_output_names.push_back(outputName);
    _output_names.push_back(_str_output_names.back().c_str());
  }
}

/* Description: clean up the predictor for next prediction */
void Predictor::Clear()
{
  for (size_t i = 0; i < _converted_output.size(); i++)
  {
    free(_converted_output[i].data_ptr);
    free((void *)_converted_output[i].shape_ptr);
    _converted_output[i].data_ptr = nullptr;
    _converted_output[i].shape_ptr = nullptr;
  }
  _converted_output.clear();
  _input_tensors.clear();
}

/* Description: Destructor of the predictor to clean up dynamic allocated memory */
Predictor::~Predictor()
{
  Clear();
}

/* Description: Do the inference in onnxruntime */
void Predictor::Predict(void)
{
  // check invalid dims size
  if (_input_tensors.size() != _input_names.size())
  {
    throw std::runtime_error(std::string("Invalid number of input tensor in Predictor::Predict."));
  }

  _output_tensors = session_.Run(Ort::RunOptions{nullptr}, _input_names.data(), _input_tensors.data(),
                         _input_tensors.size(), _output_names.data(), _output_names.size());
}

/* Description: Convert Ort::Value to an array pointed by the pointer */
void *Predictor::ConvertTensorToPointer(Ort::Value &value, size_t size)
{
  void *res = nullptr;
  switch (value.GetTensorTypeAndShapeInfo().GetElementType())
  {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    throw std::runtime_error(std::string("undefined data type detected in ConvertTensorToPointer."));
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    res = (void *)malloc(sizeof(float) * size);
    memcpy(res, value.GetTensorMutableData<float>(), sizeof(float) * size);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    res = (void *)malloc(sizeof(uint8_t) * size);
    memcpy(res, value.GetTensorMutableData<uint8_t>(), sizeof(uint8_t) * size);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    res = (void *)malloc(sizeof(int8_t) * size);
    memcpy(res, value.GetTensorMutableData<int8_t>(), sizeof(int8_t) * size);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    res = (void *)malloc(sizeof(uint16_t) * size);
    memcpy(res, value.GetTensorMutableData<uint16_t>(), sizeof(uint16_t) * size);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    res = (void *)malloc(sizeof(int16_t) * size);
    memcpy(res, value.GetTensorMutableData<int16_t>(), sizeof(int16_t) * size);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    res = (void *)malloc(sizeof(int32_t) * size);
    memcpy(res, value.GetTensorMutableData<int32_t>(), sizeof(int32_t) * size);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    res = (void *)malloc(sizeof(int64_t) * size);
    memcpy(res, value.GetTensorMutableData<int64_t>(), sizeof(int64_t) * size);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    res = (void *)malloc(sizeof(bool) * size);
    memcpy(res, value.GetTensorMutableData<bool>(), sizeof(bool) * size);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    res = (void *)malloc(sizeof(double) * size);
    memcpy(res, value.GetTensorMutableData<double>(), sizeof(double) * size);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    res = (void *)malloc(sizeof(uint32_t) * size);
    memcpy(res, value.GetTensorMutableData<uint32_t>(), sizeof(uint32_t) * size);
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    res = (void *)malloc(sizeof(uint64_t) * size);
    memcpy(res, value.GetTensorMutableData<uint64_t>(), sizeof(uint64_t) * size);
    break;
  default: // c++: FLOAT16; onnxruntime: COMPLEX64, COMPLEX128, BFLOAT16; TODO: Implement String method
    throw std::runtime_error(std::string("unsupported data type detected in Predictor::ConvertTensorToPointer."));
  }
  return res;
}

/* Description: The helper function when calling ConvertOutput for converting all outputs into array form
 *              Since Ort::Value can be a tensor, a map or a sequence, we need to decompose it by recursion
 */
void Predictor::AddOutput(Ort::Value &value)
{
  // base case
  if (value.IsTensor())
  {
    auto tensor_info = value.GetTensorTypeAndShapeInfo();
    auto dims = tensor_info.GetShape();
    int64_t *shapes = (int64_t *)malloc(sizeof(int64_t) * dims.size());
    size_t size = 1;
    for (size_t i = 0; i < dims.size(); i++)
    {
      size *= dims[i];
      shapes[i] = dims[i];
    }
    _converted_output.push_back(ORT_Value{
        .otype = tensor_info.GetElementType(),
        .data_ptr = ConvertTensorToPointer(value, size),
        .shape_ptr = shapes,
        .shape_len = dims.size()});
    return;
  }

  // need to be decomposed, it is a map or a sequence, both can be done in the same way
  size_t length = value.GetCount();

  for (size_t i = 0; i < length; i++)
  {
    auto cur_val = value.GetValue(static_cast<int>(i), _allocator);
    AddOutput(cur_val);
  }
}

/* Description: The function need to be called before reading outputs from Go */
void Predictor::ConvertOutput(void)
{
  for (size_t i = 0; i < _output_tensors.size(); i++)
  {
    AddOutput(_output_tensors[i]);
  }
}

void Predictor::EndProfiling(void)
{
  if (_enable_trace)
    _profile_filename = session_.EndProfilingAllocated(_allocator).get();
}

void ORT_EndProfiling(ORT_PredictorContext pred)
{
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorClear."));
  }
  predictor->EndProfiling();
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}

/* Description: The interface for Go to create a new predictor */
ORT_PredictorContext ORT_NewPredictor(const char *model_file, ORT_DeviceKind device, bool enable_trace, int device_id)
{
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  const auto ctx = new Predictor(model_file, device, enable_trace, device_id);
  return (ORT_PredictorContext)ctx;
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, (ORT_PredictorContext) nullptr);
}

/* Description: The interface for Go to clear the predictor */
void ORT_PredictorClear(ORT_PredictorContext pred)
{
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorClear."));
  }
  predictor->Clear();
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}

/* Description: The interface for Go to do inference */
void ORT_PredictorRun(ORT_PredictorContext pred)
{
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorRun."));
  }
  predictor->Predict();
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}

/* Description: The interface for Go to convert outputs before reading outputs */
void ORT_PredictorConvertOutput(ORT_PredictorContext pred)
{
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorConvertOutput."));
  }

  predictor->ConvertOutput();

  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}

/* Description: The interface for Go to know the number of converted outputs */
int ORT_PredictorNumOutputs(ORT_PredictorContext pred)
{
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorNumOutputs."));
  }
  return (int)((predictor->_converted_output).size());
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, 0);
}

/* Description: The interface for Go to get the number of converted outputs */
ORT_Value ORT_PredictorGetOutput(ORT_PredictorContext pred, int index)
{
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorGetOutput."));
  }

  return (predictor->_converted_output)[index];

  END_HANDLE_ORT_ERRORS(ORT_GlobalError, ORT_Value{});
}

/* Description: The interface for Go to delete the dynamic allocated predictor
 *              The destructor for the predictor will be called when deleting the predictor
 */
void ORT_PredictorDelete(ORT_PredictorContext pred)
{
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorDelete."));
  }

  if (predictor->_profile_filename != "")
    remove((predictor->_profile_filename).c_str());

  delete predictor;
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}

/* Description: The interface for Go to read the profile in framework level from onnxruntime */
char *ORT_ProfilingRead(ORT_PredictorContext pred)
{
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_ProfilingRead."));
  }

  std::stringstream ss;
  std::ifstream in(predictor->_profile_filename);
  ss << in.rdbuf();
  return strdup(ss.str().c_str());

  END_HANDLE_ORT_ERRORS(ORT_GlobalError, strdup(""));
}

/* Description: High resolution clock might not be what we want
 *              so get the offset
 */
static int64_t GetOffset(void)
{
  using namespace std::chrono;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  system_clock::time_point t2 = system_clock::now();
  system_clock::time_point t3 = system_clock::now();
  high_resolution_clock::time_point t4 = high_resolution_clock::now();
  return (static_cast<int64_t>(duration_cast<nanoseconds>(t2.time_since_epoch()).count()) - static_cast<int64_t>(duration_cast<nanoseconds>(t1.time_since_epoch()).count()) + static_cast<int64_t>(duration_cast<nanoseconds>(t3.time_since_epoch()).count()) - static_cast<int64_t>(duration_cast<nanoseconds>(t4.time_since_epoch()).count())) / 2;
}

/* Description: The interface for Go to get the start time of the profiler */
int64_t ORT_ProfilingGetStartTime(ORT_PredictorContext pred)
{
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_ProfilingGetStartTime."));
  }

  return static_cast<int64_t>(predictor->session_.GetProfilingStartTimeNs()) + GetOffset();

  END_HANDLE_ORT_ERRORS(ORT_GlobalError, -1);
}

/* Description: The interface for Go to add inputs into the predictor */
void ORT_AddInput(ORT_PredictorContext pred, void *input, int64_t *dimensions,
                  int n_dim, ONNXTensorElementDataType dType)
{
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr)
  {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_AddInput."));
  }
  std::vector<int64_t> dims;
  dims.assign(dimensions, dimensions + n_dim);
  size_t size = 1;
  for (int i = 0; i < n_dim; i++)
    size *= dims[i];

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  switch (dType)
  {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    throw std::runtime_error(std::string("undefined data type detected in ORT_AddInput."));
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    (predictor->_input_tensors).emplace_back(Ort::Value::CreateTensor<float>(memory_info, static_cast<float *>(input), size, dims.data(), dims.size()));
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    (predictor->_input_tensors).emplace_back(Ort::Value::CreateTensor<uint8_t>(memory_info, static_cast<uint8_t *>(input), size, dims.data(), dims.size()));
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    (predictor->_input_tensors).emplace_back(Ort::Value::CreateTensor<int8_t>(memory_info, static_cast<int8_t *>(input), size, dims.data(), dims.size()));
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    (predictor->_input_tensors).emplace_back(Ort::Value::CreateTensor<uint16_t>(memory_info, static_cast<uint16_t *>(input), size, dims.data(), dims.size()));
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    (predictor->_input_tensors).emplace_back(Ort::Value::CreateTensor<int16_t>(memory_info, static_cast<int16_t *>(input), size, dims.data(), dims.size()));
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    (predictor->_input_tensors).emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, static_cast<int32_t *>(input), size, dims.data(), dims.size()));
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    (predictor->_input_tensors).emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, static_cast<int64_t *>(input), size, dims.data(), dims.size()));
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    (predictor->_input_tensors).emplace_back(Ort::Value::CreateTensor<bool>(memory_info, static_cast<bool *>(input), size, dims.data(), dims.size()));
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    (predictor->_input_tensors).emplace_back(Ort::Value::CreateTensor<double>(memory_info, static_cast<double *>(input), size, dims.data(), dims.size()));
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    (predictor->_input_tensors).emplace_back(Ort::Value::CreateTensor<uint32_t>(memory_info, static_cast<uint32_t *>(input), size, dims.data(), dims.size()));
    break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    (predictor->_input_tensors).emplace_back(Ort::Value::CreateTensor<uint64_t>(memory_info, static_cast<uint64_t *>(input), size, dims.data(), dims.size()));
    break;
  default: // c++: FLOAT16; onnxruntime: COMPLEX64, COMPLEX128, BFLOAT16; TODO: Implement String method
    throw std::runtime_error(std::string("unsupported data type detected in ORT_AddInput."));
  }
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}
