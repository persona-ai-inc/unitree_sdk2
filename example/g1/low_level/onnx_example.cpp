#include <sstream>

// ONNX
#include <onnxruntime/onnxruntime_cxx_api.h>

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<int64_t> &v)
{
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int product(const std::vector<int64_t> &v)
{
    int total = 1;
    for (auto &i : v)
        total *= i;
    return total;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <onnx_model.onnx>\n", argv[0]);
        return -1;
    }
    const char *model_file = argv[1];
    printf("Loading: %s\n", model_file);

    // onnxruntime setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-example");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_file, session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // get input names and shapes
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<Ort::AllocatedStringPtr> input_names_allocated;
    std::vector<const char *> input_names;
    for (size_t i = 0; i < session.GetInputCount(); ++i)
    {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        input_names_allocated.push_back(std::move(input_name));
        input_names.push_back(input_names_allocated.back().get());
        Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
        const auto &input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shapes.push_back(input_tensor_info.GetShape());
    }

    // get output names and shapes
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<Ort::AllocatedStringPtr> output_names_allocated;
    std::vector<const char *> output_names;
    for (size_t i = 0; i < session.GetOutputCount(); ++i)
    {
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
        output_names_allocated.push_back(std::move(output_name));
        output_names.push_back(output_names_allocated.back().get());
        Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(i);
        const auto &output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_shapes.push_back(output_tensor_info.GetShape());
    }

    // print input and output names and shapes
    printf("Inputs:\n");
    for (size_t i = 0; i < input_names.size(); ++i)
    {
        printf(" - %s: %s\n", input_names[i], print_shape(input_shapes[i]).c_str());
    }
    printf("Outputs:\n");
    for (size_t i = 0; i < output_names.size(); ++i)
    {
        printf(" - %s: %s\n", output_names[i], print_shape(output_shapes[i]).c_str());
    }

    // only continue in simple cases
    if (input_names.size() != 1 || output_names.size() != 1)
    {
        printf("testing only onnx files with single input and single output node");
        return 0;
    }
    printf("\nRunning with random input...\n\n");

    auto input_shape = input_shapes[0];
    auto output_shape = output_shapes[0];
    int input_elements = product(input_shape);
    int output_elements = product(output_shape);

    // generate random numbers in the range [0, 255]
    auto gen = [&]()
    {
        return rand() % 255;
    };

    // Create a single tensor of random numbers
    std::vector<float> input_data(input_elements);
    std::generate(input_data.begin(), input_data.end(), gen);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              input_data.data(),
                                                              input_data.size(),
                                                              input_shape.data(),
                                                              input_shape.size());

    Ort::Value output_tensor;
    try
    {
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));
        std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr},
                                                             input_names.data(),
                                                             input_tensors.data(),
                                                             input_tensors.size(),
                                                             output_names.data(),
                                                             output_names.size());
        output_tensor = std::move(output_tensors[0]);
    }
    catch (const Ort::Exception &exception)
    {
        printf("error running model inference: %s\n", exception.what());
        exit(-1);
    }

    // the output numbers can be accessed like this
    const float *output_data = output_tensor.GetTensorData<float>();

    printf("%s:\n", input_names[0]);
    for (size_t i = 0; i < input_elements; ++i)
    {
        printf("%.2f ", input_data[i]);
    }
    printf("\n");

    printf("%s:\n", output_names[0]);
    for (size_t i = 0; i < output_elements; ++i)
    {
        printf("%.2f ", output_data[i]);
    }
    printf("\n");
}
