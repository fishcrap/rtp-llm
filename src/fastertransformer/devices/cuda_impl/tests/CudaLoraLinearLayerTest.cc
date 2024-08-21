#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/base_tests/LoraLinearLayerTest.hpp"

using namespace std;
using namespace fastertransformer;

class CudaLoraLinearLayerTest: public LoraLinearLayerTest {};


TEST_F(CudaLoraLinearLayerTest, LoraLinearLayerTest) {
    std::vector<int> ms = {64, 1024};
    std::vector<int> ns = {64, 1024};
    std::vector<int> ks = {64, 1024};
    std::vector<DataType> input_dtypes = {DataType::TYPE_FP16};
    std::vector<DataType> lora_dtypes = {DataType::TYPE_FP16};
    for (auto m : ms) {
        for (auto n : ns) {
            for (auto k : ks) {
                for (auto input_dtype : input_dtypes) {
                    for (auto lora_dtype : lora_dtypes) {
                        loraLinearLayerTest({1}, m, n, k, {8}, input_dtype, lora_dtype);
                        loraLinearLayerTest({1000}, m, n, k, {128}, input_dtype, lora_dtype);
                        loraLinearLayerTest({1, 77}, m, n, k, {8, 64}, input_dtype, lora_dtype);
                        loraLinearLayerTest({77, 66, 1, 1, 1}, m, n, k, {8, 64, 128, 8, 8}, input_dtype, lora_dtype);
                        loraLinearLayerTest({100, 1, 1, 1, 1}, m, n, k, {8, 64, 128, 8, 8}, input_dtype, lora_dtype);
                        noLoraLinearLayerTest({1}, m, n, k, {8}, input_dtype, lora_dtype);
                        noLoraLinearLayerTest({1, 77}, m, n, k, {8, 64}, input_dtype, lora_dtype);
                        noLoraLinearLayerTest({77, 66, 1, 1, 1}, m, n, k, {8, 64, 128, 8, 8}, input_dtype, lora_dtype);
                        noLoraLinearLayerTest({100, 1, 1, 1, 1}, m, n, k, {8, 64, 128, 8, 8}, input_dtype, lora_dtype);
                    }
                }
            }
        }
    }
}
