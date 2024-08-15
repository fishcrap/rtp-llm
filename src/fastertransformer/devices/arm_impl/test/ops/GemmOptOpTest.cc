#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/arm_impl/test/ArmTestUtils.h"
#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/arm_impl/type_bf16/hie_bfloat16.hpp"
#include "src/fastertransformer/devices/utils/Timer.h"

#include <torch/torch.h>

using namespace std;
using namespace fastertransformer;

class ArmGemmOptOpTest: public DeviceTestBase {
public:
    void BasicGemmOP(size_t m, size_t n, size_t k);
    void BasicGemmOP_FP16(size_t m, size_t n, size_t k);
    void BatchGemmOP(size_t b, size_t m, size_t n, size_t k);
    void BatchGemmFP16OP(size_t b, size_t m, size_t n, size_t k, bool check_result=true);
    void TransposeBatchGemmOP(TransposeOperation op_a,
                              TransposeOperation op_b,
                              size_t             b,
                              size_t             m1,
                              size_t             k1,
                              size_t             k2,
                              size_t             n2,
                              size_t             m3,
                              size_t             n3);
    TimerRecorder timer_recorder_ = TimerRecorder();
    Timer timer_ = Timer();
};

void ArmGemmOptOpTest::BasicGemmOP_FP16(size_t m, size_t n, size_t k) {

    auto A_host = torch::rand({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::rand({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto A_device = createDeviceBuffer<half>(A_host);
    auto B_device = createDeviceBuffer<half>(B_host);

    GemmParams params{*A_device, *B_device};
    auto       C_device = device_->gemm(params);

    auto C_host = torch::matmul(A_host, B_host).to(torch::kHalf);
    auto A      = bufferToTensor(*A_device);
    auto B      = bufferToTensor(*B_device);
    auto C      = bufferToTensor(*C_device);

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}

void ArmGemmOptOpTest::BasicGemmOP(size_t m, size_t n, size_t k) {
    // auto A_host = torch::rand({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kBFloat16);
    // auto B_host = torch::rand({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kBFloat16);
    auto A_host = torch::rand({(int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::rand({(int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto A_device = createHostBuffer<float>({m, k}, tensorToBuffer(A_host, AllocationType::HOST)->data());
    auto B_device = createHostBuffer<float>({k, n}, tensorToBuffer(B_host, AllocationType::HOST)->data());
    // auto B_device = createHostBuffer<hie::bfloat16>({k, n}, tensorToBuffer(B_host, AllocationType::HOST)->data());

    GemmParams params{*A_device, *B_device};
    auto       C_device = device_->gemm_opt(params);

    auto C_host = torch::matmul(A_host, B_host).to(torch::kFloat);
    auto A      = bufferToTensor(*A_device);
    auto B      = bufferToTensor(*B_device);
    auto C      = bufferToTensor(*C_device);

    // std::cerr << "C_host: " << C_host << std::endl;
    // std::cerr << "C: " << C << std::endl;

    // auto C_host_slice = C_host.index({torch::indexing::Slice(0, 8), torch::indexing::Slice(0, 8)});
    // auto C_slice = C.index({torch::indexing::Slice(0, 8), torch::indexing::Slice(0, 8)});

    // std::cerr << "C(pytorch): \n" << C_host_slice << std::endl;
    // std::cerr << "C(rtp-llm): \n" << C_slice << std::endl;

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}

void ArmGemmOptOpTest::BatchGemmOP(size_t b, size_t m, size_t n, size_t k) {
    auto A_host = torch::rand({(int)b, (int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kFloat) - 0.5;
    auto B_host = torch::rand({(int)b, (int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kFloat) - 0.5;

    auto A_device = createDeviceBuffer<float>(A_host);
    auto B_device = createDeviceBuffer<float>(B_host);
    // auto A_device = createHostBuffer<float>({b, m, k}, tensorToBuffer(A_host, AllocationType::HOST)->data());
    // auto B_device = createHostBuffer<float>({b, k, n}, tensorToBuffer(B_host, AllocationType::HOST)->data());

    // std::cout << "A" << std::endl;
    // std::cout << A_host << std::endl;
    GemmParams params{*A_device, *B_device};

    timer_.reset();
    auto       C_device = device_->gemm_opt(params);
    timer_recorder_.record(std::string("BatchGemmFP16OP") + ", m=" + std::to_string(m) + ", n=" + std::to_string(n) + ", k=" + std::to_string(k), timer_.elapsed_nano());


    auto C_host = torch::matmul(A_host, B_host).to(torch::kFloat);
    auto A      = bufferToTensor(*A_device);
    auto B      = bufferToTensor(*B_device);
    auto C      = bufferToTensor(*C_device);

    auto C_host_slice = C_host.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, 8), torch::indexing::Slice(0, 8)});
    auto C_slice = C.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, 8), torch::indexing::Slice(0, 8)});
    // auto C_host_slice = C_host;
    // auto C_slice = C;

    // std::cout << "C(pytorch): \n" << C_host_slice << std::endl;
    // std::cout << "C(rtp-llm): \n" << C_slice << std::endl;

    // no nan
    ASSERT_TRUE((~torch::any(torch::isnan(C))).item<bool>());
    
    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}


void ArmGemmOptOpTest::BatchGemmFP16OP(size_t b, size_t m, size_t n, size_t k, bool check_result) {
    auto A_host = torch::rand({(int)b, (int)m, (int)k}, torch::Device(torch::kCPU)).to(torch::kHalf) - 0.5;
    auto B_host = torch::rand({(int)b, (int)k, (int)n}, torch::Device(torch::kCPU)).to(torch::kHalf) - 0.5;

    auto A_device = createDeviceBuffer<float16_t>(A_host);
    auto B_device = createDeviceBuffer<float16_t>(B_host);
    // auto A_device = createDeviceBuffer<float>(A_host.to(torch::kFloat));
    // auto B_device = createDeviceBuffer<float>(B_host.to(torch::kFloat));

    GemmParams params{*A_device, *B_device, nullopt, nullptr, DataType::TYPE_FP32};

    timer_.reset();
    auto       C_device = device_->gemm_opt(params);
    timer_recorder_.record(std::string("BatchGemmFP16OP") + ", m=" + std::to_string(m) + ", n=" + std::to_string(n) + ", k=" + std::to_string(k), timer_.elapsed_nano());
    if (check_result) {
        std::cout << "BatchGemmFP16OP" << ", m=" << m << ", n=" << n << ", k=" << k << std::endl;
    }

    auto A      = bufferToTensor(*A_device);
    auto B      = bufferToTensor(*B_device);
    auto C      = bufferToTensor(*C_device);

    // no nan
    ASSERT_TRUE((~torch::any(torch::isnan(C))).item<bool>());

    if (!check_result) {
        return;
    }

    auto C_host = torch::matmul(A_host, B_host).to(torch::kFloat);
    // auto C_host = torch::matmul(A_host, B_host);
    
    auto C_host_slice = C_host.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, 8), torch::indexing::Slice(0, 8)});
    auto C_slice = C.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, 8), torch::indexing::Slice(0, 8)});
    // auto C_host_slice = C_host;
    // auto C_slice = C;

    std::cout << "C(pytorch): \n" << C_host_slice << std::endl;
    std::cout << "C(rtp-llm): \n" << C_slice << std::endl;

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}

void ArmGemmOptOpTest::TransposeBatchGemmOP(TransposeOperation op_a,
                                         TransposeOperation op_b,
                                         size_t             b,
                                         size_t             m1,
                                         size_t             k1,
                                         size_t             k2,
                                         size_t             n2,
                                         size_t             m3,
                                         size_t             n3) {
    auto A_host = torch::rand({(int)b, (int)m1, (int)k1}, torch::Device(torch::kCPU)).to(torch::kFloat);
    auto B_host = torch::rand({(int)b, (int)k2, (int)n2}, torch::Device(torch::kCPU)).to(torch::kFloat);

    auto A_device = createDeviceBuffer<float>(A_host);
    auto B_device = createDeviceBuffer<float>(B_host);

    GemmParams params{*A_device, *B_device, nullopt, nullptr, DataType::TYPE_INVALID, op_a, op_b};
    auto       C_device = device_->gemm(params);

    if (op_a == TransposeOperation::TRANSPOSE) {
        A_host = A_host.transpose(1, 2);
    }
    if (op_b == TransposeOperation::TRANSPOSE) {
        B_host = B_host.transpose(1, 2);
    }
    auto C_host = torch::matmul(A_host, B_host).to(torch::kFloat);
    auto A      = bufferToTensor(*A_device);
    auto B      = bufferToTensor(*B_device);
    auto C      = bufferToTensor(*C_device);

    ASSERT_TRUE(torch::allclose(C, C_host, rtol_, atol_));
}

// TEST_F(ArmGemmOptOpTest, BasicGemmOpTest) {
//     BasicGemmOP(1, 1024, 2048);
//     BasicGemmOP(2, 1024, 2048);
//     // BasicGemmOP_FP16(2, 1024, 4);
//     BasicGemmOP(4, 1024, 2048);
//     BasicGemmOP(8, 1024, 2048);
//     BasicGemmOP(1024, 1024, 2048);
//     BasicGemmOP(4096, 1024, 2048);
// }

/*
TEST_F(ArmGemmOptOpTest, BatchGemmOpTest) {
    // BatchGemmOP(1, 2, 4, 1);

    // randomly generate b, m, n, k, and use BatchGemmOP to test
    // b, m, n, k in range [1, 4096]
    // int size = 3;
    // for (int i = 0; i < size; i++) {
    //     int b = 1;
    //     int m = rand() % 4096 + 1;
    //     int n = rand() % 4096 + 1;
    //     int k = rand() % 4096 + 1;
    //     BatchGemmOP(b, m, n, k);
    // }

    auto m_list = vector<int>{1, 14, 144, 256, 512, 2035};

    // BatchGemmOP(1, 1, 20, 20);
    // for (int i = 0; i < 10; i++) {
    //     BatchGemmOP(1, 1, rand() % 10 + 20, rand() % 10 + 20);
    // }

    // warm up
    // for (int i = 0; i < 10; i++) {
    //     BatchGemmOP(1, rand() % 20 + 1, rand() % 2000 + 2097, rand() % 2000 + 2097);
    // }
    
    for (int i = 0; i < 100; i++) {
        for (auto m : m_list) {
            BatchGemmOP(1, m, 2048, 2048);
            BatchGemmOP(1, m, 5504, 2048);
            BatchGemmOP(1, m, 5504, 2048);
            BatchGemmOP(1, m, 2048, 5504);
            BatchGemmOP(1, m, 6144, 2048);
        }
    }
#ifdef GEMM_DEBUG
    ArmCpuDevice::print_time();
#endif
    // BatchGemmOP(1, 1, 2048, 2048);
    // BatchGemmOP(1, 1, 5504, 2048);
    // BatchGemmOP(1, 1, 5504, 2048);
    // BatchGemmOP(1, 1, 2048, 5504);
    // BatchGemmOP(1, 1, 6144, 5504);
    

    // BatchGemmOP(1, 5, 4, 1);
    // BatchGemmOP(1, 8, 16, 1);
    // BatchGemmOP(1, 8, 16, 4);
    // BatchGemmOP(1, 8, 16, 8);
    // BatchGemmOP(2, 8, 16, 8);
    // BatchGemmOP(4, 8, 16, 8);
    // BatchGemmOP(8, 8, 8, 8);
}
*/

TEST_F(ArmGemmOptOpTest, BatchGemmFP16OpTest) {

    auto m_list = vector<int>{1, 14, 144, 256, 512, 2035};

    // for (int i = 0; i < 100; i++) {
    //     BatchGemmFP16OP(1, 10, 2048, 2048);
    // }
    // BatchGemmFP16OP(1, 10, 2048, 2048);
    // BatchGemmFP16OP(1, 10, 5504, 2048);
    // BatchGemmFP16OP(1, 10, 5504, 2048);
    // BatchGemmFP16OP(1, 10, 2048, 5504);
    // BatchGemmFP16OP(1, 10, 6144, 2048);

    // BatchGemmFP16OP(1, 1, 8, 4);
    // randomly generate b, m, n, k, and use BatchGemmOP to test
    // b, m, n, k in range [1, 4096]
    // int size = 100;
    // for (int i = 0; i < size; i++) {
    //     int b = 1;
    //     int m = rand() % 20 + 1;
    //     int n = rand() % 20 + 1;
    //     int k = rand() % 20 + 1;
    //     // BatchGemmOP(b, m, n, k);
    //     BatchGemmFP16OP(b, m, n, k);
    // }

    // int size = 100;
    // for (int i = 0; i < size; i++) {
    //     int b = 1;
    //     int m = rand() % 4096 + 1;
    //     int n = rand() % 4096 + 1;
    //     int k = rand() % 4096 + 1;
    //     BatchGemmFP16OP(b, m, n, k);
    // }

    
    // for (auto m : m_list) {
    //     BatchGemmFP16OP(1, m, 2048, 2048);
    //     BatchGemmFP16OP(1, m, 5504, 2048);
    //     BatchGemmFP16OP(1, m, 5504, 2048);
    //     BatchGemmFP16OP(1, m, 2048, 5504);
    //     BatchGemmFP16OP(1, m, 6144, 2048);
    // }

    for (int i = 0; i < 100; i++) {
        for (auto m : m_list) {
            BatchGemmFP16OP(1, m, 2048, 2048, false);
            BatchGemmFP16OP(1, m, 5504, 2048, false);
            BatchGemmFP16OP(1, m, 5504, 2048, false);
            BatchGemmFP16OP(1, m, 2048, 5504, false);
            BatchGemmFP16OP(1, m, 6144, 2048, false);
        }
    }
    timer_recorder_.print();
#ifdef GEMM_DEBUG
    ArmCpuDevice::print_time();
#endif
}



/*
TEST_F(ArmGemmOptOpTest, TransposeBatchGemmOpTest) {
    auto   tran = TransposeOperation::TRANSPOSE;
    auto   none = TransposeOperation::NONE;
    size_t b    = 128;
    size_t m    = 64;
    size_t n    = 8;
    size_t k    = 16;
    TransposeBatchGemmOP(none, none, b, m, k, k, n, m, n);
    TransposeBatchGemmOP(none, tran, b, m, k, n, k, m, n);
    TransposeBatchGemmOP(tran, tran, b, k, m, n, k, m, n);
    TransposeBatchGemmOP(tran, none, b, k, m, k, n, m, n);
}
*/