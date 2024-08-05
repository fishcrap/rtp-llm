#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include <cstring>
#include "autil/StringUtil.h"
#include "type_bf16/hie_bfloat16.hpp"
#include "gemm_opt/ArmGemmKernel.h"

namespace fastertransformer {

/// @brief   basic gemm ops
/// @details D = alpha * op(A) * op(B) + beta * C
///          A [b, ..., m, k]
///          B [b, ..., k, n]
///          C [b, ..., m, n]
BufferPtr ArmCpuDevice::gemm_opt(const GemmParams& params) {

    params.check();

    std::vector<size_t> Ashape;
    std::vector<size_t> Bshape;
    std::vector<size_t> Dshape;

    size_t dim;
    size_t batch_size;
    size_t m;
    size_t k;
    size_t n;
    size_t lda;

    Ashape = params.A.shape();
    Bshape = params.B.shape();

    dim        = params.A.dim();
    batch_size = std::accumulate(Ashape.begin(), Ashape.end() - 2, (size_t)1, std::multiplies<size_t>());

    if (params.transA == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Ashape.end() - 1, Ashape.end() - 2);
    }

    if (params.transB == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Bshape.end() - 1, Bshape.end() - 2);
    }

    m = Ashape[dim - 2];
    k = Ashape[dim - 1];
    n = Bshape[dim - 1];
    lda = k;

    auto data_type = params.compute_type == DataType::TYPE_INVALID ? params.A.type() : params.compute_type;
    if (data_type != params.A.type()) {
        std::cout << "[Warning] GEMM compute type differs from input type. Not supported" << std::endl;
        data_type = params.A.type();
    }

    Dshape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
    Dshape.insert(Dshape.end(), {m, n});


    BufferPtr output;
    if (params.D) {
        output = params.D;
        RUNTIME_ASSERT_OP_ARG((data_type == params.D->type()) && (Dshape == params.D->shape()),
                              "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                              data_type,
                              autil::StringUtil::toString(Dshape).c_str(),
                              params.D->debugString().c_str());
    } else {
        output = allocateBuffer({data_type, Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }

    bool is_transA = params.transA == TransposeOperation::TRANSPOSE;
    bool is_transB = params.transB == TransposeOperation::TRANSPOSE;

    // allocate a temp workspace to pack input fp32->bf16
    size_t k_pack = std::ceil(k / 8.0) * 8;
    size_t m_aligned = m + m % 2;
    std::vector<size_t> workspace_shape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
    workspace_shape.insert(workspace_shape.end(), {m_aligned, k_pack});
    BufferPtr workspace = allocateBuffer({DataType::TYPE_BF16, workspace_shape, AllocationType::DEVICE}, {"gemm_workspace"});

    BufferPtr weight_workspace;
    const Buffer *weight_workspace_ptr = nullptr;

    if (params.B.type() == DataType::TYPE_FP32) {
        // allocate a temp workspace to pack weight fp32->bf16
        size_t width = k_pack * 2;
        size_t height = n / 2 + n % 2;
        std::vector<size_t> weight_workspace_shape = std::vector<size_t>(Bshape.begin(), Bshape.end() - 2);
        weight_workspace_shape.insert(weight_workspace_shape.end(), {height, width});
        weight_workspace = allocateBuffer({DataType::TYPE_BF16, weight_workspace_shape, AllocationType::DEVICE}, {"gemm_weight_workspace"});
        weight_workspace_ptr = weight_workspace.get();
        // pack weight
        for (size_t batch = 0; batch < batch_size; ++batch) {
            float* B_fp32_ptr = reinterpret_cast<float*>(params.B.dataWithOffset(batch * k * n));
            hie::bfloat16* weight_workspace_cur_ptr = reinterpret_cast<hie::bfloat16*>(weight_workspace->dataWithOffset(batch * height * width));
            gemm_kernel_.gemm_pack_weight_FP32toBF16_arm(n, k, k_pack, B_fp32_ptr, weight_workspace_cur_ptr);
        }
    } else if(params.B.type() == DataType::TYPE_BF16) {
        weight_workspace_ptr = &(params.B);
    } else {
        std::cerr << "Unsupported data type for B" << std::endl;
        return nullptr;
    }

    for (size_t batch = 0; batch < batch_size; ++batch) {
        float *A_fp32_ptr = reinterpret_cast<float*>(params.A.dataWithOffset(batch * m * k));
        hie::bfloat16* B_bf16_ptr = reinterpret_cast<hie::bfloat16*>(weight_workspace_ptr->dataWithOffset(batch * k * n));
        float *C_fp32_ptr = reinterpret_cast<float*>(output->dataWithOffset(batch * m * n));
        gemm_kernel_.gemm_kernel_arm(m, n, k, lda, A_fp32_ptr, B_bf16_ptr, C_fp32_ptr, nullptr, 0, workspace->data());
    }


    return std::move(output);
}

}  // namespace fastertransformer
