#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/Scheduler.h"
#include "gemm_opt/ArmGemmKernel.h"
#include "src/fastertransformer/devices/utils/Timer.h"

namespace fastertransformer {

class ArmCpuDevice : public DeviceBase {
public:
    ArmCpuDevice(const DeviceInitParams& params);
    ~ArmCpuDevice();

public:
    DeviceProperties getDeviceProperties() override;
    IAllocator* getAllocator() override { return allocator_.get(); }
    IAllocator* getHostAllocator() override { return allocator_.get(); }

public:
    void copy(const CopyParams& params);
    LayernormOutput layernorm(const LayernormParams& params);
    BufferPtr gemm(const GemmParams& params);
    BufferPtr gemm_opt(const GemmParams& params);
    GroupedGemmOutput groupedGemm(const GroupedGemmParams& params);
    BufferPtr embeddingLookup(const EmbeddingLookupParams& params);
    void activation(const ActivationParams& params);
    BufferPtr softmax(const SoftmaxParams& params);
    AttentionModuleOutput contextAttention(const AttentionModuleParams& params);
    AttentionModuleOutput decoderSelfAttention(const AttentionModuleParams& params);
    void sampleGreedy(const GreedyParams& params);
    void sampleBeamSearch(const BeamSearchParams& params);
    void broadcast(const BroadcastParams& params);
    void allReduceSum(const AllReduceParams& params);
#ifdef GEMM_DEBUG
    static void print_time();
#endif

private:
    std::unique_ptr<IAllocator> allocator_;
    arm_compute::DataType getAclDataType(DataType type);
    GemmKernel gemm_kernel_;

#ifdef GEMM_DEBUG
    static TimerRecorder timer_recorder_;
#endif
};

} // namespace fastertransformer

