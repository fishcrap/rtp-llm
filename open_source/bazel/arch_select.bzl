# to wrapper target relate with different system config
# load("@pip_gpu_torch//:requirements.bzl", requirement_gpu="requirement")
# load("@pip_gpu_cuda12_torch//:requirements.bzl", requirement_gpu_cuda12="requirement")
# load("@pip_gpu_rocm_torch//:requirements.bzl", requirement_gpu_rocm="requirement")
# load("@pip_arm_torch//:requirements.bzl", requirement_arm="requirement")

def requirement(names):
    for name in names:
        native.py_library(
            name = name,
            deps = select({
                # "//:use_cuda12": [requirement_gpu_cuda12(name)],
                # "//:using_rocm": [requirement_gpu_rocm(name)],
                # "//conditions:default": [requirement_gpu(name)],
                "//:using_arm": [],
                # "//conditions:default": [requirement_arm(name)],
                # "//conditions:default": [],
            }),
            visibility = ["//visibility:public"],
        )

def th_transformer_so():
    native.alias(
        name = "th_transformer_so",
        actual = select({
            "//:use_cuda12": "//:th_transformer",
            "//conditions:default": "//:th_transformer"
        })
    )

def embedding_arpc_deps():
    native.alias(
        name = "embedding_arpc_deps",
        actual = "//maga_transformer/cpp/embedding_engine:embedding_engine_arpc_server_impl"
    )

def whl_deps():
    return select({
        "//:use_cuda12": ["torch==2.1.0+cu121"],
        "//:using_rocm": ["torch==2.1.2"],
        # "//:using_arm": ["torch==2.3.0"],
        "//conditions:default": ["torch==2.1.0+cu118"],
    })

def torch_deps():
    torch_version = "2.1_py310"
    deps = select({
        "@//:using_rocm": [
            "@torch_2.1_py310_rocm//:torch_api",
            "@torch_2.1_py310_rocm//:torch",
            "@torch_2.1_py310_rocm//:torch_libs",],
        "//:using_arm": [
            "@torch_2.3_py310_cpu_aarch64//:torch_api",
            "@torch_2.3_py310_cpu_aarch64//:torch",
            "@torch_2.3_py310_cpu_aarch64//:torch_libs",],
        "//conditions:default": [
            "@torch_" + torch_version + "//:torch_api",
            "@torch_" + torch_version + "//:torch",
            "@torch_" + torch_version + "//:torch_libs",]
        })
    return deps

def cutlass_kernels_interface():
    native.alias(
        name = "cutlass_kernels_interface",
        actual = select({
            "//:use_cuda12": "//src/fastertransformer/cutlass:cutlass_kernels_impl",
            "//conditions:default": "//src/fastertransformer/cutlass:cutlass_kernels_impl",
        })
    )

    native.alias(
        name = "cutlass_headers_interface",
        actual = select({
            "//:use_cuda12": "//src/fastertransformer/cutlass:cutlass_headers",
            "//conditions:default": "//src/fastertransformer/cutlass:cutlass_headers",
        })
    )
