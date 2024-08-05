load("@rules_python//python:pip.bzl", "pip_install")

def pip_deps():
    pip_install(
        name = "pip_gpu_torch",
        requirements = ["//open_source/deps:requirements_torch_gpu.txt", "//open_source/deps:requirements_base.txt"],
        python_interpreter = "/home/sxj/.conda/envs/rtp-llm/bin/python",
        extra_pip_args = [
            "--index-url=https://mirrors.aliyun.com/pypi/simple/",
        ],
        timeout=12000,
    )

    pip_install(
        name = "pip_gpu_cuda12_torch",
        requirements = ["//open_source/deps:requirements_torch_gpu_cuda12.txt", "//open_source/deps:requirements_base.txt"],
        python_interpreter = "/home/sxj/.conda/envs/rtp-llm/bin/python",
        extra_pip_args = [
            "--index-url=https://mirrors.aliyun.com/pypi/simple/",
        ],
        timeout=12000,
    )

    pip_install(
        name = "pip_gpu_rocm_torch",
        requirements = ["//open_source/deps:requirements_rocm.txt", "//open_source/deps:requirements_base.txt"],
        python_interpreter = "/home/sxj/.conda/envs/rtp-llm/bin/python",
        timeout=12000,
    )

    # pip_install(
    #     name = "pip_arm_torch",
    #     requirements = ["//open_source/deps:requirements_arm.txt", "//open_source/deps:requirements_base.txt"],
    #     python_interpreter = "/home/sxj/.conda/envs/rtp-llm/bin/python",
    #     extra_pip_args = [
    #         "--index-url=https://mirrors.aliyun.com/pypi/simple/",
    #     ],
    #     timeout=12000,
    # )
