#!/bin/bash
binary="arm_gemm_opt_op_test"
binary_src_dir="src/fastertransformer/devices/arm_impl/test"
export TEST_WORKSPACE=maga_transformer
export TEST_SRCDIR=bazel-bin/$binary_src_dir/binary".runfiles"
export TEST_BINARY=$binary_src_dir/$binary
export LD_LIBRARY_PATH=/home/sxj/miniconda3/envs/rtp-llm/lib/:$LD_LIBRARY_PATH

echo "TEST_WORKSPACE: $TEST_WORKSPACE"
echo "TEST_SRCDIR: $TEST_SRCDIR"
echo "TEST_BINARY: $TEST_BINARY"