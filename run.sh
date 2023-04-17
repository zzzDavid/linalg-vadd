#!/bin/bash

llvm_libdir=`llvm-config --libdir`
echo "LLVM libdir: $llvm_libdir"

mlir-opt vadd_example.mlir \
    --convert-linalg-to-affine-loops \
    --lower-affine \
    --convert-scf-to-cf \
    --convert-memref-to-llvm \
    --convert-arith-to-llvm \
    --convert-func-to-llvm \
    --reconcile-unrealized-casts \
| mlir-cpu-runner -e main --O1 \
    --entry-point-result=void \
    --shared-libs=${llvm_libdir}/libmlir_c_runner_utils.so \
    --shared-libs=${llvm_libdir}/libmlir_runner_utils.so