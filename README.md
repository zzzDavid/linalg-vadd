# Vector Addition with Linalg

This README explains the provided MLIR (Multi-Level Intermediate Representation) example that performs vector addition on two statically initialized arrays. MLIR is a compiler infrastructure that aims to address software fragmentation, improve compilation time, and assist in code generation, among other things.

## Code Explanation
### Global Memory and Initialization
Two arrays are declared as global memory with private visibility, and they are initialized with floating-point values.

```mlir
memref.global "private" @gbarray1 : memref<4xf32> = dense<[1.0, 2.0, 3.0, 4.0]> 
memref.global "private" @gbarray2 : memref<4xf32> = dense<[5.0, 6.0, 7.0, 8.0]>
```

### Helper Function Declaration

A helper function `@printMemrefF32` is declared to print the elements of a memref (memory reference) containing floating-point values.
```mlir
func.func private @printMemrefF32(memref<*xf32>)
```
### Main Function

The main function is declared as `func.func @main() -> ()`.

### Loading Global Arrays

The global arrays are loaded into the main function:
```mlir
%array1 = memref.get_global @gbarray1 : memref<4xf32>
%array2 = memref.get_global @gbarray2 : memref<4xf32>
```
### Allocating Result Memory

Memory is allocated for the result of the vector addition:
```mlir
%result = memref.alloc() : memref<4xf32>
```
### Vector Addition

A linalg.generic operation is used to perform the vector addition. It takes the input arrays, `%array1` and `%array2`, and the output array, `%result`, as arguments. The indexing maps and iterator types are specified for parallel processing.
```mlir
linalg.generic
{args_in = 2, args_out = 1, indexing_maps = [
    affine_map<(i) -> (i)>,
    affine_map<(i) -> (i)>,
    affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]}
    ins(%array1, %array2: memref<4xf32>, memref<4xf32>)
    outs(%result: memref<4xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
}
```
### Printing the Result

The result memref is cast to match the input type of the `@printMemrefF32` function, and the function is called to print the vector addition result:
```mlir
%casted_result = memref.cast %result : memref<4xf32> to memref<*xf32>
call @printMemrefF32(%casted_result) : (memref<*xf32>) -> ()
```

## Run Script Explanation
This script uses several MLIR optimization passes and utilities to compile the MLIR code into LLVM and then runs it using the `mlir-cpu-runner`.

```bash
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

```

Script Explanation:

1. The script starts with the shebang #!/bin/bash, which specifies that it should be executed using the Bash shell.
2. The LLVM library directory is retrieved using llvm-config --libdir and stored in the variable llvm_libdir.
3. The MLIR optimization tool, mlir-opt, processes the vadd_example.mlir file with the following optimization passes:
- `--convert-linalg-to-affine-loops`: Converts Linalg operations to loops using the Affine dialect.
- `--lower-affine`: Lowers the Affine dialect to the SCF dialect (Static Control Flow).
- `--convert-scf-to-cf`: Lowers the SCF dialect to the CFG dialect (Control Flow Graph).
- `--convert-memref-to-llvm`: Converts memref types and operations to the LLVM dialect.
- `--convert-arith-to-llvm`: Converts arithmetic operations to the LLVM dialect.
- `--convert-func-to-llvm`: Converts functions and their types to the LLVM dialect.
- `--reconcile-unrealized-casts`: Reconciles unrealized cast operations.
4. The output of the mlir-opt command is passed to the mlir-cpu-runner utility, which compiles and runs the LLVM code on the CPU:
- `-e main`: Specifies the entry point function, which is main.
- `--O1`: Enables level 1 optimizations.
- `--entry-point-result=void`: Indicates that the entry point function returns void.
- `--shared-libs=${llvm_libdir}/libmlir_c_runner_utils.so`: Links the C runner utilities library.
- `--shared-libs=${llvm_libdir}/libmlir_runner_utils.so`: Links the general runner utilities library.

After running the script, the vector addition example should be compiled and executed on the CPU.