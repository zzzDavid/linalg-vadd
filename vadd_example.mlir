// vector_add.mlir
module {
  memref.global "private" @gbarray1 : memref<4xf32> = dense<[1.0, 2.0, 3.0, 4.0]> 
  memref.global "private" @gbarray2 : memref<4xf32> = dense<[5.0, 6.0, 7.0, 8.0]>
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() -> () {
    %array1 = memref.get_global @gbarray1 : memref<4xf32>
    %array2 = memref.get_global @gbarray2 : memref<4xf32>
    %result = memref.alloc() : memref<4xf32>

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

    %casted_result = memref.cast %result : memref<4xf32> to memref<*xf32>
    call @printMemrefF32(%casted_result) : (memref<*xf32>) -> ()
    return
  }
}
