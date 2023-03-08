func.func @scatter_and_scatter_1x4x2x1_i32() {
  %operand = util.unfoldable_constant dense<[[[[1], [5]], [[2], [5]], [[3], [6]], [[4], [4]]]]> : tensor<1x4x2x1xi32>
  %source = util.unfoldable_constant dense<[[[[5], [6]], [[7], [8]]]]> : tensor<1x2x2x1xi32>
  %init_value = util.unfoldable_constant dense<0> : tensor<i32>
  %extracted = tensor.extract %init_value[] : tensor<i32>
  %3 = tensor.empty() : tensor<1x4x2x1xi32>
  %extracted_3 = tensor.extract %init_value[] : tensor<i32>
  %4 = linalg.fill ins(%extracted_3 : i32) outs(%3 : tensor<1x4x2x1xi32>) -> tensor<1x4x2x1xi32>
  %result = iree_linalg_ext.select_and_scatter window_dimensions = dense<[1, 3, 1, 1]> : tensor<4xi64> window_strides = dense<[1, 2, 1, 1]> : tensor<4xi64> padding = dense<[[0, 0], [0, 1], [0, 0], [0, 0]]> : tensor<4x2xi64> ins(%operand, %source, %extracted : tensor<1x4x2x1xi32>, tensor<1x2x2x1xi32>, i32) outs(%4 : tensor<1x4x2x1xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %6 = arith.cmpi sge, %arg1, %arg0 : i32
    iree_linalg_ext.yield %6 : i1
  }, {
  ^bb0(%arg0: i32, %arg1: i32):
    %6 = arith.addi %arg1, %arg0 : i32
    iree_linalg_ext.yield %6 : i32
  } -> tensor<1x4x2x1xi32>
  check.expect_eq_const(%result, dense<[[[[0], [0]], [[0], [0]], [[5], [14]], [[7], [0]]]]> : tensor<1x4x2x1xi32>) : tensor<1x4x2x1xi32>
  return
}

func.func @scatter_and_scatter_1x3x3x1_f32() {
  %operand = util.unfoldable_constant dense<[[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]], [[7.0], [8.0], [9.0]]]]> : tensor<1x3x3x1xf32>
  %source = util.unfoldable_constant dense<[[[[10.0], [11.0]], [[12.0], [13.0]]]]> : tensor<1x2x2x1xf32>
  %init_value = util.unfoldable_constant dense<0.0> : tensor<f32>
  %extracted = tensor.extract %init_value[] : tensor<f32>
  %3 = tensor.empty() : tensor<1x3x3x1xf32>
  %extracted_3 = tensor.extract %init_value[] : tensor<f32>
  %4 = linalg.fill ins(%extracted_3 : f32) outs(%3 : tensor<1x3x3x1xf32>) -> tensor<1x3x3x1xf32>
  %result = iree_linalg_ext.select_and_scatter window_dimensions = dense<[1, 3, 2, 1]> : tensor<4xi64> window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64> padding = dense<[[0, 0], [1, 1], [1, 0], [0, 0]]> : tensor<4x2xi64> ins(%operand, %source, %extracted : tensor<1x3x3x1xf32>, tensor<1x2x2x1xf32>, f32) outs(%4 : tensor<1x3x3x1xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %6 = arith.cmpf oge, %arg1, %arg0 : f32
    iree_linalg_ext.yield %6 : i1
  }, {
  ^bb0(%arg0: f32, %arg1: f32):
    %6 = arith.addf %arg1, %arg0 : f32
    iree_linalg_ext.yield %6 : f32
  } -> tensor<1x3x3x1xf32>
  check.expect_eq_const(%result, dense<[[[[0.0], [0.0], [0.0]], [[10.0], [0.0], [11.0]], [[12.0], [0.0], [13.0]]]]> : tensor<1x3x3x1xf32>) : tensor<1x3x3x1xf32>
  return
}

func.func @scatter_and_scatter_2x3x3x2_f32() {
  %operand = util.unfoldable_constant dense<[[[[1.0, 19.0], [2.0, 18.0], [3.0, 17.0]], [[4.0, 16.0], [5.0, 15.0], [6.0, 14.0]], [[7.0, 13.0], [8.0, 12.0], [9.0, 11.0]]],[[[29.0, 31.0], [28.0, 32.0], [27.0, 33.0]], [[26.0, 34.0], [25.0, 35.0], [24.0, 36.0]], [[23.0, 37.0], [22.0, 38.0], [21.0, 39.0]]]]> : tensor<2x3x3x2xf32>
  %source = util.unfoldable_constant dense<[[[[10.0, 20.0], [11.0, 21.0]], [[12.0, 22.0], [13.0, 23.0]]],[[[30.0, 40.0], [31.0, 41.0]], [[32.0, 42.0], [33.0, 43.0]]]]> : tensor<2x2x2x2xf32>
  %init_value = util.unfoldable_constant dense<0.0> : tensor<f32>
  %extracted = tensor.extract %init_value[] : tensor<f32>
  %3 = tensor.empty() : tensor<2x3x3x2xf32>
  %extracted_3 = tensor.extract %init_value[] : tensor<f32>
  %4 = linalg.fill ins(%extracted_3 : f32) outs(%3 : tensor<2x3x3x2xf32>) -> tensor<2x3x3x2xf32>
  %result = iree_linalg_ext.select_and_scatter window_dimensions = dense<[1, 3, 2, 1]> : tensor<4xi64> window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64> padding = dense<[[0, 0], [1, 1], [1, 0], [0, 0]]> : tensor<4x2xi64> ins(%operand, %source, %extracted : tensor<2x3x3x2xf32>, tensor<2x2x2x2xf32>, f32) outs(%4 : tensor<2x3x3x2xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %6 = arith.cmpf oge, %arg1, %arg0 : f32
    iree_linalg_ext.yield %6 : i1
  }, {
  ^bb0(%arg0: f32, %arg1: f32):
    %6 = arith.addf %arg1, %arg0 : f32
    iree_linalg_ext.yield %6 : f32
  } -> tensor<2x3x3x2xf32>
  check.expect_eq_const(%result, dense<[[[[0.0, 20.0], [0.0, 21.0], [0.0, 0.0]], [[10.0, 22.0], [0.0, 23.0], [11.0, 0.0]], [[12.0, 0.0], [0.0, 0.0], [13.0, 0.0]]],[[[30.0, 0.0], [31.0, 0.0], [0.0, 0.0]], [[32.0, 40.0], [33.0, 0.0], [0.0, 41.0]], [[0.0, 42.0], [0.0, 0.0], [0.0, 43.0]]]]> : tensor<2x3x3x2xf32>) : tensor<2x3x3x2xf32>
  return
} 
