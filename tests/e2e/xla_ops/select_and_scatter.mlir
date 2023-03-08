func.func @scatter_and_scatter_1x3x3x1() {
  %operand = util.unfoldable_constant dense<[[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]], [[7.0], [8.0], [9.0]]]]> : tensor<1x3x3x1xf32>
  %source = util.unfoldable_constant dense<[[[[10.0], [11.0]], [[12.0], [13.0]]]]> : tensor<1x2x2x1xf32>
  %init_value = util.unfoldable_constant dense<0.0> : tensor<f32>
  %result = "mhlo.select_and_scatter"(%operand, %source, %init_value) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %0 = "mhlo.compare"(%arg0, %arg1) {
        comparison_direction = #mhlo<comparison_direction GE>
      } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%0) : (tensor<i1>) -> ()
    }, {
     ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %0 = "mhlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%0) : (tensor<f32>) -> ()
  }) {
  window_dimensions = dense<[1, 3, 2, 1]> : tensor<4xi64>,
  window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
  padding = dense<[[0, 0], [1, 1], [1, 0], [0, 0]]> : tensor<4x2xi64>
} : (tensor<1x3x3x1xf32>, tensor<1x2x2x1xf32>, tensor<f32>) -> tensor<1x3x3x1xf32>
  check.expect_eq_const(%result, dense<[[[[0.0], [0.0], [0.0]], [[10.0], [0.0], [11.0]], [[12.0], [0.0], [13.0]]]]> : tensor<1x3x3x1xf32>) : tensor<1x3x3x1xf32>
  return
}

func.func @scatter_and_scatter_2x3x3x2() {
  %operand = util.unfoldable_constant dense<[[[[1.0, 19.0], [2.0, 18.0], [3.0, 17.0]], [[4.0, 16.0], [5.0, 15.0], [6.0, 14.0]], [[7.0, 13.0], [8.0, 12.0], [9.0, 11.0]]],[[[29.0, 31.0], [28.0, 32.0], [27.0, 33.0]], [[26.0, 34.0], [25.0, 35.0], [24.0, 36.0]], [[23.0, 37.0], [22.0, 38.0], [21.0, 39.0]]]]> : tensor<2x3x3x2xf32>
  %source = util.unfoldable_constant dense<[[[[10.0, 20.0], [11.0, 21.0]], [[12.0, 22.0], [13.0, 23.0]]],[[[30.0, 40.0], [31.0, 41.0]], [[32.0, 42.0], [33.0, 43.0]]]]> : tensor<2x2x2x2xf32>
  %init_value = util.unfoldable_constant dense<0.0> : tensor<f32>
  %result = "mhlo.select_and_scatter"(%operand, %source, %init_value) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %0 = "mhlo.compare"(%arg0, %arg1) {
        comparison_direction = #mhlo<comparison_direction GE>
      } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%0) : (tensor<i1>) -> ()
    }, {
     ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %0 = "mhlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%0) : (tensor<f32>) -> ()
  }) {
  window_dimensions = dense<[1, 3, 2, 1]> : tensor<4xi64>,
  window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
  padding = dense<[[0, 0], [1, 1], [1, 0], [0, 0]]> : tensor<4x2xi64>
} : (tensor<2x3x3x2xf32>, tensor<2x2x2x2xf32>, tensor<f32>) -> tensor<2x3x3x2xf32>
  check.expect_eq_const(%result, dense<[[[[0.0, 20.0], [0.0, 21.0], [0.0, 0.0]], [[10.0, 22.0], [0.0, 23.0], [11.0, 0.0]], [[12.0, 0.0], [0.0, 0.0], [13.0, 0.0]]],[[[30.0, 0.0], [31.0, 0.0], [0.0, 0.0]], [[32.0, 40.0], [33.0, 0.0], [0.0, 41.0]], [[0.0, 42.0], [0.0, 0.0], [0.0, 43.0]]]]> : tensor<2x3x3x2xf32>) : tensor<2x3x3x2xf32>
  return
}

// func.func @select_and_scatter(%arg0 : tensor<2x8x8x1xi32>, %arg1 : tensor<2x4x4x1xi32>, %arg2 : tensor<i32>) -> tensor<2x8x8x1xi32> {
//   %0 = "mhlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
//   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
//     %9 = mhlo.compare  GE, %arg3, %arg4,  FLOAT : (tensor<i32>, tensor<i32>) -> tensor<i1>
//     mhlo.return %9 : tensor<i1>
//   }, {
//   ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
//     %9 = mhlo.add %arg3, %arg4 : tensor<i32>
//     mhlo.return %9 : tensor<i32>
//   }) {
//     padding = dense<0> : tensor<4x2xi64>,
//     window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
//     window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
//   } : (tensor<2x8x8x1xi32>, tensor<2x4x4x1xi32>, tensor<i32>) -> tensor<2x8x8x1xi32>

//   return %0 : tensor<2x8x8x1xi32>
// }
func.func @scatter_and_scatter_1x4x2x1_f32() {
  %operand = util.unfoldable_constant dense<[[[[1.0], [5.0]], [[2.0], [5.0]], [[3.0], [6.0]], [[4.0], [4.0]]]]> : tensor<1x4x2x1xf32>
  %source = util.unfoldable_constant dense<[[[[5.0], [6.0]], [[7.0], [8.0]]]]> : tensor<1x2x2x1xf32>
  %init_value = util.unfoldable_constant dense<0.0> : tensor<f32>
  %result = "mhlo.select_and_scatter"(%operand, %source, %init_value) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %0 = "mhlo.compare"(%arg0, %arg1) {
        comparison_direction = #mhlo<comparison_direction GE>
      } : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%0) : (tensor<i1>) -> ()
    }, {
     ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %0 = "mhlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%0) : (tensor<f32>) -> ()
  }) {
  window_dimensions = dense<[1, 3, 1, 1]> : tensor<4xi64>,
  window_strides = dense<[1, 2, 1, 1]> : tensor<4xi64>,
  padding = dense<[[0, 0], [0, 1], [0, 0], [0, 0]]> : tensor<4x2xi64>
} : (tensor<1x4x2x1xf32>, tensor<1x2x2x1xf32>, tensor<f32>) -> tensor<1x4x2x1xf32>
  check.expect_eq_const(%result, dense<[[[[0.0], [0.0]], [[0.0], [0.0]], [[5.0], [14.0]], [[7.0], [0.0]]]]> : tensor<1x4x2x1xf32>) : tensor<1x4x2x1xf32>
  return
}

func.func @scatter_and_scatter_1x4x2x1_i32() {
  %operand = util.unfoldable_constant dense<[[[[1], [5]], [[2], [5]], [[3], [6]], [[4], [4]]]]> : tensor<1x4x2x1xi32>
  %source = util.unfoldable_constant dense<[[[[5], [6]], [[7], [8]]]]> : tensor<1x2x2x1xi32>
  %init_value = util.unfoldable_constant dense<0> : tensor<i32>
  %result = "mhlo.select_and_scatter"(%operand, %source, %init_value) ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %0 = "mhlo.compare"(%arg0, %arg1) {
        comparison_direction = #mhlo<comparison_direction GE>
      } : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "mhlo.return"(%0) : (tensor<i1>) -> ()
    }, {
     ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %0 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "mhlo.return"(%0) : (tensor<i32>) -> ()
  }) {
  window_dimensions = dense<[1, 3, 1, 1]> : tensor<4xi64>,
  window_strides = dense<[1, 2, 1, 1]> : tensor<4xi64>,
  padding = dense<[[0, 0], [0, 1], [0, 0], [0, 0]]> : tensor<4x2xi64>
} : (tensor<1x4x2x1xi32>, tensor<1x2x2x1xi32>, tensor<i32>) -> tensor<1x4x2x1xi32>
  check.expect_eq_const(%result, dense<[[[[0], [0]], [[0], [0]], [[5], [14]], [[7], [0]]]]> : tensor<1x4x2x1xi32>) : tensor<1x4x2x1xi32>
  return
}
