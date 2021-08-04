func @scatter_add_slice_2D_dynamic_num_updates() {
  %arg0 = iree.unfoldable_constant dense<1> : tensor<6x3xi32>
  %arg1 = iree.dynamic_shape_constant dense<[[2], [4]]> : tensor<2x1xi32> -> tensor<?x1xi32>
  %arg2 = iree.dynamic_shape_constant dense<[[1, 2, 3],
                                             [4, 5, 6]]> : tensor<2x3xi32> -> tensor<?x3xi32>
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):  // no predecessors
    %1 = mhlo.add %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = {
      index_vector_dim = 1 : i64,
      inserted_window_dims = dense<0> : tensor<1xi64>,
      scatter_dims_to_operand_dims = dense<0> : tensor<1xi64>,
      update_window_dims = dense<1> : tensor<1xi64>
    },
    unique_indices = false
  } : (tensor<6x3xi32>, tensor<?x1xi32>, tensor<?x3xi32>) -> tensor<6x3xi32>
  check.expect_eq_const(%0, dense<[[1, 1, 1],
                                   [1, 1, 1],
                                   [2, 3, 4],
                                   [1, 1, 1],
                                   [5, 6, 7],
                                   [1, 1, 1]]> : tensor<6x3xi32>) : tensor<6x3xi32>
  return
}
