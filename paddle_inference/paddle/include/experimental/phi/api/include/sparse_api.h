#pragma once

#include <tuple>

#include "paddle/include/experimental/phi/api/include/tensor.h"
#include "paddle/include/experimental/phi/common/scalar.h"
#include "paddle/include/experimental/phi/common/int_array.h"
#include "paddle/include/experimental/utils/optional.h"

namespace paddle {
namespace experimental {
namespace sparse {


// out@SparseCooTensor, rulebook@DenseTensor
PADDLE_API std::tuple<Tensor,Tensor> conv3d(const Tensor& x, const Tensor& kernel, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides, int groups, bool subm);

// out@DenseTensor
PADDLE_API Tensor coo_to_dense(const Tensor& x);

// out@DenseTensor
PADDLE_API Tensor coo_values(const Tensor& x);

// out@SparseCooTensor
PADDLE_API Tensor create_sparse_coo_tensor(const Tensor& values, const Tensor& indices, const IntArray& dense_shape);

// out@DenseTensor
PADDLE_API Tensor csr_values(const Tensor& x);

// out@SparseCooTensor
PADDLE_API Tensor dense_to_coo(const Tensor& x, int64_t sparse_dim);

// out@SparseCooTensor
PADDLE_API Tensor relu(const Tensor& x);

// out@DenseTensor
PADDLE_API Tensor to_dense(const Tensor& x);

// out@SparseCooTensor
PADDLE_API Tensor to_sparse_coo(const Tensor& x, int64_t sparse_dim);

// out@SparseCsrTensor
PADDLE_API Tensor to_sparse_csr(const Tensor& x);


}  // namespace sparse
}  // namespace experimental
}  // namespace paddle
