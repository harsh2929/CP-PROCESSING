#include <torch/extension.h>
#include <vector>

// Forward declaration of functions
int emd_forward(const at::Tensor& xyz1, const at::Tensor& xyz2, at::Tensor& dist, at::Tensor& assignment,
                at::Tensor& price, at::Tensor& assignment_inv, at::Tensor& bid, at::Tensor& bid_increments,
                at::Tensor& max_increments, at::Tensor& unass_idx, at::Tensor& unass_cnt, at::Tensor& unass_cnt_sum,
                at::Tensor& cnt_tmp, at::Tensor& max_idx, const float eps, const int iters);

int emd_backward(const at::Tensor& xyz1, const at::Tensor& xyz2, at::Tensor& gradxyz, at::Tensor& graddist,
                 const at::Tensor& idx);

// Define Python functions using the C++ functions
// Add comments to explain the purpose of the functions and their parameters
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &emd_forward, "Compute the Earth Mover's Distance forward pass (CUDA)");
  m.def("backward", &emd_backward, "Compute the Earth Mover's Distance backward pass (CUDA)");
}

// Compute the Earth Mover's Distance forward pass
int emd_forward(const at::Tensor& xyz1, const at::Tensor& xyz2, at::Tensor& dist, at::Tensor& assignment,
                at::Tensor& price, at::Tensor& assignment_inv, at::Tensor& bid, at::Tensor& bid_increments,
                at::Tensor& max_increments, at::Tensor& unass_idx, at::Tensor& unass_cnt, at::Tensor& unass_cnt_sum,
                at::Tensor& cnt_tmp, at::Tensor& max_idx, const float eps, const int iters) {
  // Add error checking to ensure the input tensors are of the correct shape and type
  TORCH_CHECK(xyz1.dtype() == torch::kFloat32 && xyz2.dtype() == torch::kFloat32 && dist.dtype() == torch::kFloat32 &&
              assignment.dtype() == torch::kInt32 && price.dtype() == torch::k
