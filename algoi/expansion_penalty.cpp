#include <torch/extension.h>
#include <vector>

int fwd(at::Tensor xyz, int primitive_size, at::Tensor father, at::Tensor dist, double alpha, at::Tensor neighbor, at::Tensor cost, at::Tensor mean_mst_length);

int dexpbcwd(at::Tensor xyz, at::Tensor gradxyz, at::Tensor graddist, at::Tensor idx);

int dexp_fwd(at::Tensor xyz, int primitive_size, at::Tensor father, at::Tensor dist, double alpha, at::Tensor neighbor, at::Tensor cost, at::Tensor mean_mst_length) {
	return fwd(xyz, primitive_size, father, dist, alpha, neighbor, cost, mean_mst_length);
}

int expansion_penalty_backward(at::Tensor xyz, at::Tensor gradxyz, at::Tensor graddist, at::Tensor idx) {

    return dexpbcwd(xyz, gradxyz, graddist, idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dexp_fwd, "expansion_penalty forward (CUDA)");
  m.def("backward", &expansion_penalty_backward, "expansion_penalty backward (CUDA)");
}