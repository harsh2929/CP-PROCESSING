#include <torch/extension.h>
#include <vector>

int c_fwd(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist, at::Tensor assignment, at::Tensor price, 
	                 at::Tensor assignment_inv, at::Tensor bid, at::Tensor bid_increments, at::Tensor max_increments,
	                 at::Tensor unass_idx, at::Tensor unass_cnt, at::Tensor unass_cnt_sum, at::Tensor cnt_tmp, at::Tensor max_idx, float eps, int iters);

int c_bck(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz, at::Tensor graddist, at::Tensor idx);



int fwd(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist, at::Tensor assignment, at::Tensor price, 
	                 at::Tensor assignment_inv, at::Tensor bid, at::Tensor bid_increments, at::Tensor max_increments,
	                 at::Tensor unass_idx, at::Tensor unass_cnt, at::Tensor unass_cnt_sum, at::Tensor cnt_tmp, at::Tensor max_idx, float eps, int iters) {
	return c_fwd(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters);
}

int bck(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz, at::Tensor graddist, at::Tensor idx) {

    return c_bck(xyz1, xyz2, gradxyz, graddist, idx);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fwd, "emd forward (CUDA)");
  m.def("backward", &bck, "emd backward (CUDA)");
}