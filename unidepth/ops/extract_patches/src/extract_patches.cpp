
#include "extract_patches.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("extract_patches_forward", &extract_patches_forward, "Extract patches forward (CUDA)");
  m.def("extract_patches_backward", &extract_patches_backward, "Extract patches backward (CUDA)");
}
