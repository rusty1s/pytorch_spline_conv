#ifndef THC_ATOMICS_INC
#define THC_ATOMICS_INC

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
static inline  __device__  void atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*) address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
}
#elif !defined(__CUDA_ARCH__) && (CUDA_VERSION < 8000)
  static inline  __device__  void atomicAdd(double *address, double val) {}
#endif

#ifdef CUDA_HALF_TENSOR
static inline  __device__ void atomicAdd(half *address, half val) {
  unsigned int * address_as_ui = (unsigned int *) ((char *) address - ((size_t) address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
#if CUDA_VERSION < 9000
    half hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = THCNumerics<half>::add(hsum, val);
#else  // CUDA_VERSION < 9000
    __half_raw hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    half tmpres = THCNumerics<half>::add(hsum, val);
    hsum = __half_raw(tmpres);
#endif  // CUDA_VERSION
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}
#endif  // CUDA_HALF_TENSOR

#endif  // THC_ATOMICS_INC
