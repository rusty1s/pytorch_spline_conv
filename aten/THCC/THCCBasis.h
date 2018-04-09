void     THCCFloatTensor_linearBasisForward(      THCudaTensor *basis, THCudaLongTensor *weightIndex,       THCudaTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void    THCCDoubleTensor_linearBasisForward(THCudaDoubleTensor *basis, THCudaLongTensor *weightIndex, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void  THCCFloatTensor_quadraticBasisForward(      THCudaTensor *basis, THCudaLongTensor *weightIndex,       THCudaTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void THCCDoubleTensor_quadraticBasisForward(THCudaDoubleTensor *basis, THCudaLongTensor *weightIndex, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void      THCCFloatTensor_cubicBasisForward(      THCudaTensor *basis, THCudaLongTensor *weightIndex,       THCudaTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void     THCCDoubleTensor_cubicBasisForward(THCudaDoubleTensor *basis, THCudaLongTensor *weightIndex, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
