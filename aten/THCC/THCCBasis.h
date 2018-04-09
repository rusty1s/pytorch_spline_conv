void     THCCFloatTensor_linearBasisForward(      THCudaTensor *basis, THCudaLongTensor *weightIndex,       THCudaTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void    THCCDoubleTensor_linearBasisForward(THCudaDoubleTensor *basis, THCudaLongTensor *weightIndex, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void  THCCFloatTensor_quadraticBasisForward(      THCudaTensor *basis, THCudaLongTensor *weightIndex,       THCudaTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void THCCDoubleTensor_quadraticBasisForward(THCudaDoubleTensor *basis, THCudaLongTensor *weightIndex, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void      THCCFloatTensor_cubicBasisForward(      THCudaTensor *basis, THCudaLongTensor *weightIndex,       THCudaTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void     THCCDoubleTensor_cubicBasisForward(THCudaDoubleTensor *basis, THCudaLongTensor *weightIndex, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);

void     THCCFloatTensor_linearBasisBackward(      THCudaTensor *self,       THCudaTensor *gradBasis,       THCudaTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void    THCCDoubleTensor_linearBasisBackward(THCudaDoubleTensor *self, THCudaDoubleTensor *gradBasis, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void  THCCFloatTensor_quadraticBasisBackward(      THCudaTensor *self,       THCudaTensor *gradBasis,       THCudaTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void THCCDoubleTensor_quadraticBasisBackward(THCudaDoubleTensor *self, THCudaDoubleTensor *gradBasis, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void      THCCFloatTensor_cubicBasisBackward(      THCudaTensor *self,       THCudaTensor *gradBasis,       THCudaTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
void     THCCDoubleTensor_cubicBasisBackward(THCudaDoubleTensor *self, THCudaDoubleTensor *gradBasis, THCudaDoubleTensor *pseudo, THCudaLongTensor *kernelSize, THCudaByteTensor *isOpenSpline);
