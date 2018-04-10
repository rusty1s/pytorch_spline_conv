void  THCCFloatTensor_convForward(      THCudaTensor *self,       THCudaTensor *src,       THCudaTensor *weight,       THCudaTensor *basis, THCudaLongTensor *weightIndex);
void THCCDoubleTensor_convForward(THCudaDoubleTensor *self, THCudaDoubleTensor *src, THCudaDoubleTensor *weight, THCudaDoubleTensor *basis, THCudaLongTensor *weightIndex);

void  THCCFloatTensor_convBackwardSrc(      THCudaTensor *self,       THCudaTensor *gradOutput,       THCudaTensor *weight,       THCudaTensor *basis, THCudaLongTensor *weightIndex);
void THCCDoubleTensor_convBackwardSrc(THCudaDoubleTensor *self, THCudaDoubleTensor *gradOutput, THCudaDoubleTensor *weight, THCudaDoubleTensor *basis, THCudaLongTensor *weightIndex);

void  THCCFloatTensor_convBackwardBasis(      THCudaTensor *self,       THCudaTensor *gradOutput,       THCudaTensor *src,       THCudaTensor *weight, THCudaLongTensor *weightIndex);
void THCCDoubleTensor_convBackwardBasis(THCudaDoubleTensor *self, THCudaDoubleTensor *gradOutput, THCudaDoubleTensor *src, THCudaDoubleTensor *weight, THCudaLongTensor *weightIndex);

void  THCCFloatTensor_convBackwardWeight(      THCudaTensor *self,      THCudaTensor *gradOutput,        THCudaTensor *src,       THCudaTensor *basis, THCudaLongTensor *weightIndex);
void THCCDoubleTensor_convBackwardWeight(THCudaDoubleTensor *self, THCudaDoubleTensor *gradOutput, THCudaDoubleTensor *src, THCudaDoubleTensor *basis, THCudaLongTensor *weightIndex);
