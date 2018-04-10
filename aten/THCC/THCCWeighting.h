void  THCCFloatTensor_weightingForward(      THCudaTensor *self,       THCudaTensor *src,       THCudaTensor *weight,       THCudaTensor *basis, THCudaLongTensor *weightIndex);
void THCCDoubleTensor_weightingForward(THCudaDoubleTensor *self, THCudaDoubleTensor *src, THCudaDoubleTensor *weight, THCudaDoubleTensor *basis, THCudaLongTensor *weightIndex);

void  THCCFloatTensor_weightingBackwardSrc(      THCudaTensor *self,       THCudaTensor *gradOutput,       THCudaTensor *weight,       THCudaTensor *basis, THCudaLongTensor *weightIndex);
void THCCDoubleTensor_weightingBackwardSrc(THCudaDoubleTensor *self, THCudaDoubleTensor *gradOutput, THCudaDoubleTensor *weight, THCudaDoubleTensor *basis, THCudaLongTensor *weightIndex);

void  THCCFloatTensor_weightingBackwardWeight(      THCudaTensor *self,      THCudaTensor *gradOutput,        THCudaTensor *src,       THCudaTensor *basis, THCudaLongTensor *weightIndex);
void THCCDoubleTensor_weightingBackwardWeight(THCudaDoubleTensor *self, THCudaDoubleTensor *gradOutput, THCudaDoubleTensor *src, THCudaDoubleTensor *basis, THCudaLongTensor *weightIndex);

void  THCCFloatTensor_weightingBackwardBasis(      THCudaTensor *self,       THCudaTensor *gradOutput,       THCudaTensor *src,       THCudaTensor *weight, THCudaLongTensor *weightIndex);
void THCCDoubleTensor_weightingBackwardBasis(THCudaDoubleTensor *self, THCudaDoubleTensor *gradOutput, THCudaDoubleTensor *src, THCudaDoubleTensor *weight, THCudaLongTensor *weightIndex);
