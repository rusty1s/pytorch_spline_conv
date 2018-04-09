void     THFloatTensor_linearBasisForward( THFloatTensor *basis, THLongTensor *weightIndex,  THFloatTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void    THDoubleTensor_linearBasisForward(THDoubleTensor *basis, THLongTensor *weightIndex, THDoubleTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void  THFloatTensor_quadraticBasisForward( THFloatTensor *basis, THLongTensor *weightIndex,  THFloatTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void THDoubleTensor_quadraticBasisForward(THDoubleTensor *basis, THLongTensor *weightIndex, THDoubleTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void      THFloatTensor_cubicBasisForward( THFloatTensor *basis, THLongTensor *weightIndex,  THFloatTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void     THDoubleTensor_cubicBasisForward(THDoubleTensor *basis, THLongTensor *weightIndex, THDoubleTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);

void     THFloatTensor_linearBasisBackward( THFloatTensor *self,  THFloatTensor *gradBasis,  THFloatTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void    THDoubleTensor_linearBasisBackward(THDoubleTensor *self, THDoubleTensor *gradBasis, THDoubleTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void  THFloatTensor_quadraticBasisBackward( THFloatTensor *self,  THFloatTensor *gradBasis,  THFloatTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void THDoubleTensor_quadraticBasisBackward(THDoubleTensor *self, THDoubleTensor *gradBasis, THDoubleTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void      THFloatTensor_cubicBasisBackward( THFloatTensor *self,  THFloatTensor *gradBasis,  THFloatTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void     THDoubleTensor_cubicBasisBackward(THDoubleTensor *self, THDoubleTensor *gradBasis, THDoubleTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
