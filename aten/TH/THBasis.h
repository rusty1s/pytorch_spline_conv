void     THFloatTensor_linearBasisForward( THFloatTensor *basis, THLongTensor *weightIndex,  THFloatTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void    THDoubleTensor_linearBasisForward(THDoubleTensor *basis, THLongTensor *weightIndex, THDoubleTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void  THFloatTensor_quadraticBasisForward( THFloatTensor *basis, THLongTensor *weightIndex,  THFloatTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void THDoubleTensor_quadraticBasisForward(THDoubleTensor *basis, THLongTensor *weightIndex, THDoubleTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void      THFloatTensor_cubicBasisForward( THFloatTensor *basis, THLongTensor *weightIndex,  THFloatTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
void     THDoubleTensor_cubicBasisForward(THDoubleTensor *basis, THLongTensor *weightIndex, THDoubleTensor *pseudo, THLongTensor *kernelSize, THByteTensor *isOpenSpline);
