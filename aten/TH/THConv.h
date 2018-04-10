void  THFloatTensor_convForward( THFloatTensor *self,  THFloatTensor *src,  THFloatTensor *weight,  THFloatTensor *basis, THLongTensor *weightIndex);
void THDoubleTensor_convForward(THDoubleTensor *self, THDoubleTensor *src, THDoubleTensor *weight, THDoubleTensor *basis, THLongTensor *weightIndex);

void  THFloatTensor_convBackwardSrc( THFloatTensor *self,  THFloatTensor *gradOutput,  THFloatTensor *weight,  THFloatTensor *basis, THLongTensor *weightIndex);
void THDoubleTensor_convBackwardSrc(THDoubleTensor *self, THDoubleTensor *gradOutput, THDoubleTensor *weight, THDoubleTensor *basis, THLongTensor *weightIndex);

void  THFloatTensor_convBackwardBasis( THFloatTensor *self,  THFloatTensor *gradOutput,  THFloatTensor *src,  THFloatTensor *weight, THLongTensor *weightIndex);
void THDoubleTensor_convBackwardBasis(THDoubleTensor *self, THDoubleTensor *gradOutput, THDoubleTensor *src, THDoubleTensor *weight, THLongTensor *weightIndex);

void  THFloatTensor_convBackwardWeight( THFloatTensor *self,  THFloatTensor *gradOutput,  THFloatTensor *src,  THFloatTensor *basis, THLongTensor *weightIndex);
void THDoubleTensor_convBackwardWeight(THDoubleTensor *self, THDoubleTensor *gradOutput, THDoubleTensor *src, THDoubleTensor *basis, THLongTensor *weightIndex);
