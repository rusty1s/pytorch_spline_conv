void  THFloatTensor_weightingForward( THFloatTensor *self,  THFloatTensor *src,  THFloatTensor *weight,  THFloatTensor *basis, THLongTensor *weightIndex);
void THDoubleTensor_weightingForward(THDoubleTensor *self, THDoubleTensor *src, THDoubleTensor *weight, THDoubleTensor *basis, THLongTensor *weightIndex);

void  THFloatTensor_weightingBackwardSrc( THFloatTensor *self,  THFloatTensor *gradOutput,  THFloatTensor *weight,  THFloatTensor *basis, THLongTensor *weightIndex);
void THDoubleTensor_weightingBackwardSrc(THDoubleTensor *self, THDoubleTensor *gradOutput, THDoubleTensor *weight, THDoubleTensor *basis, THLongTensor *weightIndex);

void  THFloatTensor_weightingBackwardWeight( THFloatTensor *self,  THFloatTensor *gradOutput,  THFloatTensor *src,  THFloatTensor *basis, THLongTensor *weightIndex);
void THDoubleTensor_weightingBackwardWeight(THDoubleTensor *self, THDoubleTensor *gradOutput, THDoubleTensor *src, THDoubleTensor *basis, THLongTensor *weightIndex);

void  THFloatTensor_weightingBackwardBasis( THFloatTensor *self,  THFloatTensor *gradOutput,  THFloatTensor *src,  THFloatTensor *weight, THLongTensor *weightIndex);
void THDoubleTensor_weightingBackwardBasis(THDoubleTensor *self, THDoubleTensor *gradOutput, THDoubleTensor *src, THDoubleTensor *weight, THLongTensor *weightIndex);

void  THFloatTensor_weightingBackward( THFloatTensor *gradSrc,  THFloatTensor *gradWeight,  THFloatTensor *gradBasis,  THFloatTensor *gradOutput,  THFloatTensor *src,  THFloatTensor *weight,  THFloatTensor *basis, THLongTensor *weightIndex);
void THDoubleTensor_weightingBackward(THDoubleTensor *gradSrc, THDoubleTensor *gradWeight, THDoubleTensor *gradBasis, THDoubleTensor *gradOutput, THDoubleTensor *src, THDoubleTensor *weight, THDoubleTensor *basis, THLongTensor *weightIndex);
