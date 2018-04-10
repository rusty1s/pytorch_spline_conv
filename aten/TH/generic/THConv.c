#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THConv.c"
#else

void  THTensor_(convForward)(THTensor *self, THTensor *src, THTensor *weight, THTensor *basis, THLongTensor *weightIndex) {
}

void  THTensor_(convBackwardSrc)(THTensor *self, THTensor *gradOutput, THTensor *weight, THTensor *basis, THLongTensor *weightIndex) {
}

void  THTensor_(convBackwardBasis)(THTensor *self, THTensor *gradOutput, THTensor *src, THTensor *weight, THLongTensor *weightIndex) {
}

void  THTensor_(convBackwardWeight)(THTensor *self, THTensor *gradOutput, THTensor *src, THTensor *basis, THLongTensor *weightIndex) {
}

#endif // TH_GENERIC_FILE
