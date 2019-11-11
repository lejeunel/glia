#include "feat.hxx"

void glia::feat::standardize(
    std::vector<FVal> &feats,
    std::vector<std::vector<FVal>> const &featNormalizer, FVal inputFeatNullVal,
    FVal outputFeatNullVal) {
  int rowMean = 0, rowStd = 1;
  int d = feats.size();
  for (int i = 0; i < d; ++i) {
    feats[i] = isfeq(feats[i], inputFeatNullVal)
                   ? outputFeatNullVal
                   : sdivide(feats[i] - featNormalizer[rowMean][i],
                             featNormalizer[rowStd][i], outputFeatNullVal);
  }
}
