#include "bc_feat.hxx"

// Follow arXiv paper
void glia::hmt::selectFeatures (
    std::vector<FVal>& f, BoundaryClassificationFeats const& bcf)
{
  int size = f.size() + 5 + 4 * bcf.x0.region.size() +
      2 * bcf.x0.labelRegion.size();
#ifdef GLIA_USE_MEDIAN_AS_FEATS
  size += 2 * bcf.x0.boundary.size();
#else
  size += bcf.x0.boundary.size();
#endif
  f.reserve(size);
  f.push_back(bcf.x1->shape->area);
  f.push_back(bcf.x2->shape->area);
  f.push_back(bcf.x1->shape->perim);
  f.push_back(bcf.x2->shape->perim);
  f.push_back(bcf.x0.shape->boundaryLength);
  for (auto const& bf : bcf.x0.boundary) {
    f.push_back(bf->mean);
#ifdef GLIA_USE_MEDIAN_AS_FEATS
    f.push_back(bf->median);
#endif
  }
  for (auto const& rf : bcf.x0.region) {
    f.push_back(rf->meanDiff);
    f.push_back(rf->histDistL1);
    f.push_back(rf->histDistX2);
    f.push_back(rf->entropyDiff);
  }
  for (auto const& rlf : bcf.x0.labelRegion) {
    f.push_back(rlf->histDistL1);
    f.push_back(rlf->histDistX2);
  }
}
