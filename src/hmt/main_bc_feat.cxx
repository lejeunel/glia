#include "hmt/bc_feat.hxx"
#include "type/region_map.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
#include "hmt/hmt_util.hxx"
#include "pyglia.hxx"
#include "np_helpers.hxx"

using namespace glia;
using namespace glia::hmt;

std::string segImageFile;
std::string mergeOrderFile;
std::string saliencyFile;
std::string pbImageFile;
std::vector<ImageFileHistPair> rbImageFiles;
std::vector<ImageFileHistPair> rlImageFiles;
std::vector<ImageFileHistPair> rImageFiles;
std::vector<ImageFileHistPair> bImageFiles;
std::string maskImageFile;
double initSal = 1.0;
double salBias = 1.0;
std::vector<double> boundaryThresholds;
bool normalizeShape = false;
bool useLogShape = false;
bool useSimpleFeatures = false;
std::string bfeatFile;

namespace np = boost::python::numpy;
namespace bp = boost::python;

/*-------------------------------------------------------
For a given merge order and saliency metric, compute for each clique
a feature vector

Parameters:
mergeOrderList: Order of merging. Obtained with merge_order_pb
saliencies: Obtained with merge_order_pb
spLabels: Integer map of superpixel segmentation
images: LAB, HSV, SIFT codes, etc..
gpbImage: global probability boundary
histogramBins: For each element in images, number of bins for histogram computation
histogramLowerValues: For each element in images, lower value of histogram
histogramHigherValues: For each element in images, lower value of histogram
initialSaliency: default: 1.0
saliencyBias: default: 1.0
boundaryShapeThresholds: ?
normalizeSizeLength: see paper, default to true
useLogOfShapes: see paper, default to true
---------------------------------------------------------*/


np::ndarray MyHmt::bc_feat_operation (bp::list const& mergeOrderList,
                               np::ndarray const& saliencies,
                               np::ndarray const& spLabels, // SP labels
                               bp::list const& images, // LAB, HSV, SIFT codes, etc..
                               np::ndarray const& gpbImage, //gPb, UCM, etc..
                               bp::list const& histogramBins,
                               bp::list const& histogramLowerValues,
                               bp::list const& histogramHigherValues,
                               double const& initialSaliency,
                               double const& saliencyBias,
                               bp::list const& boundaryShapeThresholds,
                               bool const& normalizeSizeLength,
                               bool const& useLogOfShapes)
{
  using LabelImageType =  LabelImage<DIMENSION>;
  using RealImageType =  RealImage<DIMENSION>;
  using ImagePairs = std::vector<hmt::ImageHistPair<RealImage<DIMENSION>::Pointer>>;
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;

  // Load and set up images
  std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>>
    vecCodePairs, vecImagePairs, vecBoundaryPairs, vecLabelPairs;

  // Set histogram ranges and bins
  vecImagePairs = nph::lists_to_image_hist_pair(images,
                                bp::extract<int>(histogramBins[1]),
                                bp::extract<double>(histogramLowerValues[1]),
                                bp::extract<double>(histogramHigherValues[1]));

  auto mask = LabelImageType::Pointer(nullptr);
  auto spLabels_itk = nph::np_to_itk_label(spLabels);
  auto gpbImage_itk = nph::np_to_itk_real(gpbImage);

  // Set up normalizing area/length
  double normalizingArea =
      normalizeShape ? getImageVolume(spLabels_itk) : 1.0;
  double normalizingLength =
      normalizeShape ? getImageDiagonal(spLabels_itk) : 1.0;

  auto order = nph::np_to_vector_triple<Label>(mergeOrderList);

  auto saliencies_vec = nph::np_to_vector<double>(saliencies);
  std::unordered_map<Label, double> saliencyMap;
  if (!nph::is_empty(saliencies)) {
    genSaliencyMap(saliencyMap,
                   order,
                   saliencies_vec,
                   initialSaliency,
                   saliencyBias);
  }

  // Generate region features
  RegionMap rmap(spLabels_itk, mask, order, false);
  int rn = rmap.size();
  std::vector<std::pair<Label, std::shared_ptr<RegionFeats>>> rfeats(rn);
  parfor(rmap, true, [
      &rfeats, &vecImagePairs, &vecLabelPairs,
      &vecBoundaryPairs, &gpbImage_itk, &saliencyMap,
      normalizingArea, normalizingLength](
          RegionMap::const_iterator rit, int i) {
           rfeats[i].first = rit->first;
           rfeats[i].second = std::make_shared<RegionFeats>();
           rfeats[i].second->generate(
               rit->second, normalizingArea, normalizingLength,
               gpbImage_itk, boundaryThresholds,
               vecImagePairs, vecLabelPairs, vecBoundaryPairs,
               ccpointer(saliencyMap, rit->first));
         }, 0);

  std::unordered_map<Label, std::shared_ptr<RegionFeats>> rfmap;
  for (auto const& rfp : rfeats) { rfmap[rfp.first] = rfp.second; }

  // Generate boundary classifier features
  std::cout << "generating boundary features" << std::endl;
  int bn = order.size();
  std::vector<BoundaryClassificationFeats> bfeats(bn);
  parfor(0, bn, true, [
                       &rmap, &order, &bfeats, &rfmap, &vecBoundaryPairs,
                       &gpbImage_itk, normalizingLength](int i) {
                       Label r0 = order[i].x0;
                       Label r1 = order[i].x1;
                       Label r2 = order[i].x2;
                       bfeats[i].x1 = rfmap.find(r0)->second.get();
                       bfeats[i].x2 = rfmap.find(r1)->second.get();
                       bfeats[i].x3 = rfmap.find(r2)->second.get();
                       // Keep region 0 area <= region 1 area
                       if (bfeats[i].x1->shape->area > bfeats[i].x2->shape->area) {
                         std::swap(r0, r1);
                         std::swap(bfeats[i].x1, bfeats[i].x2);
                       }
                       RegionMap::Region::Boundary b;
                       getBoundary(b, rmap.find(r0)->second,
                                   rmap.find(r1)->second);
                       bfeats[i].x0.generate(b,
                                             normalizingLength,
                                             *bfeats[i].x1,
                                             *bfeats[i].x2,
                                             *bfeats[i].x3,
                                             gpbImage_itk,
                                             boundaryThresholds,
                                             vecBoundaryPairs);
                      }, 0);

  // Log shape

  // Get features for boundary classifier
  if (useLogShape) {
    parfor(
           0, rn, false, [&rfeats](int i) { rfeats[i].second->log(); }, 0);
    parfor(
           0, bn, false, [&bfeats](int i) { bfeats[i].x0.log(); }, 0);
  }

  std::vector<std::vector<FVal>> pickedFeats(bn);
  for (int i = 0; i < bn; ++i) {
    selectFeatures(pickedFeats[i], bfeats[i]);
  }

  return nph::vector_2d_to_np<FVal>(pickedFeats);
}
