#include "hmt/bc_feat.hxx"
#include "hmt/hmt_util.hxx"
#include "np_helpers.hxx"
#include "pyglia.hxx"
#include "type/region_map.hxx"
#include "util/mp.hxx"
#include "util/text_cmd.hxx"
#include "util/text_io.hxx"

using namespace glia;
using namespace glia::hmt;

using LabelImageType = LabelImage<DIMENSION>;
using RealImageType = RealImage<DIMENSION>;
using ImagePairs =
    std::vector<hmt::ImageHistPair<RealImage<DIMENSION>::Pointer>>;
typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;

double initSal = 1.0;
double salBias = 1.0;
std::vector<double> boundaryThresholds;
bool normalizeShape = false;
bool useLogShape = false;
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
histogramBins: For each element in images, number of bins for histogram
computation histogramLowerValues: For each element in images, lower value of
histogram histogramHigherValues: For each element in images, lower value of
histogram initialSaliency: default: 1.0 saliencyBias: default: 1.0
boundaryShapeThresholds: ?
normalizeSizeLength: see paper, default to true
useLogOfShapes: see paper, default to true
---------------------------------------------------------*/

std::vector<std::vector<FVal>> bc_feat_operation(
    std::vector<TTriple<Label>> const &order,
    std::vector<double> const &saliencies,
    LabelImageType::Pointer spLabels, // SP labels
    std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>> const
        &vecImagePairs,                    // LAB, HSV, SIFT codes, etc..
    RealImageType::Pointer const &pbImage, // gPb, UCM, etc..
    double const &saliencyBias, bp::list const &boundaryShapeThresholds,
    bool const &normalizeShape, bool const &useLogShape) {

  auto mask = LabelImageType::Pointer(nullptr);

  // Set up normalizing area/length
  double normalizingArea = normalizeShape ? getImageVolume(spLabels) : 1.0;
  double normalizingLength = normalizeShape ? getImageDiagonal(spLabels) : 1.0;

  std::unordered_map<Label, double> saliencyMap;
  if (saliencies.size() > 0) {
    genSaliencyMap(saliencyMap, order, saliencies, initSal, saliencyBias);
  }

  // Generate region features
  RegionMap rmap(spLabels, mask, order, false);
  int rn = rmap.size();
  std::vector<std::pair<Label, std::shared_ptr<RegionFeats>>> rfeats(rn);

  std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>> vecBoundaryPairs,
      vecLabelPairs;

  parfor(
      rmap, true,
      [&rfeats, &vecImagePairs, &vecLabelPairs, &vecBoundaryPairs, &pbImage,
       &saliencyMap, normalizingArea,
       normalizingLength](RegionMap::const_iterator rit, int i) {
        rfeats[i].first = rit->first;
        rfeats[i].second = std::make_shared<RegionFeats>();
        rfeats[i].second->generate(
            rit->second, normalizingArea, normalizingLength, pbImage,
            boundaryThresholds, vecImagePairs, vecLabelPairs, vecBoundaryPairs,
            ccpointer(saliencyMap, rit->first));
      },
      0);

  std::unordered_map<Label, std::shared_ptr<RegionFeats>> rfmap;
  for (auto const &rfp : rfeats) {
    rfmap[rfp.first] = rfp.second;
  }

  // Generate boundary classifier features
  std::cout << "generating boundary features" << std::endl;
  int bn = order.size();
  std::vector<BoundaryClassificationFeats> bfeats(bn);
  parfor(
      0, bn, true,
      [&rmap, &order, &bfeats, &rfmap, &vecBoundaryPairs, &pbImage,
       normalizingLength](int i) {
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
        getBoundary(b, rmap.find(r0)->second, rmap.find(r1)->second);
        bfeats[i].x0.generate(b, normalizingLength, *bfeats[i].x1,
                              *bfeats[i].x2, *bfeats[i].x3, pbImage,
                              boundaryThresholds, vecBoundaryPairs);
      },
      0);

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

  return pickedFeats;
}

np::ndarray MyHmt::bc_feat_wrp(
    bp::list const &mergeOrderList, np::ndarray const &saliencies,
    np::ndarray const &labelArray, // SP labels
    bp::list const &images,        // LAB, HSV, SIFT codes, etc..
    np::ndarray const &pbArray,    // gPb, UCM, etc..
    bp::list const &histogramBins, bp::list const &histogramLowerValues,
    bp::list const &histogramHigherValues, double const &initialSaliency,
    double const &saliencyBias, bp::list const &boundaryShapeThresholds,
    bool const &normalizeSizeLength, bool const &useLogOfShapes) {

  LabelImageType::Pointer segImage = nph::np_to_itk_label(labelArray);
  RealImageType::Pointer pbImage = nph::np_to_itk_real(pbArray);

  auto saliencies_vec = nph::np_to_vector<double>(saliencies);
  auto order = nph::np_to_vector_triple<Label>(mergeOrderList);

  // Load and set up images
  std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>> vecImagePairs;

  // Set histogram ranges and bins
  vecImagePairs = nph::lists_to_image_hist_pair(
      images, bp::extract<int>(histogramBins[1]),
      bp::extract<double>(histogramLowerValues[1]),
      bp::extract<double>(histogramHigherValues[1]));

  auto feats = bc_feat_operation(order, saliencies_vec, segImage, vecImagePairs,
                                 pbImage, salBias, boundaryShapeThresholds,
                                 normalizeShape, useLogShape);

  return nph::vector_2d_to_np<FVal>(feats);
}
