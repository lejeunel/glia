#include "alg/nn.hxx"
#include "alg/rf.hxx"
#include "hmt/bc_feat.hxx"
#include "hmt/bc_label.hxx"
#include "pyglia.hxx"
#include "util/struct_merge_bc.hxx"
#include "util/text_cmd.hxx"
#include "util/text_io.hxx"
#include "np_helpers.hxx"

using namespace glia;
using namespace glia::hmt;

bp::tuple MyHmt::merge_order_bc_operation(
    np::ndarray const &spLabels, // SP labels
    bp::list const &images,      // LAB, HSV, SIFT codes, etc..
    np::ndarray const &truth,
    np::ndarray const &gpbImage, // gPb, UCM, etc..
    bp::list const &histogramBins, bp::list const &histogramLowerValues,
    bp::list const &histogramHigherValues, bool const &useLogOfShape,
    bool const &useSimpleFeatures) {

  std::vector<double> boundaryThresholds;

  auto mask = LabelImage<DIMENSION>::Pointer(nullptr);
  auto spLabels_itk = nph::np_to_itk_label(spLabels);
  auto gpbImage_itk = nph::np_to_itk_real(gpbImage);
  auto truth_itk = nph::np_to_itk_label(truth);

  using LabelImageType = LabelImage<DIMENSION>;
  using RealImageType = RealImage<DIMENSION>;
  using ImagePairs =
      std::vector<hmt::ImageHistPair<RealImage<DIMENSION>::Pointer>>;
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;

  // Load and set up images
  std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>> vecCodePairs,
      vecImagePairs, vecBoundaryPairs, vecLabelPairs;

  // Set histogram ranges and bins
  vecImagePairs = nph::lists_to_image_hist_pair(
      images, bp::extract<int>(histogramBins[1]),
      bp::extract<double>(histogramLowerValues[1]),
      bp::extract<double>(histogramHigherValues[1]));

  // Set up normalizing area/length
  double normalizingArea = getImageVolume(spLabels_itk);
  double normalizingLength = getImageDiagonal(spLabels_itk);

  RegionMap rmap(spLabels_itk, mask, false);
  // Boundary feature compute
  std::unordered_map<std::pair<Label, Label>, std::vector<FVal>> bcfmap;
  auto fBcFeat = [normalizingArea, normalizingLength, &gpbImage_itk,
                  &vecImagePairs, &vecBoundaryPairs, &vecLabelPairs, &bcfmap,
                  useLogOfShape, useSimpleFeatures, boundaryThresholds](
                     std::vector<FVal> &data, RegionMap::Region const &reg0,
                     RegionMap::Region const &reg1,
                     RegionMap::Region const &reg2, Label r0, Label r1,
                     Label r2) {
    auto rf0 = std::make_shared<RegionFeats>();
    auto rf1 = std::make_shared<RegionFeats>();
    auto rf2 = std::make_shared<RegionFeats>();
    rf0->generate(reg0, normalizingArea, normalizingLength, gpbImage_itk,
                  boundaryThresholds, vecImagePairs, vecLabelPairs,
                  vecBoundaryPairs, nullptr);
    rf1->generate(reg1, normalizingArea, normalizingLength, gpbImage_itk,
                  boundaryThresholds, vecImagePairs, vecLabelPairs,
                  vecBoundaryPairs, nullptr);
    rf2->generate(reg2, normalizingArea, normalizingLength, gpbImage_itk,
                  boundaryThresholds, vecImagePairs, vecLabelPairs,
                  vecBoundaryPairs, nullptr);
    BoundaryClassificationFeats bcf;
    bcf.x1 = rf0.get();
    bcf.x2 = rf1.get();
    bcf.x3 = rf2.get();
    // Keep region 0 area <= region 1 area
    if (bcf.x1->shape->area > bcf.x2->shape->area) {
      std::swap(r0, r1);
      std::swap(bcf.x1, bcf.x2);
    }
    RegionMap::Region::Boundary b;
    getBoundary(b, reg0, reg1);
    bcf.x0.generate(b, normalizingLength, *bcf.x1, *bcf.x2, *bcf.x3,
                    gpbImage_itk, boundaryThresholds, vecLabelPairs);
    if (useLogOfShape) {
      bcf.x0.log();
      rf0->log();
      rf1->log();
      rf2->log();
    }
    if (useSimpleFeatures) {
      selectFeatures(data, bcf);
    } else {
      bcf.serialize(data);
    }
    bcfmap[std::make_pair(std::min(r0, r1), std::max(r0, r1))] = data;
  };
  std::shared_ptr<opt::TFunction<std::vector<FVal>>> bc;
  std::vector<std::vector<FVal>> bcfMinMax;

  // Create and initialize ensemble random forest
  // if (bcModelFiles.size() == 1) {
  //   bc = std::make_shared<alg::RandomForest>(BC_LABEL_MERGE,
  //                                            bcModelFiles.front());
  // } else {
  //   if (bcModelDistributorArgs.size() != 3) {
  //     perr("Error: model distributor needs 3 arguments...");
  //   }
  //   bc = std::make_shared<alg::EnsembleRandomForest>(
  //       BC_LABEL_MERGE, bcModelFiles,
  //       opt::ThresholdModelDistributor<FVal>(bcModelDistributorArgs[0],
  //                                            bcModelDistributorArgs[1],
  //                                            bcModelDistributorArgs[2]));
  // }

  // Boundary predictor
  // std::vector<FVal> tmpData;
  // auto fBcPred = [&bc, &bcfMinMax, &tmpData](std::vector<FVal> const &data) {
  //   return bc->operator()(data);
  // };

  // Generate merging orders
  std::vector<TTriple<Label>> order;
  std::vector<double> saliencies;
  // genMergeOrderGreedyUsingBoundaryClassifier<std::vector<FVal>>(
  //     order, saliencies, truth_itk, mask, fBcFeat, fBcPred,
  //     f_true<TBoundaryTable<std::vector<FVal>, RegionMap> &,
  //            TBoundaryTable<std::vector<FVal>, RegionMap>::iterator>);

  // store boundary classifier feats
  std::vector<std::vector<FVal>> bcfeats;
  // bcfeats.reserve(order.size());
  // for (auto const &m : order) {
  //   bcfeats.push_back(bcfmap.find(std::make_pair(m.x0, m.x1))->second);
  // }

  return bp::make_tuple(nph::vector_triple_to_np<Label>(order),
                        nph::vector_to_np<double>(saliencies),
                        nph::vector_2d_to_np<FVal>(bcfeats));
}

bp::tuple
train_bc(bp::list const &bcModels,
         np::ndarray const &bcFeats, // Features for boundary classifier
         bp::list const &images,     // LAB, HSV, SIFT codes, etc..
         np::ndarray const &truth,
         np::ndarray const &gpbImage, // gPb, UCM, etc..
         bp::list const &histogramBins, bp::list const &histogramLowerValues,
         bp::list const &histogramHigherValues,
         bp::list const &boundaryShapeThresholds,
         bool const &normalizeSizeLength, bool const &useLogOfShape,
         bool useSimpleFeatures) {

  auto truth_itk = nph::np_to_itk_label(truth);
}
