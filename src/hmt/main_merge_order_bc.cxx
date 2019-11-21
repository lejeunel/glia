#include "alg/nn.hxx"
#include "alg/rf.hxx"
#include "hmt/bc_feat.hxx"
#include "hmt/bc_label.hxx"
#include "np_helpers.hxx"
#include "pyglia.hxx"
#include "util/struct_merge_bc.hxx"
#include "util/text_cmd.hxx"
#include "util/text_io.hxx"
#include <chrono> 

using namespace glia;
using namespace glia::hmt;
using namespace std::chrono; 

using LabelImageType = LabelImage<DIMENSION>;
using RealImageType = RealImage<DIMENSION>;
using ImagePairs =
    std::vector<hmt::ImageHistPair<RealImage<DIMENSION>::Pointer>>;
typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;

/*-------------------------------------------------------
  Use a trained boundary classifier to generate a merge order

Parameters:
X: Features of boundary samples
spLabels: Integer map of superpixel segmentation
vecImagePairs: A list of images (holding image features) with corresponding histogram parameters (range, bins)
gpbImage: global probability boundary
normalizeSizeLength: see paper, default to true
useLogOfShapes: see paper, default to true
---------------------------------------------------------*/

std::tuple<std::vector<TTriple<Label>>, std::vector<double>>
merge_order_bc_operation(
    LabelImageType::Pointer spLabels, // SP labels
    std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>> const
        &vecImagePairs, // LAB, HSV, SIFT codes, etc..
    RealImageType::Pointer const &gpbImage, // gPb, UCM, etc..
    bool const &useLogOfShape, bool const &useSimpleFeatures,
    std::shared_ptr<glia::alg::EnsembleRandomForest> bc,
                         double const& cat_thr) {

  std::vector<double> boundaryThresholds;

  auto mask = LabelImage<DIMENSION>::Pointer(nullptr);

  // Load and set up images
  std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>> vecBoundaryPairs,
      vecLabelPairs;

  // Set up normalizing area/length
  double normalizingArea = getImageVolume(spLabels);
  double normalizingLength = getImageDiagonal(spLabels);

  RegionMap rmap(spLabels, mask, false);
  // Boundary feature compute
  std::unordered_map<std::pair<Label, Label>, std::vector<FVal>> bcfmap;
  auto fBcFeat = [normalizingArea, normalizingLength, &gpbImage, &vecImagePairs,
                  &vecBoundaryPairs, &vecLabelPairs, &bcfmap, useLogOfShape,
                  useSimpleFeatures, boundaryThresholds](
                     std::vector<FVal> &data, RegionMap::Region const &reg0,
                     RegionMap::Region const &reg1,
                     RegionMap::Region const &reg2, Label r0, Label r1,
                     Label r2) {
    auto rf0 = std::make_shared<RegionFeats>();
    auto rf1 = std::make_shared<RegionFeats>();
    auto rf2 = std::make_shared<RegionFeats>();
    rf0->generate(reg0, normalizingArea, normalizingLength, gpbImage,
                  boundaryThresholds, vecImagePairs, vecLabelPairs,
                  vecBoundaryPairs, nullptr);
    rf1->generate(reg1, normalizingArea, normalizingLength, gpbImage,
                  boundaryThresholds, vecImagePairs, vecLabelPairs,
                  vecBoundaryPairs, nullptr);
    rf2->generate(reg2, normalizingArea, normalizingLength, gpbImage,
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
    bcf.x0.generate(b, normalizingLength, *bcf.x1, *bcf.x2, *bcf.x3, gpbImage,
                    boundaryThresholds, vecLabelPairs);
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

  // Boundary predictor (lambda function passed to merging algorithm)
  auto fBcPred = [bc, cat_thr](std::vector<double> const &data) {
    // auto start = high_resolution_clock::now(); 
    auto data_ = SGVector<double>(data.front(), data.size());
    auto cat = categorize_sample<double>(data_, 0, 1, cat_thr);
    auto data_mat = SGMatrix<double>(data_);
    auto data__ = std::make_shared<DenseFeatures<double>>(data_mat);
    auto res =  bc->predict(data__, cat);
    // auto stop = high_resolution_clock::now(); 
    // auto duration = duration_cast<milliseconds>(stop - start); 
    // std::cout << "predicted in " << duration.count() << "ms" << std::endl;
    return res;
  };

  // Generate merging orders
  std::vector<TTriple<Label>> order;
  std::vector<double> saliencies;
  genMergeOrderGreedyUsingBoundaryClassifier<std::vector<FVal>>(
      order, saliencies, spLabels, mask, fBcFeat, fBcPred,
      f_true<TBoundaryTable<std::vector<FVal>, RegionMap> &,
             TBoundaryTable<std::vector<FVal>, RegionMap>::iterator>);

  // store new boundary classifier feats
  // std::vector<std::vector<FVal>> bcfeats;
  // bcfeats.reserve(order.size());
  // for (auto const &m : order) {
  //   bcfeats.push_back(bcfmap.find(std::make_pair(m.x0, m.x1))->second);
  // }

  return std::make_tuple(order, saliencies);
}

bp::tuple MyHmt::merge_order_bc_wrp(
    np::ndarray const &spLabels, // SP labels
    bp::list const &images,      // LAB, HSV, SIFT codes, etc..
    np::ndarray const &gpbImage, // gPb, UCM, etc..
    bp::list const &histogramBins, bp::list const &histogramLowerValues,
    bp::list const &histogramHigherValues, bool const &useLogOfShape,
    double const &cat_thr) {

  auto spLabels_itk = nph::np_to_itk_label(spLabels);
  auto gpbImage_itk = nph::np_to_itk_real(gpbImage);

  // Set histogram ranges and bins
  auto vecImagePairs = nph::lists_to_image_hist_pair(
      images, histogramBins,
      histogramLowerValues,
      histogramHigherValues);

  auto out =
      merge_order_bc_operation(spLabels_itk, vecImagePairs,
                               gpbImage_itk, useLogOfShape, false, this->bc,
                               cat_thr);
  return bp::make_tuple(nph::vector_triple_to_np<Label>(std::get<0>(out)),
                        nph::vector_to_np<double>(std::get<1>(out)));
}
