#include "hmt/bc_feat.hxx"
#include "type/region_map.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "util/mp.hxx"
#include "hmt/hmt_util.hxx"
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

//For a given merge order and saliency metric, compute for each clique
//a feature vector

//Parameters

//Input initial segmentation image
//Input merging order
//Input merging saliency  (optional)")
//Input real image  (optional)")
//Input real image histogram bins")
//Input real image histogram lowers")
//Input real image histogram uppers")
//Input region label image file names(s) (optional)")
//Input region label image histogram bins")
//Input region label image histogram lowers")
//Input region label image histogram uppers")
//Input excl. region image file name(s) (optional)")
//Input excl. region image histogram bins")
//Input excl. region image histogram lowers")
//Input excl. boundary image histogram uppers")
//Input excl. boundary image file name(s) (optional)")
//Input excl. boundary image histogram bins")
//Input excl. boundary image histogram lowers")
//Input excl. boundary image histogram uppers")
//Boundary image file for image-based shape features")
//Input mask image file name")
//Initial saliency [default: 1.0]")
//Saliency bias [default: 1.0]")
//Thresholds for image-based shape features (e.g. --bt 0.2 0.5 0.8)")
//Whether to normalize size and length [default: false]")
//Whether to use logarithms of shape as features [default: false]")
//Whether to only use simplified features (following arXiv paper) "
//Output boundary feature array
np::ndarray bc_feat_operation (bp::list const& mergeOrderList,
                        np::ndarray const& salienciesArray,
                        bp::list const& labelImages, // SP labels, etc..
                        bp::list const& Images, // LAB, HSV, SIFT codes, etc..
                        bp::list  const& boundaryImages, //gPb, UCM, etc..
                        np::ndarray  const& maskArray,
                        bp::list const& histogramBins,
                        bp::list const& histogramLowerValues,
                        bp::list  const& histogramHigherValues,
                        double  const& initialSaliency,
                        double  const& saliencyBias,
                        bp::list  const& boundaryShapeThresholds,
                        bool  const& normalizeSizeLength,
                        bool  const& useLogOfShapes)
{
  using LabelImageType =  LabelImage<DIMENSION>;
  using RealImageType =  RealImage<DIMENSION>;

  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;

  // Load and set up images
  std::vector<ImageHistPair<RealImage<DIMENSION>::Pointer>>
      vecImages, vecLabelImages, vecBoundaryImages;

  LabelImageType::Pointer segImage = np_to_itk_label(bp::extract<np::ndarray>(labelImages[0]));

  RealImageType::Pointer pbImage = np_to_itk_real(bp::extract<np::ndarray>(boundaryImages[0]));

  RealImageType::Pointer image = np_to_itk_real(bp::extract<np::ndarray>(Images[0]));

  // Set histogram ranges and bins
  vecLabelImages = lists_to_image_hist_pair(labelImages,
                                bp::extract<int>(histogramBins[0]),
                                bp::extract<double>(histogramLowerValues[0]),
                                bp::extract<double>(histogramHigherValues[0]));

  vecImages = lists_to_image_hist_pair(Images,
                                bp::extract<int>(histogramBins[1]),
                                bp::extract<double>(histogramLowerValues[1]),
                                bp::extract<double>(histogramHigherValues[1]));

  vecBoundaryImages = lists_to_image_hist_pair(boundaryImages,
                                bp::extract<int>(histogramBins[2]),
                                bp::extract<double>(histogramLowerValues[2]),
                                bp::extract<double>(histogramHigherValues[2]));

  LabelImageType::Pointer mask = (maskArray.get_nd() == 1)?
    LabelImageType::Pointer(nullptr):
    np_to_itk_label(maskArray);

  // Set up normalizing area/length
  double normalizingArea =
      normalizeShape ? getImageVolume(segImage) : 1.0;
  double normalizingLength =
      normalizeShape ? getImageDiagonal(segImage) : 1.0;

  auto order = np_to_vector_triple<Label>(mergeOrderList);

  auto saliencies = np_to_vector<double>(salienciesArray);
  std::unordered_map<Label, double> saliencyMap;
  if (!is_empty(salienciesArray)) {
    genSaliencyMap(saliencyMap,
                   order,
                   saliencies,
                   initialSaliency,
                   saliencyBias);
  }

  // Generate region features
  std::cout << "generating region features" << std::endl;
  RegionMap rmap(segImage, mask, order, false);
  int rn = rmap.size();
  std::vector<std::pair<Label, std::shared_ptr<RegionFeats>>> rfeats(rn);
  parfor(rmap, true, [
      &rfeats, &vecImages, &vecLabelImages,
      &vecBoundaryImages, &pbImage, &saliencyMap,
      normalizingArea, normalizingLength](
          RegionMap::const_iterator rit, int i) {
           rfeats[i].first = rit->first;
           rfeats[i].second = std::make_shared<RegionFeats>();
           rfeats[i].second->generate(
               rit->second, normalizingArea, normalizingLength,
               pbImage, boundaryThresholds,
               vecImages, vecLabelImages, vecBoundaryImages,
               ccpointer(saliencyMap, rit->first));
         }, 0);

  std::unordered_map<Label, std::shared_ptr<RegionFeats>> rfmap;
  for (auto const& rfp : rfeats) { rfmap[rfp.first] = rfp.second; }

  // Generate boundary classifier features
  std::cout << "generating boundary features" << std::endl;
  int bn = order.size();
  std::vector<BoundaryClassificationFeats> bfeats(bn);
  parfor(0, bn, true, [
                       &rmap, &order, &bfeats, &rfmap, &vecBoundaryImages,
                       &pbImage, normalizingLength](int i) {
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
                                             pbImage,
                                             boundaryThresholds,
                                             vecBoundaryImages);
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

  return vector_2d_to_np<FVal>(pickedFeats);
}

//int main (int argc, char* argv[])
//{
//  std::vector<std::string>
//      _rbImageFiles, _rlImageFiles, _rImageFiles, _bImageFiles;
//  std::vector<unsigned int>
//      _rbHistBins, _rlHistBins, _rHistBins, _bHistBins;
//  std::vector<double>
//      _rbHistLowers, _rlHistLowers, _rHistLowers, _bHistLowers;
//  std::vector<double>
//      _rbHistUppers, _rlHistUppers, _rHistUppers, _bHistUppers;
//  bpo::options_description opts("Usage");
//  opts.add_options()
//      ("help", "Print usage info")
//      ("segImage,s", bpo::value<std::string>(&segImageFile)->required(),
//       "Input initial segmentation image file name")
//      ("mergeOrder,o", bpo::value<std::string>(&mergeOrderFile)->required(),
//       "Input merging order file name")
//      ("saliency,y", bpo::value<std::string>(&saliencyFile),
//       "Input merging saliency file name (optional)")
//      ("rbi", bpo::value<std::vector<std::string>>(&_rbImageFiles),
//       "Input real image file name(s) (optional)")
//      ("rbb", bpo::value<std::vector<unsigned int>>(&_rbHistBins),
//       "Input real image histogram bins")
//      ("rbl", bpo::value<std::vector<double>>(&_rbHistLowers),
//       "Input real image histogram lowers")
//      ("rbu", bpo::value<std::vector<double>>(&_rbHistUppers),
//       "Input real image histogram uppers")
//      ("rli", bpo::value<std::vector<std::string>>(&_rlImageFiles),
//       "Input region label image file names(s) (optional)")
//      ("rlb", bpo::value<std::vector<unsigned int>>(&_rlHistBins),
//       "Input region label image histogram bins")
//      ("rll", bpo::value<std::vector<double>>(&_rlHistLowers),
//       "Input region label image histogram lowers")
//      ("rlu", bpo::value<std::vector<double>>(&_rlHistUppers),
//       "Input region label image histogram uppers")
//      ("ri", bpo::value<std::vector<std::string>>(&_rImageFiles),
//       "Input excl. region image file name(s) (optional)")
//      ("rb", bpo::value<std::vector<unsigned int>>(&_rHistBins),
//       "Input excl. region image histogram bins")
//      ("rl", bpo::value<std::vector<double>>(&_rHistLowers),
//       "Input excl. region image histogram lowers")
//      ("ru", bpo::value<std::vector<double>>(&_rHistUppers),
//       "Input excl. boundary image histogram uppers")
//      ("bi", bpo::value<std::vector<std::string>>(&_bImageFiles),
//       "Input excl. boundary image file name(s) (optional)")
//      ("bb", bpo::value<std::vector<unsigned int>>(&_bHistBins),
//       "Input excl. boundary image histogram bins")
//      ("bl", bpo::value<std::vector<double>>(&_bHistLowers),
//       "Input excl. boundary image histogram lowers")
//      ("bu", bpo::value<std::vector<double>>(&_bHistUppers),
//       "Input excl. boundary image histogram uppers")
//      ("pb", bpo::value<std::string>(&pbImageFile)->required(),
//       "Boundary image file for image-based shape features")
//      ("maskImage,m", bpo::value<std::string>(&maskImageFile),
//       "Input mask image file name")
//      ("s0", bpo::value<double>(&initSal),
//       "Initial saliency [default: 1.0]")
//      ("sb", bpo::value<double>(&salBias),
//       "Saliency bias [default: 1.0]")
//      ("bt",
//       bpo::value<std::vector<double>>(&boundaryThresholds)->multitoken(),
//       "Thresholds for image-based shape features (e.g. --bt 0.2 0.5 0.8)")
//      ("ns,n", bpo::value<bool>(&normalizeShape),
//       "Whether to normalize size and length [default: false]")
//      ("logs,l", bpo::value<bool>(&useLogShape),
//       "Whether to use logarithms of shape as features [default: false]")
//      ("simpf", bpo::value<bool>(&useSimpleFeatures),
//       "Whether to only use simplified features (following arXiv paper) "
//       "[default: false]")
//      ("bfeat,b", bpo::value<std::string>(&bfeatFile),
//       "Output boundary feature file name (optional)");
//  if (!parse(argc, argv, opts))
//  { perr("Error: unable to parse input arguments"); }
//  rbImageFiles.reserve(_rbImageFiles.size());
//  rlImageFiles.reserve(_rlImageFiles.size());
//  rImageFiles.reserve(_rImageFiles.size());
//  bImageFiles.reserve(_bImageFiles.size());
//  for (int i = 0; i < _rbImageFiles.size(); ++i) {
//    rbImageFiles.emplace_back(_rbImageFiles[i], _rbHistBins[i],
//                              _rbHistLowers[i], _rbHistUppers[i]);
//  }
//  for (int i = 0; i < _rlImageFiles.size(); ++i) {
//    rlImageFiles.emplace_back(_rlImageFiles[i], _rlHistBins[i],
//                              _rlHistLowers[i], _rlHistUppers[i]);
//  }
//  for (int i = 0; i < _rImageFiles.size(); ++i) {
//    rImageFiles.emplace_back(_rImageFiles[i], _rHistBins[i],
//                             _rHistLowers[i], _rHistUppers[i]);
//  }
//  for (int i = 0; i < _bImageFiles.size(); ++i) {
//    bImageFiles.emplace_back(_bImageFiles[i], _bHistBins[i],
//                             _bHistLowers[i], _bHistUppers[i]);
//  }
//  return operation() ? EXIT_SUCCESS: EXIT_FAILURE;
//}
