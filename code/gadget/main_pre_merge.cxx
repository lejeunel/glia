#include "util/struct_merge.hxx"
#include "util/stats.hxx"
#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
#include "np_helpers.hxx"
#include "glia_image.hxx"
#include <itkImageConstIterator.h>
#include "itkImageLinearIteratorWithIndex.h"

using namespace glia;
namespace np = boost::python::numpy;
namespace bp = boost::python;

//  "Label array
//  "Input pb image file name")
//  "Input mask image file name (optional)")
//  "Region size threshold(s) (e.g. -t 50 100)")
//  "Region average boundary probability threshold")
//  "Whether to relabel image to consecutive labels [default: false]")
np::ndarray pre_merge_operation (np::ndarray const& labelArray,
                                np::ndarray const& pbArray,
                                np::ndarray const& maskArray,
                                bp::list const& sizeThresholdsList,
                                double rpbThreshold,
                                bool relabel)
{


std::vector<int> sizeThresholds = list_to_vector<int>(bp::extract<bp::object>(sizeThresholdsList));


  using LabelImageType =  LabelImage<DIMENSION>;
  using RealImageType =  RealImage<DIMENSION>;
  LabelImageType::Pointer segImage = np_to_itk_label(labelArray);

  RealImageType::Pointer pbImage = np_to_itk_real(pbArray);

  //typedef itk::ImageFileWriter<LabelImageType> Writer;
  //auto writer = Writer::New();
  //writer->SetFileName("segImage.png");
  //writer->SetInput(segImage);
  //writer->Update();
  //std::cout << "wrote segImage" << std::endl;

  LabelImageType::Pointer mask = (maskArray.get_nd() == 1)?
    LabelImageType::Pointer(nullptr):
    np_to_itk_label(maskArray);
  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap rmap(segImage, mask, false);
  std::vector<TTriple<Label>> order;
  std::vector<double> saliencies;
  std::unordered_map<Label, double> rpbs;
  genMergeOrderGreedyUsingPbMean
      (order, saliencies, rmap, true, pbImage,
       [&pbImage, &sizeThresholds, &rmap, rpbThreshold, &rpbs]
       (TBoundaryTable<std::pair<double, int>, RegionMap> const& bt,
        TBoundaryTable<std::pair<double, int>, RegionMap>::iterator btit)
       {
         Label key0 = btit->first.first;
         Label key1 = btit->first.second;
         auto const* pr0 = &rmap.find(key0)->second;
         auto const* pr1 = &rmap.find(key1)->second;
         auto sz0 = pr0->size();
         auto sz1 = pr1->size();
         if (sz0 > sz1) {
           std::swap(key0, key1);
           std::swap(pr0, pr1);
           std::swap(sz0, sz1);
         }
         if (sz0 < sizeThresholds[0]) { return true; }
         if (sizeThresholds.size() > 1) {
           if (sz0 < sizeThresholds[1]) {
             auto rpit0 = rpbs.find(key0);
             if (rpit0 != rpbs.end()) {
               if (rpit0->second > rpbThreshold) { return true; }
             } else {
               double rpb = 0.0;
               pr0->traverse(
                   [&pbImage, &rpb, rpbThreshold](
                       RegionMap::Region::Point const& p) {
                     rpb += pbImage->GetPixel(p); });
               rpb = sdivide(rpb, sz0, 0.0);
               rpbs[key0] = rpb;
               if (rpb > rpbThreshold) { return true; }
             }
           }
           if (sz1 < sizeThresholds[1]) {
             auto rpit1 = rpbs.find(key1);
             if (rpit1 != rpbs.end()) {
               if (rpit1->second > rpbThreshold) { return true; }
             } else {
               double rpb = 0.0;
               pr1->traverse(
                   [&pbImage, &rpb, rpbThreshold](
                       RegionMap::Region::Point const& p) {
                     rpb += pbImage->GetPixel(p); });
               rpb = sdivide(rpb, sz1, 0.0);
               rpbs[key1] = rpb;
               if (rpb > rpbThreshold) { return true; }
             }
           }
         }
         return false;
       });
  std::unordered_map<Label, Label> lmap;
  transformKeys(lmap, order);
  transformImage(segImage, rmap, lmap);
  if (relabel) { relabelImage(segImage, 0); }
  //if (write16) {
  //  castWriteImage<UInt16Image<DIMENSION>>
  //      (outputImageFile, segImage, compress);
  //}
  //else { writeImage(outputImageFile, segImage, compress); }
  return itk_to_np<LabelImageType, Label>(segImage);
}
