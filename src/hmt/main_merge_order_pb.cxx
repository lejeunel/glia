#include "glia_image.hxx"
#include "np_helpers.hxx"
#include "pyglia.hxx"
#include "type/tuple.hxx"
#include "util/image_io.hxx"
#include "util/struct_merge.hxx"
#include "util/text_cmd.hxx"
#include "util/text_io.hxx"

using namespace glia;
namespace np = boost::python::numpy;
namespace bp = boost::python;

using LabelImageType = LabelImage<DIMENSION>;
using RealImageType = RealImage<DIMENSION>;
typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;

//"Input initial segmentation image (superpixels)
//"Input boundary probability image (contour map)
//"Boundary intensity stats type (1: median, 2: mean) [default: 1]")
std::tuple<std::vector<TTriple<Label>>, std::vector<double>>
merge_order_pb_operation(LabelImageType::Pointer segImage,
                         RealImageType::Pointer pbImage,
                         int const &bd_intens_stats_type) {

  std::vector<TTriple<Label>> order;
  std::vector<double> saliencies;

  LabelImageType::Pointer mask = LabelImageType::Pointer(nullptr);
  RegionMap rmap(segImage, mask, false);

  if (bd_intens_stats_type == 1) {
    genMergeOrderGreedyUsingPbApproxMedian(
        order, saliencies, rmap, false, pbImage,
        f_true<TBoundaryTable<std::vector<double>, RegionMap> &,
               TBoundaryTable<std::vector<double>, RegionMap>::iterator>,
        f_null<std::vector<double> &, Label, Label>);
  } else if (bd_intens_stats_type == 2) {
    genMergeOrderGreedyUsingPbMean(
        order, saliencies, rmap, false, pbImage,
        f_true<TBoundaryTable<std::pair<double, int>, RegionMap> &,
               TBoundaryTable<std::pair<double, int>, RegionMap>::iterator>);
  } else {
    perr("Error: unsupported boundary stats type...");
  }

  return std::make_tuple(order, saliencies);
}

bp::tuple MyHmt::merge_order_pb_wrp(np::ndarray const &labelArray,
                                    np::ndarray const &pbArray,
                                    int const &bd_intens_stats_type) {

  LabelImageType::Pointer segImage = nph::np_to_itk_label(labelArray);
  RealImageType::Pointer pbImage = nph::np_to_itk_real(pbArray);

  auto out_tuple =
      merge_order_pb_operation(segImage, pbImage, bd_intens_stats_type);

  return bp::make_tuple(nph::vector_triple_to_np<Label>(std::get<0>(out_tuple)),
                        nph::vector_to_np<double>(std::get<1>(out_tuple)));
}

