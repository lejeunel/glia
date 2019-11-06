#include "util/image_io.hxx"
#include "util/struct_merge.hxx"
#include "util/text_io.hxx"
#include "util/text_cmd.hxx"
#include "np_helpers.hxx"
#include "glia_image.hxx"
#include "type/tuple.hxx"

using namespace glia;
namespace np = boost::python::numpy;
namespace bp = boost::python;


//"Input initial segmentation image
//"Input boundary probability image
//"Boundary intensity stats type (1: median, 2: mean) [default: 1]")
//"Output merging order file name (optional)")
//"Output merging saliency file name (optional)");

bp::tuple merge_order_pb_operation (np::ndarray const& labelArray,
                                    np::ndarray const& pbArray,
                                    int const & bd_intens_stats_type)
{

  using LabelImageType =  LabelImage<DIMENSION>;
  using RealImageType =  RealImage<DIMENSION>;

  std::vector<TTriple<Label>> order;
  std::vector<double> saliencies;

  LabelImageType::Pointer segImage = np_to_itk_label(labelArray);
  RealImageType::Pointer pbImage = np_to_itk_real(pbArray);

  LabelImageType::Pointer mask = LabelImageType::Pointer(nullptr);

  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap rmap(segImage, mask, true); // Only use contours
  if (bd_intens_stats_type == 1) {
    genMergeOrderGreedyUsingPbApproxMedian(
        order, saliencies, rmap, false, pbImage, f_true
        <TBoundaryTable<std::vector<double>, RegionMap>&,
        TBoundaryTable<std::vector<double>, RegionMap>::iterator>,
        f_null<std::vector<double>&, Label, Label>);
  } else if (bd_intens_stats_type == 2) {
    genMergeOrderGreedyUsingPbMean(
        order, saliencies, rmap, false, pbImage, f_true
        <TBoundaryTable<std::pair<double, int>, RegionMap>&,
        TBoundaryTable<std::pair<double, int>, RegionMap>::iterator>);
  } else { perr("Error: unsupported boundary stats type..."); }

  return bp::make_tuple(vector_triple_to_np<Label>(order),
                        vector_to_np<double>(saliencies));
}

