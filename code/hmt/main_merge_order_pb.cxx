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
//"Input mask image file name (optional)")
//"Boundary intensity stats type (1: median, 2: mean) [default: 1]")
//"Output merging order file name (optional)")
//"Output merging saliency file name (optional)");

bp::tuple merge_order_pb_operation (np::ndarray const& labelArray,
                                    np::ndarray const& pbArray,
                                    np::ndarray const& maskArray,
                                    int const & bd_intens_stats_type)
{

  using LabelImageType =  LabelImage<DIMENSION>;
  using RealImageType =  RealImage<DIMENSION>;

  std::vector<TTriple<Label>> order;
  std::vector<double> saliencies;

  LabelImageType::Pointer segImage = np_to_itk_label(labelArray);
  RealImageType::Pointer pbImage = np_to_itk_real(pbArray);

  LabelImageType::Pointer mask = (maskArray.get_nd() == 1)?
    LabelImageType::Pointer(nullptr):
    np_to_itk_label(maskArray);

  typedef TRegionMap<Label, Point<DIMENSION>> RegionMap;
  RegionMap rmap(segImage, mask, true); // Only use contours
  if (bd_intens_stats_type == 1) {
    genMergeOrderGreedyUsingPbApproxMedian(
        order, saliencies, rmap, false, pbImage, f_true
        <TBoundaryTable<std::vector<double>, RegionMap>&,
        TBoundaryTable<std::vector<double>, RegionMap>::iterator>,
        f_null<std::vector<double>&, Label, Label>);
  } else if (bd_intens_stats_type == 2) {
    std::cout << "using pb mean" << std::endl,
    genMergeOrderGreedyUsingPbMean(
        order, saliencies, rmap, false, pbImage, f_true
        <TBoundaryTable<std::pair<double, int>, RegionMap>&,
        TBoundaryTable<std::pair<double, int>, RegionMap>::iterator>);
  } else { perr("Error: unsupported boundary stats type..."); }

  //if (!mergeOrderFile.empty()) { writeData(mergeOrderFile, order, "\n"); }
  //if (!saliencyFile.empty()) { writeData(saliencyFile, saliencies, "\n"); }

  return bp::make_tuple(vector_triple_to_np<Label>(order),
                        vector_to_np<double>(saliencies));

}


//int main (int argc, char* argv[])
//{
//  bpo::options_description opts("Usage");
//  opts.add_options()
//      ("help", "Print usage info")
//      ("segImage,s", bpo::value<std::string>(&segImageFile)->required(),
//       "Input initial segmentation image file name")
//      ("pbImage,p", bpo::value<std::string>(&pbImageFile)->required(),
//       "Input boundary probability image file name")
//      ("maskImage,m", bpo::value<std::string>(&maskImageFile),
//       "Input mask image file name (optional)")
//      ("type,t", bpo::value<int>(&type),
//       "Boundary intensity stats type (1: median, 2: mean) [default: 1]")
//      ("mergeOrder,o", bpo::value<std::string>(&mergeOrderFile),
//       "Output merging order file name (optional)")
//      ("saliency,y", bpo::value<std::string>(&saliencyFile),
//       "Output merging saliency file name (optional)");
//  return parse(argc, argv, opts) && operation() ?
//      EXIT_SUCCESS : EXIT_FAILURE;
//}
