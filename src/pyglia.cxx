#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_internal_reference.hpp>
#include "pyglia.hxx"

namespace bp = boost::python;
namespace np = boost::python::numpy;


BOOST_PYTHON_MODULE(libglia)
{
  Py_Initialize();
  np::initialize();

  bp::def("watershed", watershed_operation,
      bp::args("inputImageFile",
               "level",
               "relabel"),
      "foo's docstring");

  bp::def("pre_merge", pre_merge_operation,
      bp::args("labelArray",
               "pbArray",
               "maskArray",
               "sizeThresholds",
               "rpbThreshold",
               "relabel"),
      "foo's docstring");

  bp::def("merge_order_pb", merge_order_pb_operation,
      bp::args("labelArray",
               "pbArray",
               "maskArray",
               "bd_intens_stats_type"),
      "foo's docstring");

  bp::def("bc_feat", bc_feat_operation,
      bp::args("mergeList",
               "salienciesArray",
               "labelImages",
               "Images",
               "boundaryImages",
               "histogramBins",
               "histogramLowerValues",
               "histogramHigherValues",
               "initialSaliency",
               "saliencBias",
               "boundaryShapeThresholds",
               "normalizesizelength",
               "useLogOfShapes"),
      "foo's docstring");

  bp::def("bc_label_ri", bc_label_ri_operation,
      bp::args("mergeOrderList",
               "labels",
               "groundtruth",
               "mask",
               "usePairF1",
               "globalOpt",
               "optSplit",
               "tweak",
               "maxPrecDrop"),
      "foo's docstring");

  bp::def("test_conversion", test_conversion,
      bp::args("inputImage_np",
               "inputImageStr",
               "outputImageStr_itk_rgb",
               "outputImageStr_itk_real",
               "outputImageStr_np_rgb",
               "outputImageStr_np_real"),
      "foo's docstring");
}
