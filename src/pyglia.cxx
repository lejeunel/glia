#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <shogun/base/init.h>
#include "pyglia.hxx"

namespace bp = boost::python;
namespace np = boost::python::numpy;


BOOST_PYTHON_MODULE(libglia)
{
  Py_Initialize();
  np::initialize();
  shogun::init_shogun_with_defaults();

  bp::def("watershed", watershed_operation,
      bp::args("image",
               "level",
               "relabel"),
      "Generate watershed segmentation");

  bp::def("pre_merge", pre_merge_operation,
      bp::args("label",
               "pbArray",
               "sizeThresholds",
               "rpbThreshold",
               "relabel"),
      "Merge labels to eliminate small elements");

  bp::def("merge_order_pb", merge_order_pb_operation,
      bp::args("label",
               "pbArray",
               "bd_intens_stats_type"),
      "Perform greedy merge according to boundary probability");

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
      "Generate features for boundary classifier");

  bp::def("train_rf", train_rf_operation,
      bp::args("X",
               "Y",
               "n_trees",
               "node_size",
               "sample_size_ratio",
               "num_features",
               "balance"),
      "Train RF classifier");

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
      "Generate for each clique a label indicating split/merge");

  bp::def("test_conversion_shogun_feats", test_conversion_shogun_feats,
      bp::args("A"),
      "Test ndarray to armadillo matrix");

  bp::def("test_conversion_shogun_labels", test_conversion_shogun_labels,
      bp::args("A"),
      "Test ndarray to armadillo matrix");

  bp::def("test_conversion_itk", test_conversion_itk,
      bp::args("inputImage_np",
               "inputImageStr",
               "outputImageStr_itk_rgb",
               "outputImageStr_itk_real",
               "outputImageStr_np_rgb",
               "outputImageStr_np_real"),
      "Test ndarray to ITK image");
}
