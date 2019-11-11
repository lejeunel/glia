#include "pyglia.hxx"
#include <boost/python.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <shogun/base/init.h>

namespace bp = boost::python;
namespace np = boost::python::numpy;
BOOST_PYTHON_MODULE(libglia) {
  Py_Initialize();
  np::initialize();
  shogun::init_shogun_with_defaults();

  bp::class_<MyHmt>("hmt", bp::init<int, int, double, bool>())
      .def("watershed", &MyHmt::watershed_operation,
           bp::args("image", "level", "relabel"),
           "Generate watershed segmentation")
      .def("pre_merge", &MyHmt::pre_merge_operation,
           bp::args("label", "pbArray", "sizeThresholds", "rpbThreshold",
                    "relabel"),
           "Merge labels to eliminate small elements")

      .def("merge_order_pb", &MyHmt::merge_order_pb_operation,
           bp::args("label", "pbArray", "bd_intens_stats_type"),
           "Perform greedy merge according to boundary probability")

      .def("merge_order_bc", &MyHmt::merge_order_bc_operation,
           bp::args("label", "images", "truth", "pbArray",
                    "histogramBins", "histogramLowerValues",
                    "histogramHigherValues", "useLogOfShapes", "useSimpleFeatures"),
           "Perform greedy merge according to boundary probability")

      .def("bc_feat", &MyHmt::bc_feat_operation,
           bp::args("mergeList", "salienciesArray", "labelImages", "Images",
                    "boundaryImages", "histogramBins", "histogramLowerValues",
                    "histogramHigherValues", "initialSaliency", "saliencBias",
                    "boundaryShapeThresholds", "normalizesizelength",
                    "useLogOfShapes"),
           "Generate features for boundary classifier")

      .def("train_rf", &MyHmt::train_rf_operation,
           bp::args("X", "Y"),
           "Train RF classifier")

      .def("bc_label_ri", &MyHmt::bc_label_ri_operation,
           bp::args("mergeOrderList", "labels", "groundtruth", "mask",
                    "usePairF1", "globalOpt", "optSplit", "tweak",
                    "maxPrecDrop"),
           "Generate for each clique a label indicating split/merge")

      .def("hello", &MyHmt::hello);
}
