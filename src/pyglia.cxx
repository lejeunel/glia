#include "shogun_helpers.hxx"
#include "pyglia.hxx"
#include <boost/python.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_internal_reference.hpp>

namespace bp = boost::python;
namespace np = boost::python::numpy;

BOOST_PYTHON_MODULE(libglia) {
  Py_Initialize();
  np::initialize();
  // shogun::init_shogun_with_defaults();
  // shogun::init_shogun();

  bp::class_<MyHmt, std::shared_ptr<MyHmt>>("hmt", bp::init<>())
    .def("create", &MyHmt::create )
    .def("load_models", &MyHmt::load_models,
         bp::args("models_list"), "")
    .def("config", &MyHmt::config,
         bp::args("n_cats", "n_trees", "num_features", "sample_size_ratio", "balance"), "")
      .def("watershed", &MyHmt::watershed_operation,
           bp::args("image", "level", "relabel"),
           "Generate watershed segmentation")

      .def("merge_order_pb", &MyHmt::merge_order_pb_wrp,
           bp::args("label", "pbArray", "bd_intens_stats_type"),
           "Perform greedy merge according to boundary probability")

      .def("merge_order_bc", &MyHmt::merge_order_bc_wrp,
           bp::args("X_prev", "label", "images", "pbArray",
                    "histogramBins", "histogramLowerValues",
                    "histogramHigherValues", "useLogOfShapes"),
           "Perform greedy merge according to boundary probability")

      .def("bc_feat", &MyHmt::bc_feat_wrp,
           bp::args("mergeList", "salienciesArray", "labelImages", "Images",
                    "boundaryImages", "histogramBins", "histogramLowerValues",
                    "histogramHigherValues", "initialSaliency", "saliencBias",
                    "boundaryShapeThresholds", "normalizesizelength",
                    "useLogOfShapes"),
           "Generate features for boundary classifier")

      .def("train_rf", &MyHmt::train_rf_operation,
           bp::args("X", "Y"),
           "Train RF classifier")

      .def("get_models", &MyHmt::get_models,
           "Return models in JSON format")

      .def("bc_label_ri", &MyHmt::bc_label_ri_wrp,
           bp::args("mergeOrderList", "labels", "groundtruth", 
                    "usePairF1", "globalOpt", "optSplit", "tweak",
                    "maxPrecDrop"),
           "Generate for each clique a label indicating split/merge")

      .def("hello", &MyHmt::hello);
}
