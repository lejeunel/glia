#ifndef _pyglia_hxx_
#define _pyglia_hxx_

#include "alg/rf.hxx"
#include "np_helpers.hxx"
#include "shogun_helpers.hxx"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::python::numpy;
using namespace boost;

class MyHmt {
private:
  std::shared_ptr<glia::alg::EnsembleRandomForest> bc;
  int n_trees;
  int num_features;
  double sample_size_ratio;
  bool balance;
  int n_cats;

public:
  static std::shared_ptr<MyHmt> create() {
    return std::shared_ptr<MyHmt>(new MyHmt);
  }
  void config(int n_cats_ = 3, int const &n_trees_ = 100,
              int const &num_features_ = 0,
              double const &sample_size_ratio_ = 0.7,
              bool const &balance_ = true) {
    n_trees = n_trees_;
    num_features = num_features_;
    sample_size_ratio = sample_size_ratio_;
    balance = balance_;
    n_cats = n_cats_;

    bc = std::make_shared<glia::alg::EnsembleRandomForest>(
        n_cats, n_trees, sample_size_ratio, num_features, balance);
  };

  std::string hello() { return "Just nod if you can hear me!"; }
  void set_model(bp::list models){};
  np::ndarray watershed_operation(np::ndarray const &, double, bool);
  np::ndarray pre_merge_operation(np::ndarray const &, np::ndarray const &,
                                  bp::list const &, double, bool);
  bp::tuple merge_order_pb_operation(np::ndarray const &, np::ndarray const &,
                                     int const &);

  // models is a list of lists
  void load_models(bp::list const &models) {

    // re-create models
    bc->models.clear();
    for (int i = 0; i < bp::len(models); ++i)
      bc->models.push_back(std::make_shared<glia::alg::MyRandomForest>(
          n_cats, n_trees, sample_size_ratio, num_features, balance));

    // load models
    for (int i = 0; i < bp::len(models); ++i) {

      bp::list model = bp::extract<bp::list>(models[i]);
      auto models_vec = list_to_std_vector<std::string>(models[i]);
      bc->models[i]->from_serialized(models_vec);
    }
  };

  bp::list get_models() {
    auto serial_vec = bc->to_serialized();
    return std_2d_vector_to_list(serial_vec);
  };

  np::ndarray bc_label_ri_operation(bp::list const &, np::ndarray const &,
                                    np::ndarray const &,
                                    bool const &, int const &, bool const &,
                                    bool const &, double const &);
  np::ndarray bc_feat_operation(bp::list const &, np::ndarray const &,
                                np::ndarray const &, // SP labels
                                bp::list const &, // LAB, HSV, SIFT codes, etc..
                                np::ndarray const &, // gPb, UCM, etc..
                                bp::list const &, bp::list const &,
                                bp::list const &, double const &,
                                double const &, bp::list const &, bool const &,
                                bool const &);
  bp::tuple merge_order_bc_operation(
      np::ndarray const &, // boundary features of previous run
      np::ndarray const &, // SP labels
      bp::list const &,    // LAB, HSV, SIFT codes, etc..
      np::ndarray const &,
      np::ndarray const &, // gPb, UCM, etc..
      bp::list const &, bp::list const &, bp::list const &, bool const &,
      bool const &);

  void train_rf_operation(np::ndarray const &, np::ndarray const &);
};
#endif
