// #include "glia_base.hxx"
// #include "alg/rf.hxx"
#include "glia_base.hxx"
#include "pyglia.hxx"
#include "shogun_helpers.hxx"
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/RandomForest.h>

namespace sg = shogun;
/*-------------------------------------------------------
Trains a RF model and return it as a string (JSON format)

Parameters:

  X: input features
  Y: labels
  label_category_idx: In each feature vector of X, indicates the index that
determines the category of sample n_trees: number of trees to train node_size:
number of trees to train sample_size_ratio: sample ratio per tree num_features:
num of features to examine at each node. 0 for sqrt(D) balance: Whether to
balance samples
---------------------------------------------------------*/

bp::list MyHmt::train_rf_operation(np::ndarray const &X_, np::ndarray const &Y_) {

  // sg::init_shogun_with_defaults();
  if (X_.shape(0) != Y_.shape(0) || Y_.get_nd() != 1) {
    glia::perr("Error: incorrect matrices dimension...");
  }

  auto X = np_to_shogun_feats<double>(X_);
  auto Y = np_to_shogun_labels<int>(Y_);

  // std::cout << "Xsg count: " << X.use_count() << std::endl;

  auto X_cat = std::make_shared<CategorizedFeatures>(X, 0, 1);

  std::cout << "X_cat 0: " << std::endl;
  X_cat->get(0)->get_feature_matrix().display_matrix();
  std::cout << "X_cat 1: " << std::endl;
  X_cat->get(1)->get_feature_matrix().display_matrix();
  // std::cout << "X_cat 2: " << std::endl;
  // X_cat->get(2)->get_feature_matrix().display_matrix();


  // std::cout << "cat_X count: " << X_cat.use_count() << std::endl;

  // X_cat->display_feats();

  // bc->rand_forest[0]->set_labels(Ysg.get());
  // auto f_type = sg::SGVector<bool>(Xsg->get_num_features());
  // f_type.display_vector();
  // bc->rand_forest[0]->set_feature_types(f_type);
  // std::cout << "train" << std::endl;
  // bc->rand_forest[0]->train(Xsg.get());
  // std::cout << "ok" << std::endl;

  // std::cout << "set_labels" << std::endl;
  // bc->rand_forest[0]->set_labels(Ysg.get());
  // std::cout << "num_bags: " << bc->rand_forest[0]->get_num_bags() << std::endl;
  // std::cout << "num_bags: " << this->n_trees << std::endl;
  // bc->train(X_cat, Y);
  auto serial_str = bc->to_serialized();
  // for(int i=0; i<serial_str.size(); ++i){
  //   std::cout << "cat. " << i << std::endl;
  //   std::cout << serial_str[i] << std::endl;
  // }


  return std_vector_to_list(serial_str);
}
