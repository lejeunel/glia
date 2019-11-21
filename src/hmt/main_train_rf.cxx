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

void MyHmt::train_rf_operation(np::ndarray const &X_, np::ndarray const &Y_) {

  // sg::init_shogun_with_defaults();
  if (X_.shape(0) != Y_.shape(0) || Y_.get_nd() != 1) {
    glia::perr("Error: incorrect matrices dimension...");
  }

  auto X = np_to_shogun_feats<double>(X_);
  auto Y = np_to_shogun_labels<int>(Y_);

  auto X_cat = std::make_shared<CategorizedFeatures>(X, 0, 1);

  bc->train(X_cat, Y);

  cat_threshold = X_cat->get_threshold();

}
