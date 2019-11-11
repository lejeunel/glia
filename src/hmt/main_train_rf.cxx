// #include "glia_base.hxx"
// #include "alg/rf.hxx"
#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/RandomForest.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/io/SerializableAsciiFile.h>
#include "shogun_helpers.hxx"
#include "glia_base.hxx"
#include "pyglia.hxx"

namespace sg = shogun;
/*-------------------------------------------------------
Trains a RF model and return it as a string (JSON format)

Parameters:

  X: input features
  Y: labels
  label_category_idx: In each feature vector of X, indicates the index that determines the category of sample
  n_trees: number of trees to train
  node_size: number of trees to train
  sample_size_ratio: sample ratio per tree
  num_features: num of features to examine at each node. 0 for sqrt(D)
  balance: Whether to balance samples
---------------------------------------------------------*/

bool MyHmt::train_rf_operation (np::ndarray const& X, 
                                np::ndarray const& Y){ 

  if (X.shape(0) != Y.shape(0) || Y.get_nd() != 1)
    { glia::perr("Error: incorrect matrices dimension..."); }

  auto Xsg = np_to_shogun_feats<double>(X);
  auto Ysg = np_to_shogun_labels<int>(Y);

  // auto cat_features = categorize_features(Xsg, 0);

  // Xsg->get_feature_matrix().display_matrix();
  auto cat_X = new CategorizedFeatures(Xsg, 0, 1);

  cat_X->display_feats();

  // bc.get_params(0);
  std::cout << "ok" << std::endl;
  auto file = new sg::CSerializableAsciiFile("serialized.dat", 'w');
  bc->rand_forest[0]->save_serializable(file);
  file->print_serializable();
  file->close();
  SG_UNREF(file);
  // Xsg->free_features();

  bc->train(Ysg);




  // free up memory
  // SG_UNREF(svm);

  // sg::exit_shogun();
  return true;
}

