#include "glia_base.hxx"
#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/RandomForest.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/io/SerializableAsciiFile.h>
#include "np_helpers.hxx"
#include "shogun_helpers.hxx"

namespace sg = shogun;
/*-------------------------------------------------------
Trains a RF model and return it as a string (JSON format)

Parameters:

  X: input features
  Y: labels
  n_trees: number of trees to train
  node_size: number of trees to train
  sample_size_ratio: sample ratio per tree
  num_features: num of features to examine at each node. 0 for sqrt(D)
  balance: Whether to balance samples
---------------------------------------------------------*/

bool train_rf_operation (np::ndarray const& X, 
                         np::ndarray const& Y, 
                         int const& n_trees, 
                         int const& node_size, 
                         double const& sample_size_ratio, 
                         int const& num_features, 
                         bool const& balance){

  sg::init_shogun_with_defaults();
  if (X.shape(0) != Y.shape(0) || Y.get_nd() != 1)
    { perr("Error: incorrect matrices dimension..."); }

  auto Xsg = np_to_shogun_feats<float64_t>(X);
  auto Ysg = np_to_shogun_labels<int>(Y);

  sg::SGVector<double> weights(Ysg->get_num_labels());
  if(balance){

    make_balanced_weight_vector(Ysg, weights);
  }
  else{
  // std::cout << "filling weights with 1's" << std::endl;
    sg::SGVector<double>::fill_vector(weights, Ysg->get_num_labels(), 1);
  }

  std::cout << "--- weights" << std::endl;
  weights.display_vector();
  // std::cout << "--- Y" << std::endl;
  // Ysg->get_labels().display_vector();
  // std::cout << "--- X" << std::endl;
  // Xsg->get_feature_vectors().display_vector();
  auto m_vote = sg::some<sg::CMajorityVote>();
  auto rand_forest = sg::some<sg::CRandomForest>();
  rand_forest->set_num_bags(n_trees);
  rand_forest->set_weights(weights);
  rand_forest->set_bag_size(sample_size_ratio * Xsg->get_num_features());
  // std::cout << "set combination rule" << std::endl;
  rand_forest->set_combination_rule(m_vote);
  // std::cout << "ok" << std::endl;

  sg::SGVector<bool> f_type(Xsg->get_num_features());
  // std::cout << "fill_vector f_type with " << Xsg->get_num_features() << " elems" << std::endl;
  sg::SGVector<bool>::fill_vector(f_type, Xsg->get_num_features(), false);
  // std::cout << "--- f_type" << std::endl;
  // f_type.display_vector();
  // std::cout << "set feature type" << std::endl;
  rand_forest->set_feature_types(f_type);
  rand_forest->set_machine_problem_type(shogun::EProblemType::PT_MULTICLASS);
  rand_forest->set_labels(Ysg);

  // std::cout << "train" << std::endl;
  rand_forest->train(Xsg);
  // std::cout << "ok" << std::endl;
  auto file=new sg::CSerializableAsciiFile("serialized.dat", 'w');
  rand_forest->save_serializable(file);
  file->close();
  SG_UNREF(file);
  return true;
}

