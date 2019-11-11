#ifndef _glia_alg_rf_hxx_
#define _glia_alg_rf_hxx_

#include "shogun_helpers.hxx"
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/machine/BaggingMachine.h>
#include <shogun/machine/RandomForest.h>

typedef sg::Some<sg::CRandomForest> RandomForestModel;

namespace glia {
namespace alg {

class RandomForest {
public:
  CategorizedFeatures *feats;
  // Apply weights to samples for class imbalance
  bool balance;
  // ratio of total samples to bag for each tree
  double sample_size_ratio;
  // number of features to take at each node split (0 for sqrt(D))
  int num_features;
  // number of trees to train on each random forest
  int n_trees;
  // number of categories (3)
  int n_cats;

  std::vector<RandomForestModel> rand_forest;

  RandomForest() {}

  // Construct from Ascii file
  RandomForest(std::string const &modelFile) {
    auto file = new sg::CSerializableAsciiFile(modelFile.c_str(), 'r');
    // load_serializable(file);

    file->close();
    SG_UNREF(file);
  }

  // Construct from parameters
  RandomForest(int const &n_cats_, int const &n_trees_,
               double const &sample_size_ratio_, int const &num_features_,
               bool const &balance_) {
    n_cats = n_cats_;
    sample_size_ratio = sample_size_ratio_;
    num_features = num_features_;
    balance = balance_;

    for (int i = 0; i < n_cats; ++i) {
      auto rf = sg::wrap(new sg::CRandomForest());

      rf->set_num_bags(n_trees);
      auto m_vote = sg::some<sg::CMajorityVote>();
      rf->set_combination_rule(m_vote);
      rand_forest.push_back(rf);
    }
  }

  ~RandomForest() {}

  void get_params(int const &n) {
    return rand_forest[n]->print_serializable();
  }

  void train(std::vector<BinaryLabels> &labels) {

    for (int i = 0; i < feats->n_cats; ++i) {
      sg::SGVector<double> weights(labels[i].get_num_labels());
      if (balance) {

        make_balanced_weight_vector(labels[i], weights);
      } else {
        // std::cout << "filling weights with 1's" << std::endl;
        sg::SGVector<double>::fill_vector(weights, labels[i].get_num_labels(),
                                          1);
      }

      rand_forest[i]->train(feats->get(i).get());
    }
  }

  // Predict operator
  std::vector<BinaryLabelsPtr> operator()(CategorizedFeatures *feats_) {

    std::vector<BinaryLabelsPtr> predictions;
    for (int i = 0; i < feats_->n_cats; ++i) {
      // initialize

      rand_forest[i]->set_bag_size(sample_size_ratio *
                                   feats_->get(i)->get_num_vectors());

      auto m_vote = sg::some<sg::CMajorityVote>();
      rand_forest[i]->set_combination_rule(m_vote);

      // all features are continuous (non-categorical)
      sg::SGVector<bool> f_type(feats_->get(i)->get_dim_feature_space());
      sg::SGVector<bool>::fill_vector(
          f_type, feats_->get(i)->get_dim_feature_space(), false);

      rand_forest[i]->set_machine_problem_type(sg::EProblemType::PT_BINARY);

      auto pred_labels = rand_forest[i]->apply_binary(feats_->get(i).get());
      predictions.push_back(BinaryLabelsPtr(pred_labels));
    }
    return predictions;
  }
};

// class EnsembleRandomForest : public virtual opt::TFunction<std::vector<FVal>>
// { public:
//   typedef opt::TFunction<std::vector<FVal>> Super;
//   typedef EnsembleRandomForest Self;
//   typedef Self *Pointer;
//   typedef Self const *ConstPointer;
//   typedef std::vector<FVal> Input;
//   typedef std::function<int(Input)> Distributor;

//   std::vector<std::shared_ptr<RandomForest>> models;
//   Distributor fdist;

//   virtual void initialize(int predictLabel,
//                           std::vector<std::string> const &modelFiles,
//                           Distributor const &fdist_) {
//     int n = modelFiles.size();
//     models.resize(n);
//     for (int i = 0; i < n; ++i) {
//       models[i] = std::make_shared<RandomForest>(predictLabel,
//       modelFiles[i]);
//     }
//     fdist = fdist_;
//   }

//   EnsembleRandomForest() {}

//   EnsembleRandomForest(int predictLabel,
//                        std::vector<std::string> const &modelFiles,
//                        Distributor const &fdist) {
//     initialize(predictLabel, modelFiles, fdist);
//   }

//   ~EnsembleRandomForest() override {}

//   double operator()(Input const &x) override {
//     return models[fdist(x)]->operator()(x);
//   }

//   double operator()(double *g, Input const &x) override {
//     perr("Error: no gradient available for random forest...");
//     return operator()(x);
//   }
// };

}; // namespace alg
}; // namespace glia

#endif
