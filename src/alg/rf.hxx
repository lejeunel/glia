#ifndef _glia_alg_rf_hxx_
#define _glia_alg_rf_hxx_

#include "shogun_helpers.hxx"
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/io/stream/ByteArrayOutputStream.h>
#include <shogun/io/stream/ByteArrayInputStream.h>
#include <shogun/lib/any.h>
#include <shogun/machine/RandomForest.h>


using namespace shogun;

namespace glia {
namespace alg {

class MyRandomForest {
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

  std::vector<std::shared_ptr<RandomForest>> rand_forest;

  MyRandomForest() {}

  // Construct from Ascii file
  MyRandomForest(std::string const &modelFile) {
    // auto file = new sg::CSerializableAsciiFile(modelFile.c_str(), 'r');
    // load_serializable(file);

    // file->close();
    // SG_UNREF(file);
  }

  // Construct from parameters
  MyRandomForest(int const &n_cats_, int const &n_trees_,
               double const &sample_size_ratio_, int const &num_features_,
               bool const &balance_) {

    n_cats = n_cats_;
    n_trees = n_trees_;
    sample_size_ratio = sample_size_ratio_;
    num_features = num_features_;
    balance = balance_;

    for (int i = 0; i < n_cats; ++i) {
      auto rf = std::make_shared<RandomForest>();
      rf->set_num_bags(n_trees);
      auto m_vote = std::make_shared<MajorityVote>();
      rf->set_combination_rule(m_vote);

      rand_forest.push_back(rf);
    }
  }

  ~MyRandomForest() {}

  void from_serialized(std::vector<std::string> const& params) {
    n_cats = params.size();

    rand_forest.clear();
    auto deserializer = std::make_shared<io::JsonDeserializer>();

    for (int i = 0; i < params.size(); ++i) {
      auto stream = std::make_shared<io::ByteArrayInputStream>(params[i]);
      deserializer->attach(stream);
      auto obj = deserializer->read_object()->as<RandomForest>();
      std::cout << obj->get_name() << "\n";
      rand_forest.push_back(obj);

    }
  }

  std::vector<std::string> to_serialized() {
    auto params = std::vector<std::string>(n_cats);
    for (int i = 0; i < n_cats; ++i) {

      auto serializer = std::make_shared<io::JsonSerializer>();
      auto stream = std::make_shared<io::ByteArrayOutputStream>();
      serializer->attach(stream);
      serializer->write(rand_forest[i]);
      params[i] = stream->as_string();
    }
    return params;
  }

  void train(CategorizedFeaturesPtr X, MulticlassLabelsPtr Y) {

    for (int i = 0; i < X->n_cats; ++i) {
      SGVector<double> weights(X->get(i)->get_num_vectors());
      SGVector<double>::fill_vector(
          weights, X->filter_labels(Y, i)->get_num_labels(), 1);
      // if(balance)
      //   make_balanced_weight_vector(X.filter_labels(Y, i), weights);

      rand_forest[i]->set_labels(X->filter_labels(Y, i));
      rand_forest[i]->set_weights(weights);

      auto f_type = SGVector<bool>(X->n_dims);
      f_type.display_vector();
      rand_forest[i]->set_feature_types(f_type);
      rand_forest[i]->train(X->get(i));
    }
  }

  // Predict operator
  std::vector<MulticlassLabelsPtr> operator()(CategorizedFeaturesPtr feats_) {

    std::vector<MulticlassLabelsPtr> predictions;
    for (int i = 0; i < feats_->n_cats; ++i) {
      // initialize

      rand_forest[i]->set_bag_size(sample_size_ratio *
                                   feats_->get(i)->get_num_vectors());

      auto m_vote = std::shared_ptr<MajorityVote>();
      rand_forest[i]->set_combination_rule(m_vote);

      // all features are continuous (non-categorical)
      SGVector<bool> f_type(feats_->get(i)->get_dim_feature_space());
      SGVector<bool>::fill_vector(
          f_type, feats_->get(i)->get_dim_feature_space(), false);

      rand_forest[i]->set_machine_problem_type(EProblemType::PT_BINARY);

      auto pred_labels = rand_forest[i]->apply_multiclass(feats_->get(i));
      predictions.push_back(MulticlassLabelsPtr(pred_labels));
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
