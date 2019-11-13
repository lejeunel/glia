#ifndef _glia_alg_rf_hxx_
#define _glia_alg_rf_hxx_

#include "shogun_helpers.hxx"
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/io/stream/ByteArrayInputStream.h>
#include <shogun/io/stream/ByteArrayOutputStream.h>
#include <shogun/lib/any.h>
#include <shogun/machine/RandomForest.h>

using namespace shogun;

namespace glia {
namespace alg {

class MyRandomForest {
public:
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

  void from_serialized(std::vector<std::string> const &params) {
    n_cats = params.size();

    rand_forest.clear();
    auto deserializer = std::make_shared<io::JsonDeserializer>();

    for (int i = 0; i < params.size(); ++i) {
      auto stream = std::make_shared<io::ByteArrayInputStream>(params[i]);
      deserializer->attach(stream);
      auto obj = deserializer->read_object()->as<RandomForest>();
      rand_forest.push_back(obj);
    }
  }

  std::vector<std::string> to_serialized() {
    auto params = std::vector<std::string>(n_cats);
    auto serializer = std::make_shared<io::JsonSerializer>();
    for (int i = 0; i < n_cats; ++i) {
      auto stream = std::make_shared<io::ByteArrayOutputStream>();

      serializer->attach(stream);
      serializer->write(rand_forest[i]);
      params[i] = stream->as_string();
    }
    return params;
  }

  void train(CategorizedFeaturesPtr X, MulticlassLabelsPtr Y) {

    for (int i = 0; i < X->n_cats; ++i) {
      auto feats_this_cat = X->get(i);
      auto labels_this_cat = X->filter_labels(Y, i);
      std::cout << " cat. " << i
                << " n vectors: " << feats_this_cat->get_num_vectors()
                << " n unique labels: "
                << labels_this_cat->get_unique_labels().size() << std::endl;
      if (labels_this_cat->get_unique_labels().size() > 1) {
        SGVector<double> weights(feats_this_cat->get_num_vectors());
        SGVector<double>::fill_vector(
            weights, X->filter_labels(Y, i)->get_num_labels(), 1);
        if (balance)
          make_balanced_weight_vector(labels_this_cat, weights);

        rand_forest[i]->set_labels(labels_this_cat);
        rand_forest[i]->set_weights(weights);

        auto f_type = SGVector<bool>(X->n_dims);
        // X->filter_labels(Y, i)->get_labels().display_vector();
        rand_forest[i]->set_feature_types(f_type);
        rand_forest[i]->train(feats_this_cat);
      }
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

class EnsembleRandomForest {

public:
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
  std::vector<std::shared_ptr<MyRandomForest>> models;

  EnsembleRandomForest(int const &n_cats_,
                       int const &n_trees_,
                       double const &sample_size_ratio_,
                       int const &num_features_,
                       bool const &balance_) {

    n_cats = n_cats_;
    n_trees = n_trees_;
    sample_size_ratio = sample_size_ratio_;
    num_features = num_features_;
    balance = balance_;
  }

  ~EnsembleRandomForest() {}

  // This will create a new RF in the ensemble
  void train(CategorizedFeaturesPtr X, MulticlassLabelsPtr Y) {

    // create new model
    std::cout << "training (n_trees: " << n_trees << ")" << std::endl;
    auto m = std::make_shared<MyRandomForest>(
        n_cats, n_trees, sample_size_ratio, num_features, balance);

    m->train(X, Y);

    models.push_back(m);
  }

  // first dim: stage, second dim: category
  std::vector<std::vector<std::string>> to_serialized() {
    std::vector<std::vector<std::string>> params;
    for (int i = 0; i < models.size(); ++i) {
      auto params_ = models[i]->to_serialized();
      params.push_back(params_);
    }
    return params;
  }

  // first dim: stage, second dim: category
  virtual void
  from_serialized(std::vector<std::vector<std::string>> const &params) {
    models.clear();
    models = std::vector<std::shared_ptr<MyRandomForest>>(params.size());
    for (int i = 0; i < params.size(); ++i) {
      models[i]->from_serialized(params[i]);
    }
  }

  // double operator()(Input const &x) override {
  //   return models[fdist(x)]->operator()(x);
  // }

  // double operator()(double *g, Input const &x) override {
  //   perr("Error: no gradient available for random forest...");
  //   return operator()(x);
  // }
};

}; // namespace alg
}; // namespace glia

#endif
