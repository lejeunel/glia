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
typedef std::shared_ptr<sg::RandomForest> RandomForestModel;

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
    // auto file = new sg::CSerializableAsciiFile(modelFile.c_str(), 'r');
    // load_serializable(file);

    // file->close();
    // SG_UNREF(file);
  }

  // Construct from parameters
  RandomForest(int const &n_cats_, int const &n_trees_,
               double const &sample_size_ratio_, int const &num_features_,
               bool const &balance_) {

    n_cats = n_cats_;
    n_trees = n_trees_;
    sample_size_ratio = sample_size_ratio_;
    num_features = num_features_;
    balance = balance_;

    for (int i = 0; i < n_cats; ++i) {
      std::shared_ptr<sg::RandomForest> rf(new sg::RandomForest());

      rf->set_num_bags(n_trees);
      auto m_vote = std::make_shared<sg::MajorityVote>();
      rf->set_combination_rule(m_vote);

      rand_forest.push_back(rf);
    }
  }

  ~RandomForest() {}

  void from_serialized(std::vector<std::string> const& params) {
    n_cats = params.size();

    rand_forest.clear();
    for (int i = 0; i < params.size(); ++i) {

      auto deserializer = std::make_shared<sg::io::JsonDeserializer>();
      auto stream = std::make_shared<sg::io::ByteArrayInputStream>(params[i]);
      deserializer->attach(stream);
      auto obj = deserializer->read_object()->as<sg::RandomForest>();
      rand_forest.push_back(obj);

    }
  }

  std::vector<std::string> to_serialized() {
    auto params = std::vector<std::string>(n_cats);
    for (int i = 0; i < n_cats; ++i) {

      auto serializer = std::make_shared<sg::io::JsonSerializer>();
      auto stream = std::make_shared<sg::io::ByteArrayOutputStream>();
      serializer->attach(stream);
      serializer->write(rand_forest[i]);
      params[i] = stream->as_string();
    }
    return params;
  }

  void train(CategorizedFeatures const &X, MulticlassLabelsPtr Y) {

    for (int i = 0; i < X.n_cats; ++i) {
      sg::SGVector<double> weights(X.get(i)->get_num_vectors());
      sg::SGVector<double>::fill_vector(
          weights, X.filter_labels(Y, i)->get_num_labels(), 1);
      // if(balance)
      //   make_balanced_weight_vector(X.filter_labels(Y, i), weights);

      std::cout << "set_labels" << std::endl;
      rand_forest[i]->set_labels(X.filter_labels(Y, i));
      std::cout << "set_weights" << std::endl;
      rand_forest[i]->set_weights(weights);
      // X.filter_labels(Y, i)->get_labels().display_vector();
      std::cout << "num samples: " << X.get(i)->get_num_vectors() << std::endl;

      // rand_forest[i]->set_num_random_features(X.get(i)->get_num_features());
      std::cout << "training cat. " << i << std::endl;

      auto f_type = sg::SGVector<bool>(X.n_dims);
      std::cout << "set_feature_types" << std::endl;
      f_type.display_vector();
      rand_forest[i]->set_feature_types(f_type);
      std::cout << "ok" << std::endl;
      // std::cout << "is_label_valid: " <<
      // rand_forest[i]->is_label_valid(X.filter_labels(Y, i).get()) <<
      // std::endl;
      bool ok;
      ok = rand_forest[i]->train(X.get(i));
      std::cout << "ok: " << ok << std::endl;
    }
  }

  // Predict operator
  std::vector<MulticlassLabelsPtr> operator()(CategorizedFeatures *feats_) {

    std::vector<MulticlassLabelsPtr> predictions;
    for (int i = 0; i < feats_->n_cats; ++i) {
      // initialize

      rand_forest[i]->set_bag_size(sample_size_ratio *
                                   feats_->get(i)->get_num_vectors());

      auto m_vote = std::shared_ptr<sg::MajorityVote>();
      rand_forest[i]->set_combination_rule(m_vote);

      // all features are continuous (non-categorical)
      sg::SGVector<bool> f_type(feats_->get(i)->get_dim_feature_space());
      sg::SGVector<bool>::fill_vector(
          f_type, feats_->get(i)->get_dim_feature_space(), false);

      rand_forest[i]->set_machine_problem_type(sg::EProblemType::PT_BINARY);

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
