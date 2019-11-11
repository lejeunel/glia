#ifndef _shogun_helpers_hxx_
#define _shogun_helpers_hxx_

#include <algorithm>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubSamplesFeatures.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

namespace sg = shogun;
namespace np = boost::python::numpy;
namespace bp = boost::python;

typedef sg::CBinaryLabels BinaryLabels;
typedef sg::CDenseFeatures<double> Features;
typedef sg::CDenseSubSamplesFeatures<double> SubFeatures;
typedef boost::shared_ptr<sg::CDenseSubSamplesFeatures<double>> SubFeaturesPtr;
typedef boost::shared_ptr<Features> FeaturesPtr;
typedef boost::shared_ptr<BinaryLabels> BinaryLabelsPtr;

// Get the median of an unordered set of numbers of arbitrary
// type without modifying the underlying dataset.
template <typename T> T median(T *begin, T *end);

class CategorizedFeatures {
public:
  std::vector<std::vector<int>> indices;
  int n_cats = 3;
  double threshold;

private:
  std::vector<SubFeaturesPtr> feats;

public:
  CategorizedFeatures(FeaturesPtr feats_, 
                      int const &idx_first,
                      int const &idx_second) {

    // Compute median of features on given idx
    auto mat = feats_->get_feature_matrix();
    mat.display_matrix();
    auto first = mat.get_column(idx_first);
    auto second = mat.get_column(idx_second);
    auto first_vec = std::vector<double>(first.vector, first.vector + first.size());
    auto second_vec = std::vector<double>(first.vector, first.vector + first.size());
    first_vec.insert(first_vec.end(), second_vec.begin(), second_vec.end());

    threshold = median<double>(&first_vec.front(),
                               &first_vec.front() + first_vec.size());
    std::cout << "median: " << threshold << std::endl;

    // store indices for each category
    indices = std::vector<std::vector<int>>(3);
    for (int i = 0; i < feats_->get_num_vectors(); ++i) {
      auto this_feat = feats_->get_feature_vector(i);
      if (std::max(this_feat.get_element(idx_first),
                   this_feat.get_element(idx_second)) > threshold){
        indices[0].push_back(i);
        // std::cout << "pushed vec " << i << " to cat. 0" << std::endl;
      }
      else if (std::min(this_feat.get_element(idx_first),
                        this_feat.get_element(idx_second)) < threshold){
        indices[1].push_back(i);
        // std::cout << "pushed vec " << i << " to cat. 1" << std::endl;
      }
      else{
        indices[2].push_back(i);
        // std::cout << "pushed vec " << i << " to cat. 2" << std::endl;
      }
    }
    std::cout << "store ok" << std::endl;

    // populate
    for (int i = 0; i < indices.size(); ++i) {
      std::cout << "populate category: " << i << " with " << indices[i].size() << " vectors" << std::endl;
      // FeaturesPtr out(new Features(mat));

      SubFeaturesPtr f(new SubFeatures(
          feats_.get(),
          sg::SGVector<int>(&indices[i].front(), indices[i].size())));
      feats.push_back(f);
    }
    std::cout << "populate ok" << std::endl;
  }

  SubFeaturesPtr get(int const &cat) { return feats[cat]; }

  void display_feats() {
    for (int i = 0; i < indices.size(); ++i) {
      std::cout << "category: " << i << ", n_feats:" << indices[i].size() << std::endl;
      if(indices[i].size() > 0)
        get(i)->get_computed_dot_feature_matrix().display_matrix();
    }
  }
};

template <typename T> T
median(T *begin, T *end) {
  std::vector<T> data(begin, end);
  std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
  return data[data.size() / 2];
}

template <typename T> std::vector<T> get_sorted_uniques(T *v, int const &len) {

  std::vector<T> uniques(v, v + len);
  std::sort(uniques.begin(), uniques.end());
  auto last = std::unique(uniques.begin(), uniques.end());
  uniques.erase(last, uniques.end());
  return uniques;
}

template <typename Tx>
BinaryLabelsPtr np_to_shogun_labels(np::ndarray const &X) {

  np::ndarray X_ = X.astype(np::dtype::get_builtin<Tx>());
  Tx *data = reinterpret_cast<Tx *>(X_.get_data());
  // auto labels = new sg::CBinaryLabels(X_.shape(0));
  BinaryLabelsPtr labels(new BinaryLabels(X_.shape(0)));

  for (int i = 0; i < labels->get_num_labels(); ++i) {
    labels->set_int_label(i, data[i]);
  }
  return labels;
}

template <typename Tx> FeaturesPtr np_to_shogun_feats(np::ndarray const &X) {
  Tx *data = reinterpret_cast<Tx *>(X.get_data());
  auto mat = sg::SGMatrix<Tx>(data, X.shape(1), X.shape(0), false);
  // auto out = sg::wrap(new sg::CDenseFeatures<double>(mat));
  FeaturesPtr out(new Features(mat));
  return out;
}

inline sg::SGVector<double>
make_balanced_weight_vector(BinaryLabels Y, sg::SGVector<double> &weights) {

  auto Yvec = Y.get_labels();
  // Yvec.display_vector();
  // auto label_counts = new double[2];
  std::shared_ptr<double[]> label_counts(new double[2]);

  label_counts[0] = 0;
  label_counts[1] = 0;
  for (int i = 0; i < Yvec.vlen; ++i) {
    if (Yvec[i] == 0)
      label_counts[0] += 1;
    if (Yvec[i] == 1)
      label_counts[1] += 1;
  }

  weights.zero();
  for (int i = 0; i < Yvec.vlen; ++i) {
    if (Yvec[i] == 0)
      weights[i] = Yvec.vlen / (2 * label_counts[0]);
    if (Yvec[i] == 1)
      weights[i] = Yvec.vlen / (2 * label_counts[1]);
  }

  return weights;
}

#endif
