#ifndef _shogun_helpers_hxx_
#define _shogun_helpers_hxx_

#include <algorithm>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubSamplesFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

namespace sg = shogun;
namespace np = boost::python::numpy;
namespace bp = boost::python;

using namespace shogun;

typedef DenseFeatures<double> Features;
typedef DenseSubSamplesFeatures<double> SubFeatures;
typedef std::shared_ptr<DenseSubSamplesFeatures<double>> SubFeaturesPtr;
typedef std::shared_ptr<DenseFeatures<double>> FeaturesPtr;
typedef std::shared_ptr<MulticlassLabels> MulticlassLabelsPtr;

// Get the median of an unordered set of numbers of arbitrary
// type without modifying the underlying dataset.
template <typename T> T median(T *begin, T *end);

class CategorizedFeatures {
public:
  std::vector<std::vector<int>> indices;
  int n_cats = 3;
  double threshold;
  int n_dims = 0;
  FeaturesPtr feats;
  std::vector<FeaturesPtr> feats_by_cat;

public:
  CategorizedFeatures() {}
  CategorizedFeatures(FeaturesPtr feats_, int const &idx_first,
                      int const &idx_second) {

    // Compute median of features on given idx
    auto mat = feats_->get_feature_matrix();
    mat.display_matrix();
    auto first = mat.get_row_vector(idx_first);
    auto second = mat.get_row_vector(idx_second);
    auto first_vec =
        std::vector<double>(first.vector, first.vector + first.size());
    auto second_vec =
        std::vector<double>(first.vector, first.vector + first.size());
    first_vec.insert(first_vec.end(), second_vec.begin(), second_vec.end());

    n_dims = mat.get_column(0).size();

    threshold = median<double>(&first_vec.front(),
                               &first_vec.front() + first_vec.size());
    std::cout << "median: " << threshold << std::endl;

    // store indices for each category
    indices = std::vector<std::vector<int>>(3);
    for (int i = 0; i < feats_->get_num_vectors(); ++i) {
      auto this_feat = feats_->get_feature_vector(i);
      if (std::max(this_feat.get_element(idx_first),
                   this_feat.get_element(idx_second)) > threshold) {
        indices[0].push_back(i);
      } else if (std::min(this_feat.get_element(idx_first),
                          this_feat.get_element(idx_second)) < threshold) {
        indices[1].push_back(i);
      } else {
        indices[2].push_back(i);
      }
    }

    for(int i= 0; i < indices.size(); ++i){
      std::cout << "got " << indices[i].size() << " samples in cat. " << i << std::endl;
      auto mat_this_cat = SGMatrix<double>(feats_->get_num_features(), indices[i].size());
      for(int j= 0; j < indices[i].size(); ++j){
        SGVector<double> vec = mat.get_column(indices[i][j]);
        vec.display_vector();
        mat_this_cat.set_column(j, vec);
      }
      mat_this_cat.display_matrix();

      auto feats_this_cat = std::make_shared<DenseFeatures<double>>(mat_this_cat);
      feats_by_cat.push_back(feats_this_cat);
    }

    feats = feats_;
  }

  FeaturesPtr get(int const &cat) {
    return feats_by_cat[cat];
  }

  SubFeaturesPtr get_subsamples(int const &cat) {
    auto vec = SGVector<int>(indices[cat].begin(), indices[cat].end());
    return std::make_shared<SubFeatures>(feats, vec);
  }

  MulticlassLabelsPtr filter_labels(MulticlassLabelsPtr labels,
                                    int const &cat) const {

    labels->remove_all_subsets();
    auto idx = SGVector<int>(indices[cat].begin(), indices[cat].end());
    labels->add_subset(idx);
    return labels;
  }

  void display_feats() {
    for (int i = 0; i < indices.size(); ++i) {
      std::cout << "category: " << i << ", n_feats:" << indices[i].size()
                << std::endl;
      if (indices[i].size() > 0)
        get(i)->get_computed_dot_feature_matrix().display_matrix();
    }
  }
};

typedef std::shared_ptr<CategorizedFeatures> CategorizedFeaturesPtr;

template <typename T> T median(T *begin, T *end) {
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
MulticlassLabelsPtr np_to_shogun_labels(np::ndarray const &X) {

  // np::ndarray X_ = X.astype(np::dtype::get_builtin<Tx>());
  // auto X_copy = X_.copy();
  // Tx *data = reinterpret_cast<Tx *>(X_copy.get_data());

  Tx *data = reinterpret_cast<Tx *>(X.get_data());

  auto labels = std::make_shared<MulticlassLabels>(X.shape(0));

  for (int i = 0; i < labels->get_num_labels(); ++i) {
    labels->set_int_label(i, data[i]);
  }
  return labels;
}

template <typename Tx> FeaturesPtr np_to_shogun_feats(np::ndarray const &X) {
  Tx *data = reinterpret_cast<Tx *>(X.get_data());
  auto mat = SGMatrix<Tx>(X.shape(1), X.shape(0));
  std::cout << X.shape(0) << std::endl;
  std::cout << X.shape(1) << std::endl;
  for (int i = 0; i < X.shape(0); ++i) {
        auto vec = SGVector<Tx>(X.shape(1));
        for (int j = 0; j<X.shape(1); ++j){
          vec.set_element(data[i * X.shape(1) + j], j);
    }
    // vec.display_vector();
    mat.set_column(i, vec);
  }
  // mat.display_matrix();
  auto out = std::make_shared<DenseFeatures<double>>(mat);
  return out;
}

inline SGVector<double>
make_balanced_weight_vector(MulticlassLabelsPtr Y,
                            SGVector<double> &weights) {

  auto Yvec = Y->get_labels();
  // Yvec.display_vector();
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

//from https://gist.github.com/marcinwol/b8df949bf8009cf856a3
template <class T>
inline
boost::python::list std_vector_to_list(std::vector<T> vector) {
    typename std::vector<T>::iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
        list.append(*iter);
    }
    return list;
}
template<typename T>
inline
std::vector< T > list_to_std_vector( const bp::object& iterable )
{
    return std::vector< T >( boost::python::stl_input_iterator< T >( iterable ),
                             boost::python::stl_input_iterator< T >( ) );
}

#endif
