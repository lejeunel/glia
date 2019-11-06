#ifndef _shogun_helpers_hxx_
#define _shogun_helpers_hxx_

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <iostream>
#include <shogun/base/some.h>

namespace sg = shogun;
namespace np = boost::python::numpy;
namespace bp = boost::python;

typedef sg::CMulticlassLabels MulticlassLabels;
template <typename Tx>
using Features = sg::CDenseFeatures<Tx>;

template<typename Tx>
sg::Some<MulticlassLabels> np_to_shogun_labels(np::ndarray const& X)
{

  np::ndarray X_ = X.astype(np::dtype::get_builtin<Tx>());
  Tx* data = reinterpret_cast<Tx*>(X_.get_data());
  auto labels = sg::wrap(new sg::CMulticlassLabels(X_.shape(0)));

  for (int i = 0; i < labels->get_num_labels(); ++i) {
    labels->set_int_label(i, data[i]);
  }
  return labels;
}

template<typename Tx>
sg::Some<Features<Tx>> np_to_shogun_feats(np::ndarray const& X)
{
  Tx* data = reinterpret_cast<Tx*>(X.get_data());
  auto mat = sg::SGMatrix<Tx>(data, X.shape(1), X.shape(0), false);
  auto features = sg::wrap(new sg::CDenseFeatures<Tx>(mat));

  return features;
}

sg::SGVector<double> make_balanced_weight_vector(sg::Some<MulticlassLabels> Y,
                                                 sg::SGVector<double>& weights){

  auto Yvec = Y->get_labels();
  // Yvec.display_vector();
    // auto label_counts = new double[2];
    std::shared_ptr<double[]> label_counts(new double[2]);

    label_counts[0] = 0;
    label_counts[1] = 0;
    for(int i=0; i < Yvec.vlen; ++i){
      if(Yvec[i] == 0) label_counts[0] += 1;
      if(Yvec[i] == 1) label_counts[1] += 1;
                             }

    weights.zero();
    for(int i=0; i < Yvec.vlen; ++i){
      if(Yvec[i] == 0) weights[i] = Yvec.vlen / (2 * label_counts[0]);
      if(Yvec[i] == 1) weights[i] = Yvec.vlen /  (2 * label_counts[1]);
    }


    return weights;

}

#endif
