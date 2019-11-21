#ifndef _np_helpers_hxx_
#define _np_helpers_hxx_

#include "boost/shared_ptr.hpp"
#include "glia_base.hxx"
#include "hmt/bc_feat.hxx"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImportImageFilter.h"
#include "itkPasteImageFilter.h"
#include "itkRGBToLuminanceImageFilter.h"
#include "type/tuple.hxx"
#include "util/image_alg.hxx"
#include "util/image_io.hxx"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <itkImageConstIterator.h>
#include <memory>
#include <vector>

using namespace glia;
namespace np = boost::python::numpy;
namespace bp = boost::python;

namespace nph {

inline void print(np::ndarray arr) {
  std::cout << bp::extract<char const *>(bp::str(arr)) << std::endl;
}

// Converts an ITK image to numpy array
template <typename ImageType, typename PixelType>
np::ndarray itk_to_np(ImageType *inputImage) {

  auto region = inputImage->GetLargestPossibleRegion();
  auto size = region.GetSize();

  using IteratorType = itk::ImageLinearIteratorWithIndex<ImageType>;
  using ConstIteratorType = itk::ImageLinearConstIteratorWithIndex<ImageType>;

  np::ndarray out_np = np::empty(bp::make_tuple(size[0] * size[1]),
                                 np::dtype::get_builtin<PixelType>());
  ConstIteratorType it(inputImage, inputImage->GetRequestedRegion());
  it.SetDirection(0);
  unsigned int i = 0;
  for (it.GoToBegin(); !it.IsAtEnd(); it.NextLine()) {
    it.GoToBeginOfLine();
    while (!it.IsAtEndOfLine()) {
      out_np[i] = it.Get();
      ++it;
      ++i;
    }
  }
  return out_np.reshape(bp::make_tuple(size[1], size[0]));
}


// Converts a numpy array to ITK image
inline LabelImage<DIMENSION>::Pointer
np_to_itk_label(const np::ndarray &inputArray) {

  auto X = inputArray.astype(np::dtype::get_builtin<Label>());

  using LabelImageType = LabelImage<DIMENSION>;
  using ShapeType = const long int *;
  typedef itk::ImportImageFilter<Label, DIMENSION> ImportFilterType;
  typedef itk::PasteImageFilter<LabelImageType> PasteFilterType;
  typename LabelImageType::Pointer image = LabelImageType::New();
  typename ImportFilterType::SizeType size;
  typename ImportFilterType::RegionType region;
  typename ImportFilterType::IndexType start;

  ShapeType shape = reinterpret_cast<ShapeType>(X.get_shape());
  size[1] = shape[0];
  size[0] = shape[1];
  start.Fill(0);

  region.SetIndex(start);
  region.SetSize(size);

  image->SetRegions(region);
  image->Allocate();

  Label *data = reinterpret_cast<Label *>(X.get_data());
  int i = 0;
  for (int y = 0; y < size[1]; ++y) {
    for (int x = 0; x < size[0]; ++x) {
    const LabelImageType::IndexType pixelIndex = {{x, y}};
    auto val = data[i];
    image->SetPixel(pixelIndex, val);
    // std::cout << "(" << x << "," << y << "): " << val << std::endl;
    i++;
    }
  }


  return image;
}


inline RealImage<2>::Pointer np_to_itk_real(const np::ndarray &inputArray) {

  auto X = inputArray.astype(np::dtype::get_builtin<Real>());
  using RealImageType = RealImage<2>;
  using ShapeType = const long int *;
  typedef itk::ImportImageFilter<Real, 2> ImportFilterType;
  typedef itk::PasteImageFilter<RealImageType> PasteFilterType;
  typename RealImageType::Pointer image = RealImageType::New();
  typename ImportFilterType::SizeType size;
  typename ImportFilterType::RegionType region;
  typename ImportFilterType::IndexType start;
  ImportFilterType::Pointer importFilter = ImportFilterType::New();

  ShapeType shape = reinterpret_cast<ShapeType>(X.get_shape());
  size[1] = shape[0];
  size[0] = shape[1];
  start.Fill(0);

  region.SetIndex(start);
  region.SetSize(size);

  image->SetRegions(region);
  image->Allocate();

  Real *data = reinterpret_cast<Real *>(X.get_data());
  int i = 0;
  for (int y = 0; y < size[1]; ++y) {
    for (int x = 0; x < size[0]; ++x) {
    const RealImageType::IndexType pixelIndex = {{x, y}};
    auto val = data[i];
    image->SetPixel(pixelIndex, val);
    i++;
    }
  }

  return image;
}

template <typename T>
std::vector<T> list_to_vector(const bp::object &iterable) {
  return std::vector<T>(bp::stl_input_iterator<T>(iterable),
                        bp::stl_input_iterator<T>());
}

template <typename T> np::ndarray vector_to_np(std::vector<T> const &vec) {

  // This should work but doesn't. Data gets corrupted somehow
  // np::ndarray out_np = np::from_data(vec.data(),
  //                                   np::dtype::get_builtin<T>(),
  //                                   bp::make_tuple(vec.size()),
  //                                   bp::make_tuple(sizeof(T)),
  //                                   bp::object());

  np::ndarray out_np =
      np::empty(bp::make_tuple(vec.size()), np::dtype::get_builtin<T>());
  std::copy(vec.begin(), vec.end(), reinterpret_cast<T *>(out_np.get_data()));

  return out_np;
}

template <typename T>
np::ndarray vector_2d_to_np(std::vector<std::vector<T>> const &vec) {

  // flatten vector
  std::vector<T> flat_vec;
  for (unsigned int i = 0; i < vec.size(); ++i) {
    for (unsigned int j = 0; j < vec[i].size(); ++j) {
      flat_vec.push_back(vec[i][j]);
    }
  }

  bp::tuple shape = bp::make_tuple(vec.size(), vec[0].size());
  return vector_to_np<T>(flat_vec).reshape(shape);
}

template <typename T> std::vector<T> np_to_vector(np::ndarray const &arr) {

  // This should work but doesn't. Data gets corrupted somehow
  // np::ndarray out_np = np::from_data(vec.data(),
  //                                   np::dtype::get_builtin<T>(),
  //                                   bp::make_tuple(vec.size()),
  //                                   bp::make_tuple(sizeof(T)),
  //                                   bp::object());

  using ShapeType = const long int *;
  ShapeType shape = reinterpret_cast<ShapeType>(arr.get_shape());

  T *arr_data = reinterpret_cast<T *>(arr.get_data());
  std::vector<T> vec_out(shape[0]);

  for (unsigned int i = 0; i < shape[0]; ++i) {
    vec_out.push_back(arr_data[i]);
  }

  return vec_out;
}

inline bool is_empty(np::ndarray const &arr) {
  using ShapeType = const long int *;
  ShapeType shape = reinterpret_cast<ShapeType>(arr.get_shape());
  if (shape[0] == 0)
    return true;

  return false;
}

template <typename T>
bp::list vector_triple_to_np(std::vector<glia::TTriple<T>> const &vec) {

  // get all components of triple
  std::vector<T> tx0;
  std::vector<T> tx1;
  std::vector<T> tx2;

  for (unsigned int i = 0; i < vec.size(); ++i) {
    // std::cout << vec[i].x0 << std::endl;
    tx0.push_back(vec[i].x0);
  }

  for (unsigned int i = 0; i < vec.size(); ++i) {
    // std::cout << vec[i].x1 << std::endl;
    tx1.push_back(vec[i].x1);
  }

  for (unsigned int i = 0; i < vec.size(); ++i) {
    // std::cout << vec[i].x2 << std::endl;
    tx2.push_back(vec[i].x2);
  }

  np::ndarray out0 = vector_to_np<T>(tx0);
  np::ndarray out1 = vector_to_np<T>(tx1);
  np::ndarray out2 = vector_to_np<T>(tx2);

  bp::list out;
  out.append(out0);
  out.append(out1);
  out.append(out2);

  return out;
}

template <typename T>
std::vector<glia::TTriple<T>> np_to_vector_triple(bp::list const &list) {

  // get all components of list
  std::vector<glia::TTriple<T>> tpl_out;
  T x0;
  T x1;
  T x2;

  for (unsigned int i = 0; i < bp::len(list[0]); ++i) {
    x0 = bp::extract<T>(list[0][i]);
    x1 = bp::extract<T>(list[1][i]);
    x2 = bp::extract<T>(list[2][i]);
    auto t_ = glia::TTriple<T>(x0, x1, x2);
    tpl_out.push_back(t_);
  }

  return tpl_out;
}

// convert python list of arrays to vector of itk real images
// This builds vector of ImageHistPair
inline std::vector<hmt::ImageHistPair<RealImage<DIMENSION>::Pointer>>
lists_to_image_hist_pair(bp::list const &im_list,
                         bp::list const &histogramBins,
                         bp::list const &histogramLowerValues,
                         bp::list const &histogramHigherValues) {

  int n_imgs = bp::len(im_list);
  std::vector<hmt::ImageHistPair<RealImage<DIMENSION>::Pointer>> out;

  out.reserve(n_imgs);

  // typename RealImageType::Pointer img = RealImageType::New();
  for (unsigned int i = 0; i < bp::len(im_list); ++i) {
    std::pair<double, double> hist_range;
    int n_bins;
    np::ndarray arr = bp::extract<np::ndarray>(im_list[i]);
    hist_range.first = bp::extract<double>(histogramLowerValues[i]);
    hist_range.second = bp::extract<double>(histogramHigherValues[i]);
    n_bins = bp::extract<int>(histogramBins[i]);
    out.emplace_back(np_to_itk_real(arr), n_bins, hist_range);
  }

  return out;
}

template <typename T, typename TPointer>
void write_image(TPointer img, std::string const &fname) {
  using WriterType = itk::ImageFileWriter<T>;
  TPointer writer = WriterType::New();
  writer->SetFileName(fname);
  writer->SetInput(img);
  writer->Update();
}

}; // namespace nph
#endif
