#include "glia_image.hxx"
#include "itkImageLinearIteratorWithIndex.h"
#include "np_helpers.hxx"
#include "shogun_helpers.hxx"
#include "util/image_alg.hxx"
#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <itkImageConstIterator.h>

using namespace glia;
namespace np = boost::python::numpy;
namespace bp = boost::python;

void test_conversion_shogun_labels(np::ndarray const &A) {

  // np::ndarray B = A.astype(np::dtype::get_builtin<int>());
  auto out = np_to_shogun_labels<float64_t>(A);
  std::cout << "Num samples: " << out->get_num_labels() << std::endl;
  out->get_labels().display_vector();
}

void test_conversion_shogun_feats(np::ndarray const &A) {

  auto out = np_to_shogun_feats<double>(A);
  std::cout << "Num features: " << out->get_num_features() << std::endl;
  std::cout << "Num samples: " << out->get_num_vectors() << std::endl;
  for (int i = 0; i < out->get_num_vectors(); i++) {
    out->get_feature_vector(i).display_vector();
  }
}

void test_conversion_itk(np::ndarray const &inputImage_np,
                         std::string const &inputImageStr,
                         std::string const &outputImageStr_itk_rgb,
                         std::string const &outputImageStr_itk_real,
                         std::string const &outputImageStr_np_rgb,
                         std::string const &outputImageStr_np_real) {
  std::cout << "reading itk rgb from str: " << inputImageStr << std::endl;
  auto inputImage_from_str_rgb = readImage<RgbImage>(inputImageStr);
  std::cout << "writing itk rgb from str" << std::endl;
  castWriteImage<RgbImage>(outputImageStr_itk_rgb, inputImage_from_str_rgb,
                           false);

  std::cout << "reading itk real from str" << std::endl;
  auto inputImage_from_str_real =
      readImage<RealImage<DIMENSION>>(inputImageStr);
  std::cout << "writing itk real from str" << std::endl;
  castWriteImage<RgbImage>(outputImageStr_itk_real, inputImage_from_str_real,
                           true);

  std::cout << "np array to itk rgb" << std::endl;
  auto inputImage_np_rgb = np_to_itk_rgb(inputImage_np);
  std::cout << "writing itk rgb (np)" << std::endl;
  castWriteImage<RgbImage>(outputImageStr_np_rgb, inputImage_np_rgb, false);

  std::cout << "np array to itk real" << std::endl;
  auto inputImage_np_real = np_to_itk_real(inputImage_np);
  std::cout << "writing itk real (np)" << std::endl;
  castWriteImage<RgbImage>(outputImageStr_np_real, inputImage_np_real, true);
}
