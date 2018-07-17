#include "util/image_alg.hxx"
#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
#include "glia_image.hxx"
#include <boost/python/numpy.hpp>
#include <boost/python.hpp>
#include <itkImageConstIterator.h>
#include "itkImageLinearIteratorWithIndex.h"
#include "np_helpers.hxx"


using namespace glia;
namespace np = boost::python::numpy;
namespace bp = boost::python;

void test_conversion (np::ndarray const& inputImage_np,
                      std::string const& inputImageStr,
                      std::string const& outputImageStr_itk_rgb,
                      std::string const& outputImageStr_itk_real,
                      std::string const& outputImageStr_np_rgb,
                      std::string const& outputImageStr_np_real)
{
  std::cout << "reading itk rgb from str" << std::endl;
  auto inputImage_from_str_rgb = readImage<RgbImage>(inputImageStr);
  std::cout << "writing itk rgb from str" << std::endl;
  castWriteImage<RgbImage>
    (outputImageStr_itk_rgb, inputImage_from_str_rgb, false);

  std::cout << "reading itk real from str" << std::endl;
  auto inputImage_from_str_real = readImage<RealImage<DIMENSION>>(inputImageStr);
  std::cout << "writing itk real from str" << std::endl;
  castWriteImage<RgbImage>
    (outputImageStr_itk_real, inputImage_from_str_real, true);

  std::cout << "np array to itk rgb" << std::endl;
  auto inputImage_np_rgb = np_to_itk_rgb(inputImage_np);
  std::cout << "writing itk rgb (np)" << std::endl;
  castWriteImage<RgbImage>
    (outputImageStr_np_rgb, inputImage_np_rgb, false);


  std::cout << "np array to itk real" << std::endl;
  auto inputImage_np_real = np_to_itk_real(inputImage_np);
  std::cout << "writing itk real (np)" << std::endl;
  castWriteImage<RgbImage>
    (outputImageStr_np_real, inputImage_np_real, true);

}
