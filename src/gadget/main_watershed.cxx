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

//       "Input image file name
//       "Watershed water level
//       "Whether to relabel output image
np::ndarray watershed_operation (np::ndarray const& inputImage_np, //grayscale!
                                 double level,
                                 bool relabel)
{

  auto inputImage = np_to_itk_real(inputImage_np);
  //std::cout << inputImage << std::endl;
  //auto inputImage = readImage<RealImage<DIMENSION>>(inputImageFile);
  auto outputImage = watershed<LabelImage<DIMENSION>>(inputImage, level);
  using LabelImageType = LabelImage<DIMENSION>;
  if (relabel) {
    relabelImage(outputImage, 0); }

    return itk_to_np<LabelImageType, Label>(outputImage);
}
