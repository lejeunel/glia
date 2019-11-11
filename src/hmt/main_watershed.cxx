#include "util/image_alg.hxx"
#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
#include "glia_image.hxx"
#include <itkImageConstIterator.h>
#include "itkImageLinearIteratorWithIndex.h"
#include "pyglia.hxx"
#include "np_helpers.hxx"


using namespace glia;
namespace np = boost::python::numpy;
namespace bp = boost::python;

//       "Input image file name
//       "Watershed water level
//       "Whether to relabel output image
np::ndarray MyHmt::watershed_operation (np::ndarray const& image, //grayscale!
                                        double level,
                                        bool relabel)
{

  auto image_itk = nph::np_to_itk_real(image);
  auto outputImage = watershed<LabelImage<DIMENSION>>(image_itk, level);
  using LabelImageType = LabelImage<DIMENSION>;
  if (relabel) {
    relabelImage(outputImage, 0); }

  return nph::itk_to_np<LabelImageType, Label>(outputImage);
}
