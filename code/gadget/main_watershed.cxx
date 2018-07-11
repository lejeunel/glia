#include "util/image_alg.hxx"
#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
#include "glia_base.hxx"
#include <boost/python/numpy.hpp>
#include <itkImageConstIterator.h>
#include "itkImageLinearIteratorWithIndex.h"
#include "np_helpers.hxx"

using namespace glia;
namespace np = boost::python::numpy;
namespace bp = boost::python;

np::ndarray watershed_operation (std::string const& outputImageFile,
                                 std::string const& inputImageFile,
                                 double level,
                                 bool relabel,
                                 bool write16,
                                 bool compress)
{

  auto inputImage = readImage<RealImage<DIMENSION>>(inputImageFile);
  auto outputImage = watershed<LabelImage<DIMENSION>>(inputImage, level);
  using LabelImageType = LabelImage<DIMENSION>;
  if (relabel) { relabelImage(outputImage, 0); }
  //if (write16) {
  //  castWriteImage<UInt16Image<DIMENSION>>
  //      (outputImageFile, outputImage, compress);
  //}
  //else { writeImage(outputImageFile, outputImage, compress); }

    LabelImage<DIMENSION>::RegionType region = outputImage->GetLargestPossibleRegion();
    auto size = region.GetSize();

    return itk_to_np<LabelImageType, Label>(outputImage);
}


//int main (int argc, char* argv[])
//{
//  std::string inputImageFile, outputImageFile;
//  double level;
//  bool relabel = false, write16 = false, compress = false;
//  bpo::options_description opts("Usage");
//  opts.add_options()
//      ("help", "Print usage info")
//      ("inputImage,i", bpo::value<std::string>(&inputImageFile)->required(),
//       "Input image file name")
//      ("level,l", bpo::value<double>(&level)->required(),
//       "Watershed water level")
//      ("relabel,r", bpo::value<bool>(&relabel),
//       "Whether to relabel output image [default: false]")
//      ("write16,u", bpo::value<bool>(&write16),
//       "Whether to write to uint16 image [default: false]")
//      ("compress,z", bpo::value<bool>(&compress),
//       "Whether to compress output image file(s) [default: false]")
//      ("outputImage,o",
//       bpo::value<std::string>(&outputImageFile)->required(),
//       "Output image file name");
//  return
//      parse(argc, argv, opts) &&
//      operation(outputImageFile, inputImageFile, level, relabel, write16,
//                compress)? EXIT_SUCCESS: EXIT_FAILURE;
//}
