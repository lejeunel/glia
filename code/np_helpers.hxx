#include "util/image_alg.hxx"
#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
#include "glia_base.hxx"
#include <boost/python/numpy.hpp>
#include <itkImageConstIterator.h>
#include "itkImageLinearIteratorWithIndex.h"

using namespace glia;
namespace np = boost::python::numpy;
namespace bp = boost::python;

// Converts an ITK image to numpy array
template <typename ImageType, typename PixelType>
np::ndarray
itk_to_np (ImageType * inputImage) {


    PixelType * imgArray = inputImage->GetBufferPointer();
    auto region = inputImage->GetLargestPossibleRegion();
    auto size = region.GetSize();


    using IteratorType = itk::ImageLinearIteratorWithIndex<ImageType>;
    using ConstIteratorType = itk::ImageLinearConstIteratorWithIndex<ImageType>;

    np::ndarray out_np = np::empty(bp::make_tuple(size[0]*size[1]),
                                   np::dtype::get_builtin<PixelType>());
    ConstIteratorType it(inputImage, inputImage->GetRequestedRegion());
    it.SetDirection(0);
    unsigned int i = 0;
    for ( it.GoToBegin(); !it.IsAtEnd(); it.NextLine())
        {
        it.GoToBeginOfLine();
        while ( ! it.IsAtEndOfLine() )
            {
            out_np[i] = it.Get();
            ++it;
            ++i;
            }
      }
    return out_np.reshape(bp::make_tuple(size[1], size[0]));
}
