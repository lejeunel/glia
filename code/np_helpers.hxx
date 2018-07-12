#ifndef _np_helpers_hxx_
#define _np_helpers_hxx_

#include "util/image_alg.hxx"
#include "util/image_io.hxx"
#include "util/text_cmd.hxx"
#include "glia_base.hxx"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "boost/shared_ptr.hpp"
#include "boost/python/stl_iterator.hpp"
#include <vector>
#include <memory>
#include <itkImageConstIterator.h>
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImportImageFilter.h"
#include "itkPasteImageFilter.h"

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

// Converts a numpy array to ITK image
LabelImage<DIMENSION>::Pointer
np_to_itk_label(const np::ndarray & inputArray) {


    using LabelImageType = LabelImage<DIMENSION>;
    using ShapeType = const long int *;
    typedef itk::ImportImageFilter<Label, DIMENSION > ImportFilterType;
    typedef itk::PasteImageFilter< LabelImageType > PasteFilterType;
    typename LabelImageType::Pointer image = LabelImageType::New();
    typename ImportFilterType::SizeType size;
    typename ImportFilterType::RegionType region;
    typename ImportFilterType::IndexType start;
    ImportFilterType::Pointer importFilter = ImportFilterType::New();

    ShapeType shape = reinterpret_cast<ShapeType>(inputArray.get_shape());
    size[1] = shape[0];
    size[0] = shape[1];
    //std::cout << "shape 0,1: " << shape[0] << "," << shape[1] << std::endl;
    //std::cout << "size 0,1: " << size[0] << "," << size[1] << std::endl;
    start.Fill(0);

    region.SetIndex(start);
    region.SetSize(size);

    image->SetRegions(region);
    const itk::SpacePrecisionType origin[ 2 ] = { 0.0, 0.0};
    importFilter->SetOrigin( origin );
    const itk::SpacePrecisionType  spacing[ 2 ] =  { 1.0, 1.0};
    importFilter->SetSpacing( spacing );
    image->Allocate();
    //std::cout << "image at np_to_itk_label" << std::endl;
    //std::cout << image << std::endl;
    //std::cout << "before cast" << std::endl
    //        << bp::extract<char const *>(bp::str(inputArray)) << std::endl;

    Label * data = reinterpret_cast<Label*>(inputArray.get_data());

    //std::cout << "data[0]: " << data[0] << std::endl;
    //std::cout << "data[1000]: " << data[1000] << std::endl;
    const bool importImageFilterWillOwnTheBuffer = false;
    importFilter->SetImportPointer( data,
                                    size[0]*size[1], //number of pixels
                                    importImageFilterWillOwnTheBuffer );

    importFilter->SetRegion(region);
    importFilter->Update();
    return importFilter->GetOutput();

}

RealImage<DIMENSION>::Pointer
np_to_itk_real(const np::ndarray & inputArray) {


    using RealImageType = RealImage<DIMENSION>;
    using ShapeType = const long int *;
    typedef itk::ImportImageFilter<Real, DIMENSION > ImportFilterType;
    typename RealImageType::Pointer image = RealImageType::New();
    typename ImportFilterType::SizeType size;
    typename ImportFilterType::RegionType region;
    typename ImportFilterType::IndexType start;
    ImportFilterType::Pointer importFilter = ImportFilterType::New();

    ShapeType shape = reinterpret_cast<ShapeType>(inputArray.get_shape());
    size[1] = shape[0];
    size[0] = shape[1];
    start.Fill(0);

    region.SetIndex(start);
    region.SetSize(size);

    image->SetRegions(region);
    image->Allocate();

    const itk::SpacePrecisionType origin[ 2 ] = { 0.0, 0.0};
    importFilter->SetOrigin( origin );
    const itk::SpacePrecisionType  spacing[ 2 ] =  { 1.0, 1.0};
    importFilter->SetSpacing( spacing );

    Real * data = reinterpret_cast<Real*>(inputArray.get_data());

    const bool importImageFilterWillOwnTheBuffer = false;
    importFilter->SetImportPointer( data,
                                    size[0]*size[1], //number of pixels
                                    importImageFilterWillOwnTheBuffer );
    importFilter->SetRegion(region);
    importFilter->Update();

    return importFilter->GetOutput();
}

template<typename T>
std::vector< T > list_to_vector( const bp::object& iterable )
{
    return std::vector< T >( bp::stl_input_iterator< T >( iterable ),
                             bp::stl_input_iterator< T >( ) );
}

#endif
