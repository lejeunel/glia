#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/def.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/return_internal_reference.hpp>
#include "gadget/gadgets.hxx"

namespace bp = boost::python;
namespace np = boost::python::numpy;


BOOST_PYTHON_MODULE(libglia)
{
  Py_Initialize();
  np::initialize();

  bp::def("watershed", watershed_operation,
      bp::args("inputImageFile",
               "level",
               "relabel"),
      "foo's docstring");

  bp::def("pre_merge", pre_merge_operation,
      bp::args("labelArray",
               "pbArray",
               "maskArray",
               "sizeThresholds",
               "rpbThreshold",
               "relabel"),
      "foo's docstring");
}
