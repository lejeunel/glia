#ifndef _glia_type_point_hxx_
#define _glia_type_point_hxx_

#include "glia_image.hxx"
#include "type/object.hxx"

namespace glia {

template <UInt D>
using Point = itk::Index<D>;

template <UInt D, typename T>
class TPixel : public Point<D> {
 public:
  typedef Point<D> Super;
  typedef TPixel<D, T> Self;
  typedef T Data;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;
  static const UInt Dimension = D;
  T data;

  TPixel () {}

  TPixel (Super const& p, T const& data) : Super(p), data(data) {}

  ~TPixel () override {}
};


template <UInt D>
class fPoint : public Object, public itk::Point<double, D> {
 public:
  typedef Object SuperObject;
  typedef itk::Point<double, D> Super;
  typedef fPoint<D> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;

  fPoint () {}

  fPoint (std::initializer_list<double> const& coordinate) {
    assert("Error: incorrect initializing coordinate dimension" &&
           coordinate.size() == D);
    std::copy(coordinate.begin(), coordinate.begin() + D,
              Super::Begin());
  }

  fPoint (Super const& coordinate) : Super(coordinate) {}

  ~fPoint () override {}

  virtual Self& operator+= (Point<D> const& p) {
    for (auto i = 0; i < D; ++i) { Super::operator[](i) += p[i]; }
    return *this;
  }

  virtual Self& operator-= (Point<D> const& p) {
    for (auto i = 0; i < D; ++i) { Super::operator[](i) -= p[i]; }
    return *this;
  }

  virtual void print (std::ostream& os) const {
    os << "(";
    for (auto i = 0; i < D; ++i) {
      os << Super::operator[](i);
      if (i < D - 1) { os << ", "; }
    }
    os << ")";
  }

  friend std::ostream& operator<< (std::ostream& os, Self const& p) {
    for (auto i = 0; i < D; ++i) { os << p[i] << " "; }
    return os;
  }
};


template <UInt D>
class BoundingBox : public Object, public itk::ImageRegion<D> {
 public:
  typedef Object SuperObject;
  typedef itk::ImageRegion<D> Super;
  typedef BoundingBox<D> Self;
  typedef Self* Pointer;
  typedef Self const* ConstPointer;

  BoundingBox () {}

  // BoundingBox (Point<D> const& p)
  // { std::copy(p.begin(), p.end(), Super::GetModifiableIndex()); }

  ~BoundingBox () override {}

  virtual UInt& operator[] (int d)
  { return Super::GetModifiableSize()[d]; }

  virtual UInt operator[] (int d) const { return Super::GetSize()[d]; }

  virtual Self& operator+= (BoundingBox const& rhs) {
    auto& thisIndex = this->GetModifiableIndex();
    auto& thisSize = this->GetModifiableSize();
    auto const& rhsIndex = rhs.GetIndex();
    auto const& rhsSize = rhs.GetSize();
    for (auto i = 0; i < D; ++i) {
      auto lower = std::min(thisIndex[i], rhsIndex[i]);
      auto upper = std::max(thisIndex[i] + thisSize[i],
                            rhsIndex[i] + rhsSize[i]);
      thisIndex[i] = lower;
      thisSize[i] = upper - lower;
    }
    return *this;
  }
};

};

#endif
