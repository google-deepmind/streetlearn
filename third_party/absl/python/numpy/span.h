// Copyright 2018 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This library provides Clif conversion functions from NumpyArray to
// template specializations of absl::Span.
// Note that no data is copied, the `Span`ned memory is owned by the NumpyArray.
//
// Only one direction of conversion is provided: a Span argument can be exposed
// to Python as a Numpy array argument. No conversion is provided for a Span
// return value.
//
// Lifetime management will become complicated if a `Span` returned by a C++
// object is converted into a Numpy array which outlives the C++ object. Such
// memory management concerns are normal for C++ but undesirable to introduce to
// Python.
//
// A `Span` argument can be used to expose an output argument to Python so that
// Python is wholly responsible for memory management of the output object.
// Using an output argument rather than a return value means that memory can be
// reused.
//
// C++:
//  Simulation::Simulation(int buffer_size);
//  void Simulation::RenderFrame(int frame_index, Span<uint8> buffer);
//
// Python:
//  buffer = np.zeroes(1024*768, dtype='uint8')
//  simulation = Simulation(1024*768)
//  simulation.renderFrame(0, buffer)
//  # RGB data can now be read from the buffer.

#ifndef THIRD_PARTY_ABSL_PYTHON_NUMPY_SPAN_H_
#define THIRD_PARTY_ABSL_PYTHON_NUMPY_SPAN_H_

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <cstddef>

#include "absl/types/span.h"
#include "clif/python/postconv.h"
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"

namespace absl {
namespace py_span_internal {

template <typename T>
struct NumpyType;

// Define NumpyType<const T> in terms of NumpyType<T> to avoid having to use
// std::remove_const<T>::type
template <typename T>
struct NumpyType<const T> : NumpyType<T> {};

#ifdef ABSL_NUMPY_MAKE_TYPE_TRAIT
#error "Redefinition of Numpy type trait partial specialization macro"
#endif

#define ABSL_NUMPY_MAKE_TYPE_TRAIT(cxx_type, npy_const, npy_str) \
  template <>                                                    \
  struct NumpyType<cxx_type> {                                   \
    static constexpr NPY_TYPES value = npy_const;                \
    static const char* type_error() {                            \
      return "expected numpy." #npy_str " data";                 \
    }                                                            \
  }

ABSL_NUMPY_MAKE_TYPE_TRAIT(bool, NPY_BOOL, bool8);
ABSL_NUMPY_MAKE_TYPE_TRAIT(signed char, NPY_BYTE, byte);
ABSL_NUMPY_MAKE_TYPE_TRAIT(unsigned char, NPY_UBYTE, ubyte);
ABSL_NUMPY_MAKE_TYPE_TRAIT(short, NPY_SHORT, short);  // NOLINT(runtime/int)
ABSL_NUMPY_MAKE_TYPE_TRAIT(unsigned short, NPY_USHORT,
                           ushort);  // NOLINT(runtime/int)
ABSL_NUMPY_MAKE_TYPE_TRAIT(int, NPY_INT, intc);
ABSL_NUMPY_MAKE_TYPE_TRAIT(unsigned int, NPY_UINT, uintc);
ABSL_NUMPY_MAKE_TYPE_TRAIT(long int, NPY_LONG, long);  // NOLINT(runtime/int)
ABSL_NUMPY_MAKE_TYPE_TRAIT(unsigned long int, NPY_ULONG,
                           ulong);  // NOLINT(runtime/int)
ABSL_NUMPY_MAKE_TYPE_TRAIT(long long int, NPY_LONGLONG,
                           longlong);  // NOLINT(runtime/int)
ABSL_NUMPY_MAKE_TYPE_TRAIT(unsigned long long int, NPY_ULONGLONG,
                           ulonglong);  // NOLINT(runtime/int)
ABSL_NUMPY_MAKE_TYPE_TRAIT(float, NPY_FLOAT, single);
ABSL_NUMPY_MAKE_TYPE_TRAIT(double, NPY_DOUBLE, double);
ABSL_NUMPY_MAKE_TYPE_TRAIT(long double, NPY_LONGDOUBLE, longfloat);

#undef ABSL_NUMPY_MAKE_TYPE_TRAIT

template <typename T>
bool SatisfiesWriteableRequirements(PyArrayObject* npy, Span<T>* c) {
  return PyArray_FLAGS(npy) & NPY_ARRAY_WRITEABLE;
}

template <typename T>
bool SatisfiesWriteableRequirements(PyArrayObject* npy, Span<const T>* c) {
  return true;
}
}  // namespace py_span_internal

// Accept a 1-D Numpy array as a Python representation of a Span.
// Note that the Span will only be valid for the lifetime of the PyObject.
template <typename T>
bool Clif_PyObjAs(PyObject* py, Span<T>* c) {
  // CHECK(c != nullptr);

  if (PyArray_API == nullptr) {
    import_array1(0);
  }

  if (!PyArray_Check(py)) {
    PyErr_SetString(PyExc_TypeError,
                    "The given input is not a NumPy array or matrix.");
    return false;
  }

  auto* npy = reinterpret_cast<PyArrayObject*>(py);

  if (!PyArray_ISCONTIGUOUS(npy)) {
    PyErr_SetString(PyExc_ValueError, "Array must be contiguous.");
    return false;
  }

  if (!py_span_internal::SatisfiesWriteableRequirements(npy, c)) {
    PyErr_SetString(PyExc_TypeError, "Expected a writeable numpy array");
    return false;
  }

  const int numpy_type = PyArray_TYPE(npy);
  if (py_span_internal::NumpyType<T>::value != numpy_type) {
    PyErr_SetString(PyExc_TypeError,
                    py_span_internal::NumpyType<T>::type_error());
    return false;
  }

  const int num_dimensions = PyArray_NDIM(npy);
  npy_intp* dimensions = PyArray_DIMS(npy);
  if (num_dimensions != 1) {
    PyErr_SetString(PyExc_TypeError, "expected a 1-D array");
    return false;
  }

  std::size_t size = *dimensions;

  T* numpy_data;
  PyArray_Descr* descr =
      PyArray_DescrFromType(py_span_internal::NumpyType<T>::value);
  PyArray_AsCArray(&py, &numpy_data, dimensions, 1, descr);

  *c = Span<T>(numpy_data, size);
  return true;
}

// CLIF use `::absl::Span` as NumpyArray

}  // namespace absl

#endif  // THIRD_PARTY_ABSL_PYTHON_NUMPY_SPAN_H_
