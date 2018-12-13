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

// This library provides Clif conversion functions for specializations of
// absl::optional. Clif refers to absl::optional<T> as NoneOr<T>.

#ifndef THIRD_PARTY_ABSL_PYTHON_OPTIONAL_H_
#define THIRD_PARTY_ABSL_PYTHON_OPTIONAL_H_

#include "absl/types/optional.h"
#include "clif/python/postconv.h"
#include "clif/python/types.h"

// If Abseil claims std::optional does not exist then it provides an
// absl::optional<T> of its own. In that case, wrap that type. Otherwise, rely
// on third_party/clif/python/types.h to wrap std::optional<T>.
#if !defined(ABSL_HAVE_STD_OPTIONAL)

namespace absl {

// CLIF use `::absl::optional` as NoneOr

// C++ object to python
template <typename T>
PyObject* Clif_PyObjFrom(const optional<T>& opt,
                         const ::clif::py::PostConv& pc) {
  if (!opt) Py_RETURN_NONE;
  using ::clif::Clif_PyObjFrom;
  return Clif_PyObjFrom(*opt, pc.Get(0));
}

// Python to C++ object
template <typename T>
bool Clif_PyObjAs(PyObject* py, optional<T>* c) {
  DCHECK(c != nullptr);
  if (Py_None == py) {  // Uninitialized case.
    c->reset();
    return true;
  }
  // Initialized case.
  using ::clif::Clif_PyObjAs;
  if (!Clif_PyObjAs(py, &c->emplace())) {
    c->reset();
    return false;
  }
  return true;
}

}  // namespace absl

#endif  // !defined(ABSL_HAVE_STD_OPTIONAL)

#endif  // THIRD_PARTY_ABSL_PYTHON_OPTIONAL_H_
