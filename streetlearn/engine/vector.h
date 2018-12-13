// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_VECTOR_H_
#define THIRD_PARTY_STREETLEARN_ENGINE_VECTOR_H_

namespace streetlearn {

// Trivially simple 2d vector class with no support for any operations.
template <class T>
class Vector2 {
 public:
  Vector2() : Vector2(0, 0) {}
  Vector2(T x, T y) : data_{x, y} {}

  T x() const { return data_[0]; }
  T y() const { return data_[1]; }

 private:
  T data_[2];
};

using Vector2_d = Vector2<double>;
using Vector2_i = Vector2<int>;

};  // namespace streetlearn

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_VECTOR_H_
