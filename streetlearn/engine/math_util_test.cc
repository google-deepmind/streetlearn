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

#include "streetlearn/engine/math_util.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace streetlearn {
namespace math {
namespace {

using ::testing::Eq;

template <typename T>
class MathUtilTest : public ::testing::Test {};
TYPED_TEST_SUITE_P(MathUtilTest);

TYPED_TEST_P(MathUtilTest, ClampTest) {
  constexpr TypeParam kLow = static_cast<TypeParam>(7);
  constexpr TypeParam kHigh = static_cast<TypeParam>(42);
  constexpr TypeParam kTestValue = static_cast<TypeParam>(33);

  EXPECT_THAT(Clamp(kLow, kHigh, kTestValue), Eq(kTestValue));
  EXPECT_THAT(Clamp(kLow, kHigh, kLow), Eq(kLow));
  EXPECT_THAT(Clamp(kLow, kHigh, kHigh), Eq(kHigh));
  EXPECT_THAT(Clamp(kLow, kHigh, kLow - 1), Eq(kLow));
  EXPECT_THAT(Clamp(kLow, kHigh, kHigh + 1), Eq(kHigh));

  EXPECT_THAT(Clamp(kLow, kLow, kLow - 1), Eq(kLow));
  EXPECT_THAT(Clamp(kLow, kLow, kLow), Eq(kLow));
  EXPECT_THAT(Clamp(kLow, kLow, kLow + 1), Eq(kLow));
}

TEST(StreetLearn, AngleConversionsTest) {
  EXPECT_DOUBLE_EQ(DegreesToRadians(0), 0);
  EXPECT_DOUBLE_EQ(DegreesToRadians(90), M_PI / 2);
  EXPECT_DOUBLE_EQ(DegreesToRadians(-90), -M_PI / 2);
  EXPECT_DOUBLE_EQ(DegreesToRadians(180), M_PI);
}

REGISTER_TYPED_TEST_SUITE_P(MathUtilTest, ClampTest);

using Types = ::testing::Types<int, float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(TpedMathUtilTests, MathUtilTest, Types);

}  // namespace
}  // namespace math
}  // namespace streetlearn
