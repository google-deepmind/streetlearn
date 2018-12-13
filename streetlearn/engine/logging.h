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

#ifndef THIRD_PARTY_STREETLEARN_ENGINE_LOGGING_H
#define THIRD_PARTY_STREETLEARN_ENGINE_LOGGING_H

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

#ifdef __GNUC__
#define PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#define PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define NORETURN __attribute__((noreturn))
#else
#define PREDICT_TRUE(x) (x)
#define PREDICT_FALSE(x) (x)
#define NORETURN
#endif

namespace streetlearn {
namespace internal {

struct CheckOpString {
  explicit CheckOpString(std::string* str) : str_(str) {}
  explicit operator bool() const { return PREDICT_FALSE(str_ != nullptr); }
  std::string* const str_;
};

template <typename T1, typename T2>
CheckOpString MakeCheckOpString(const T1& v1, const T2& v2,
                                const char* exprtext) {
  std::ostringstream oss;
  oss << exprtext << " (" << v1 << " vs. " << v2 << ")";
  return CheckOpString(new std::string(oss.str()));
}

#define DEFINE_CHECK_OP_IMPL(name, op)                                    \
  template <typename T1, typename T2>                                     \
  inline CheckOpString name##Impl(const T1& v1, const T2& v2,             \
                                  const char* exprtext) {                 \
    if (PREDICT_TRUE(v1 op v2))                                           \
      return CheckOpString(nullptr);                                      \
    else                                                                  \
      return (MakeCheckOpString)(v1, v2, exprtext);                       \
  }                                                                       \
  inline CheckOpString name##Impl(int v1, int v2, const char* exprtext) { \
    return (name##Impl<int, int>)(v1, v2, exprtext);                      \
  }
DEFINE_CHECK_OP_IMPL(Check_EQ, ==)
DEFINE_CHECK_OP_IMPL(Check_NE, !=)
DEFINE_CHECK_OP_IMPL(Check_LE, <=)
DEFINE_CHECK_OP_IMPL(Check_LT, <)
DEFINE_CHECK_OP_IMPL(Check_GE, >=)
DEFINE_CHECK_OP_IMPL(Check_GT, >)
#undef DEFINE_CHECK_OP_IMPL

class LogMessage {
 public:
  LogMessage(const char* file, int line) {
    std::clog << "[" << file << ":" << line << "] ";
  }

  ~LogMessage() { std::clog << "\n"; }

  std::ostream& stream() && { return std::clog; }
};

class LogMessageFatal {
 public:
  LogMessageFatal(const char* file, int line) {
    stream_ << "[" << file << ":" << line << "] ";
  }

  LogMessageFatal(const char* file, int line, const CheckOpString& result) {
    stream_ << "[" << file << ":" << line << "] Check failed: " << *result.str_;
  }

  ~LogMessageFatal() NORETURN;

  std::ostream& stream() && { return stream_; }

 private:
  std::ostringstream stream_;
};

inline LogMessageFatal::~LogMessageFatal() {
  std::cerr << stream_.str() << std::endl;
  std::abort();
}

struct NullStream {};

template <typename T>
NullStream&& operator<<(NullStream&& s, T&&) {
  return std::move(s);
}

enum class LogLevel { INFO, WARNING, ERROR, FATAL, QFATAL };

LogMessage LogStream(std::integral_constant<LogLevel, LogLevel::INFO>);
LogMessage LogStream(std::integral_constant<LogLevel, LogLevel::WARNING>);
LogMessage LogStream(std::integral_constant<LogLevel, LogLevel::ERROR>);
LogMessageFatal LogStream(std::integral_constant<LogLevel, LogLevel::FATAL>);
LogMessageFatal LogStream(std::integral_constant<LogLevel, LogLevel::QFATAL>);

struct Voidify {
  void operator&(std::ostream&) {}
};

}  // namespace internal
}  // namespace streetlearn

#define CHECK_OP_LOG(name, op, val1, val2, log)                            \
  while (::streetlearn::internal::CheckOpString _result =                  \
             ::streetlearn::internal::name##Impl(val1, val2,               \
                                                 #val1 " " #op " " #val2)) \
  log(__FILE__, __LINE__, _result).stream()

#define CHECK_OP(name, op, val1, val2) \
  CHECK_OP_LOG(name, op, val1, val2, ::streetlearn::internal::LogMessageFatal)

#define CHECK_EQ(val1, val2) CHECK_OP(Check_EQ, ==, val1, val2)
#define CHECK_NE(val1, val2) CHECK_OP(Check_NE, !=, val1, val2)
#define CHECK_LE(val1, val2) CHECK_OP(Check_LE, <=, val1, val2)
#define CHECK_LT(val1, val2) CHECK_OP(Check_LT, <, val1, val2)
#define CHECK_GE(val1, val2) CHECK_OP(Check_GE, >=, val1, val2)
#define CHECK_GT(val1, val2) CHECK_OP(Check_GT, >, val1, val2)

#define QCHECK_EQ(val1, val2) CHECK_OP(Check_EQ, ==, val1, val2)
#define QCHECK_NE(val1, val2) CHECK_OP(Check_NE, !=, val1, val2)
#define QCHECK_LE(val1, val2) CHECK_OP(Check_LE, <=, val1, val2)
#define QCHECK_LT(val1, val2) CHECK_OP(Check_LT, <, val1, val2)
#define QCHECK_GE(val1, val2) CHECK_OP(Check_GE, >=, val1, val2)
#define QCHECK_GT(val1, val2) CHECK_OP(Check_GT, >, val1, val2)

#define CHECK(condition)                                           \
  while (auto _result = ::streetlearn::internal::CheckOpString(    \
             (condition) ? nullptr : new std::string(#condition))) \
  ::streetlearn::internal::LogMessageFatal(__FILE__, __LINE__, _result).stream()

#define QCHECK(condition) CHECK(condition)

#define LOG(level)                                                          \
  decltype(::streetlearn::internal::LogStream(                              \
      std::integral_constant<::streetlearn::internal::LogLevel,             \
                             ::streetlearn::internal::LogLevel::level>()))( \
      __FILE__, __LINE__)                                                   \
      .stream()

#define VLOG(level) ::streetlearn::internal::NullStream()

#define LOG_IF(level, condition)                                            \
  !(condition) ? static_cast<void>(0)                                       \
               : ::streetlearn::internal::Voidify() &                       \
                     decltype(::streetlearn::internal::LogStream(           \
                         std::integral_constant<                            \
                             ::streetlearn::internal::LogLevel,             \
                             ::streetlearn::internal::LogLevel::level>()))( \
                         __FILE__, __LINE__)                                \
                         .stream()

#endif  // THIRD_PARTY_STREETLEARN_ENGINE_LOGGING_H
