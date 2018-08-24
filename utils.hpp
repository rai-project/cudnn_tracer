#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "json.hpp"

using json = nlohmann::json;

using timestamp_t = std::chrono::time_point<std::chrono::system_clock>;

static timestamp_t now() {
  return std::chrono::system_clock::now();
}

static double elapsed_time(timestamp_t start, timestamp_t end) {
  const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
  return elapsed;
}

static uint64_t to_nanoseconds(timestamp_t t) {
  const auto duration = t.time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}
