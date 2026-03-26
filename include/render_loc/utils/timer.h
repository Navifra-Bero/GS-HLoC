#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <iostream>
#include <iomanip>

namespace renderloc {

class Timer {
public:
  void start(const std::string& name) {
    starts_[name] = Clock::now();
  }
  double stop(const std::string& name) {
    auto it = starts_.find(name);
    if (it == starts_.end()) return 0.0;
    double ms = std::chrono::duration<double, std::milli>(
      Clock::now() - it->second).count();
    elapsed_[name] = ms;
    return ms;
  }
  double get(const std::string& name) const {
    auto it = elapsed_.find(name);
    return it != elapsed_.end() ? it->second : 0.0;
  }
  void printAll() const {
    std::cout << std::fixed << std::setprecision(2) << "--- Timing ---\n";
    double total = 0;
    for (const auto& [n, ms] : elapsed_) {
      std::cout << "  " << std::setw(20) << std::left << n << ": " << ms << " ms\n";
      total += ms;
    }
    std::cout << "  " << std::setw(20) << std::left << "TOTAL" << ": " << total << " ms\n";
  }

private:
  using Clock = std::chrono::high_resolution_clock;
  std::unordered_map<std::string, Clock::time_point> starts_;
  std::unordered_map<std::string, double> elapsed_;
};

}  // namespace renderloc
