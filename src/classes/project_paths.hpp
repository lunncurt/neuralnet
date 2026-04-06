#pragma once

#include <filesystem>
#include <string_view>

#ifndef NEURALNET_PROJECT_ROOT
#define NEURALNET_PROJECT_ROOT "."
#endif

inline std::filesystem::path project_root() {
  return std::filesystem::path(NEURALNET_PROJECT_ROOT);
}

inline std::filesystem::path project_path(
    std::initializer_list<std::string_view> parts) {
  auto path = project_root();

  for (const auto part : parts) {
    path /= part;
  }

  return path;
}
