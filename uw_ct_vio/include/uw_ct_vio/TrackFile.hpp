// Precomputed feature tracks (scripts/learned_tracks.py) in undistorted normalised coordinates.
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "uw_ct_vio/Estimator.hpp"

namespace uw_ct_vio {

/**
 * Sequential reader for "UWTRK1" files: per frame float64 t (left image header stamp, s), uint32 n,
 * then n records {uint64 id, float32 lx ly rx ry, uint8 has_right}. Frames must be requested in
 * increasing time; frames the runner never asks for are skipped.
 */
class TrackFile {
 public:
  explicit TrackFile(const std::string& path);
  /// Observations of the frame stamped t (within kToleranceS); false if the file has none for t.
  bool read(double t, std::vector<FeatureObs>* out);
  static constexpr double kToleranceS = 0.0025;

 private:
  bool header();
  std::string path_;
  std::ifstream in_;
  bool pending_ = false;
  double pendingT_ = 0.0;
  uint32_t pendingN_ = 0;
  double lastT_ = -1e300;
};

}  // namespace uw_ct_vio
