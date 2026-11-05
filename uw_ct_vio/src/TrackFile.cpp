#include "uw_ct_vio/TrackFile.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace uw_ct_vio {

namespace {
constexpr char kMagic[8] = {'U', 'W', 'T', 'R', 'K', '1', '\0', '\0'};
constexpr uint32_t kVersion = 1;
constexpr size_t kRecordBytes = 8 + 4 * 4 + 1;

template <typename T>
T load(const char* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}
}  // namespace

TrackFile::TrackFile(const std::string& path) : path_(path), in_(path, std::ios::binary) {
  if (!in_) throw std::runtime_error("track file '" + path + "' cannot be opened");
  char h[16];
  in_.read(h, sizeof(h));
  if (!in_ || std::memcmp(h, kMagic, 8) != 0) throw std::runtime_error("'" + path + "' is not a UWTRK1 track file");
  if (load<uint32_t>(h + 8) != kVersion) throw std::runtime_error("'" + path + "' has an unsupported version");
}

bool TrackFile::header() {
  char h[12];
  in_.read(h, sizeof(h));
  if (in_.gcount() == 0 && in_.eof()) return false;
  if (in_.gcount() != static_cast<std::streamsize>(sizeof(h)))
    throw std::runtime_error("track file '" + path_ + "' is truncated in a frame header");
  pendingT_ = load<double>(h);
  pendingN_ = load<uint32_t>(h + 8);
  pending_ = true;
  return true;
}

bool TrackFile::read(double t, std::vector<FeatureObs>* out) {
  out->clear();
  if (!(t > lastT_)) throw std::runtime_error("track file frames must be requested in increasing time");
  lastT_ = t;
  while (true) {
    if (!pending_ && !header()) return false;
    if (pendingT_ >= t - kToleranceS) break;
    in_.seekg(static_cast<std::streamoff>(pendingN_) * kRecordBytes, std::ios::cur);
    if (!in_) throw std::runtime_error("track file '" + path_ + "' is truncated while skipping a frame");
    pending_ = false;
  }
  if (pendingT_ > t + kToleranceS) return false;
  std::vector<char> buf(static_cast<size_t>(pendingN_) * kRecordBytes);
  in_.read(buf.data(), static_cast<std::streamsize>(buf.size()));
  if (in_.gcount() != static_cast<std::streamsize>(buf.size()))
    throw std::runtime_error("track file '" + path_ + "' is truncated in a frame");
  pending_ = false;
  out->reserve(pendingN_);
  for (uint32_t i = 0; i < pendingN_; ++i) {
    const char* r = buf.data() + static_cast<size_t>(i) * kRecordBytes;
    FeatureObs f;
    f.track_id = load<uint64_t>(r);
    const float lx = load<float>(r + 8), ly = load<float>(r + 12), rx = load<float>(r + 16), ry = load<float>(r + 20);
    if (f.track_id == 0 || !std::isfinite(lx) || !std::isfinite(ly)) continue;
    f.left = gtsam::Point2(lx, ly);
    f.has_right = load<uint8_t>(r + 24) != 0 && std::isfinite(rx) && std::isfinite(ry);
    if (f.has_right) f.right = gtsam::Point2(rx, ry);
    out->push_back(f);
  }
  return true;
}

}  // namespace uw_ct_vio
