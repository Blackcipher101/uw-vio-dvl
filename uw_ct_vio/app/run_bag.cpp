// Offline runner: rosbag2 -> KLT stereo tracker -> Estimator -> TUM trajectory of the left camera.
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <yaml-cpp/yaml.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Eigenvalues>

#include <cctype>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "uw_ct_vio/DenseRegistration.hpp"
#include "uw_ct_vio/DenseStereo.hpp"
#include "uw_ct_vio/Estimator.hpp"
#include "uw_ct_vio/KltStereoTracker.hpp"
#include "uw_ct_vio/TrackFile.hpp"

using namespace uw_ct_vio;

namespace {

constexpr double kStereoPairTolS = 0.005;
constexpr double kGravity = 9.81;

struct Args {
  std::string bag, intrinsics, imucam, imu_yaml, left = "cam0", right = "cam1";
  std::string dense_log;  ///< optional CSV: one row per dense registration (diagnostics)
  std::string left_topic, right_topic, imu_topic, pressure_topic, out;
  std::string track_file, track_mode = "augment";  ///< learned tracks (scripts/learned_tracks.py)
  double water_density = 1000.0, pixel_sigma = 1.0, lag_s = 3.0, resize = 1.0;
  double pressure_scale = 1.0;  ///< Pa per message unit (1 for Pa, 100 for mbar drivers such as Bar30)
  int max_features = 150;
  int lcn = 0, snr_sigma = 0, gravity_prior = 0, degeneracy = 0, keep_velocity = 0;
  double ncc_min = 0.0, lcn_min_floor = 0.0, lcn_auto_max_range = 150.0;
  int ncc_noise_aware = 0, lcn_stereo = 1;
  int map_predict = 0, reacquire = 0;
  /// 1: left-right stereo pose from the --intrinsics camchain (T_cn_cnm1, e.g. calibrated in water) instead of
  /// from the two cam-IMU extrinsics in --imucam (e.g. calibrated in air); the left cam-IMU pose is kept.
  int stereo_from_intrinsics = 0;
  double gate_sigma = 5.0, huber_k = 1.345;  ///< see EstimatorConfig
  int dense = 0;                       ///< step 6: dense stereo keyframe registration factors
  double dense_interval_s = 0.5;       ///< keyframe spacing [s]
  double dense_cov_scale = 3.0, dense_min_eig = 1e3;  ///< see EstimatorConfig
  // DVL (see EstimatorConfig::dvl_*). Enabled by --dvl-file or --dvl-topic; both need --dvl-extrinsic.
  std::string dvl_file;       ///< CSV t,vx,vy,vz,valid (DVL frame [m/s], bag clock); header line optional
  std::string dvl_topic;      ///< geometry_msgs TwistWithCovarianceStamped or TwistStamped, twist.linear in the DVL frame
  std::string dvl_extrinsic;  ///< yaml with T_B_D: 4x4 DVL frame in the body (IMU) frame
  EstimatorConfig dvl_defaults;  ///< only the dvl_* fields are read
  /// Test only: "t0:dur[,t0:dur...]" seconds after the first image; frames in a window get empty feature lists.
  std::string vision_blackout;
  /// Optional per-frame CSV for failure prediction models; columns are fixed (see kFrameLogHeader).
  std::string frame_log;
};

Args parse(int argc, char** argv) {
  Args a;
  std::map<std::string, std::string*> s{{"--bag", &a.bag}, {"--intrinsics", &a.intrinsics}, {"--imucam", &a.imucam},
                                        {"--imu-yaml", &a.imu_yaml}, {"--left", &a.left}, {"--right", &a.right},
                                        {"--left-topic", &a.left_topic}, {"--right-topic", &a.right_topic},
                                        {"--imu-topic", &a.imu_topic}, {"--pressure-topic", &a.pressure_topic},
                                        {"--out", &a.out}, {"--dense-log", &a.dense_log},
                                        {"--track-file", &a.track_file}, {"--track-mode", &a.track_mode},
                                        {"--dvl-file", &a.dvl_file}, {"--dvl-topic", &a.dvl_topic},
                                        {"--dvl-extrinsic", &a.dvl_extrinsic},
                                        {"--vision-blackout", &a.vision_blackout}, {"--frame-log", &a.frame_log}};
  std::map<std::string, double*> d{{"--water-density", &a.water_density}, {"--pixel-sigma", &a.pixel_sigma},
                                   {"--lag", &a.lag_s},
                                   {"--resize", &a.resize}, {"--pressure-scale", &a.pressure_scale},
                                   {"--dense-interval", &a.dense_interval_s}, {"--dense-cov-scale", &a.dense_cov_scale},
                                   {"--dense-min-eig", &a.dense_min_eig},
                                   {"--gate-sigma", &a.gate_sigma}, {"--huber-k", &a.huber_k},
                                   {"--dvl-sigma-abs", &a.dvl_defaults.dvl_sigma_abs_mps},
                                   {"--dvl-sigma-rel", &a.dvl_defaults.dvl_sigma_rel},
                                   {"--dvl-time-offset", &a.dvl_defaults.dvl_time_offset_s},
                                   {"--dvl-gate-quantile", &a.dvl_defaults.dvl_gate_quantile},
                                   {"--dvl-gate-pred-sigma", &a.dvl_defaults.dvl_gate_pred_sigma_mps},
                                   {"--dvl-relock-after", &a.dvl_defaults.dvl_relock_after_s},
                                   {"--dvl-max-speed", &a.dvl_defaults.dvl_max_speed_mps},
                                   {"--dvl-recovery-vel-sigma", &a.dvl_defaults.dvl_recovery_vel_sigma_mps}};
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (s.count(k)) *s[k] = argv[i + 1];
    else if (d.count(k)) *d[k] = std::stod(argv[i + 1]);
    else if (k == "--max-features") a.max_features = std::stoi(argv[i + 1]);
    else if (k == "--lcn") a.lcn = std::stoi(argv[i + 1]);
    else if (k == "--gravity-prior") a.gravity_prior = std::stoi(argv[i + 1]);
    else if (k == "--degeneracy") a.degeneracy = std::stoi(argv[i + 1]);
    else if (k == "--keep-velocity") a.keep_velocity = std::stoi(argv[i + 1]);
    else if (k == "--snr-sigma") a.snr_sigma = std::stoi(argv[i + 1]);
    else if (k == "--ncc-min") a.ncc_min = std::stod(argv[i + 1]);
    else if (k == "--lcn-min-floor") a.lcn_min_floor = std::stod(argv[i + 1]);
    else if (k == "--ncc-noise-aware") a.ncc_noise_aware = std::stoi(argv[i + 1]);
    else if (k == "--lcn-stereo") a.lcn_stereo = std::stoi(argv[i + 1]);
    else if (k == "--lcn-auto-max-range") a.lcn_auto_max_range = std::stod(argv[i + 1]);
    else if (k == "--map-predict") a.map_predict = std::stoi(argv[i + 1]);
    else if (k == "--reacquire") a.reacquire = std::stoi(argv[i + 1]);
    else if (k == "--dense") a.dense = std::stoi(argv[i + 1]);
    else if (k == "--stereo-from-intrinsics") a.stereo_from_intrinsics = std::stoi(argv[i + 1]);
    else throw std::invalid_argument("unknown argument " + k);
  }
  for (auto* req : {&a.bag, &a.intrinsics, &a.imucam, &a.imu_yaml, &a.left_topic, &a.right_topic, &a.imu_topic, &a.out})
    if (req->empty()) throw std::invalid_argument("missing a required argument (see README in uw_ct_vio)");
  if ((!a.dvl_file.empty() || !a.dvl_topic.empty()) && a.dvl_extrinsic.empty())
    throw std::invalid_argument("--dvl-file / --dvl-topic need --dvl-extrinsic (yaml with T_B_D)");
  if (!a.dvl_file.empty() && !a.dvl_topic.empty()) throw std::invalid_argument("give --dvl-file or --dvl-topic, not both");
  return a;
}

CameraModel camera(const YAML::Node& n, const std::string& name) {
  CameraModel c;
  const std::string model = n["distortion_model"].as<std::string>();
  if (n["camera_model"].as<std::string>() != "pinhole") throw std::invalid_argument(name + ": only pinhole supported");
  if (model == "equidistant") c.distortion = CameraModel::Distortion::Equidistant;
  else if (model == "radtan") c.distortion = CameraModel::Distortion::RadTan;
  else throw std::invalid_argument(name + ": unsupported distortion " + model);
  const auto in = n["intrinsics"].as<std::vector<double>>();
  const auto res = n["resolution"].as<std::vector<int>>();
  c.fx = in.at(0), c.fy = in.at(1), c.cx = in.at(2), c.cy = in.at(3);
  c.width = res.at(0), c.height = res.at(1);
  c.coeffs = n["distortion_coeffs"].as<std::vector<double>>();
  c.validate(name);
  return c;
}

/// Intrinsics for images resized by s (same convention as OKVIS/SVIn: c * s, distortion unchanged).
CameraModel scaled(CameraModel c, double s) {
  c.fx *= s, c.fy *= s, c.cx *= s, c.cy *= s;
  c.width = static_cast<int>(std::lround(c.width * s)), c.height = static_cast<int>(std::lround(c.height * s));
  return c;
}

gtsam::Pose3 pose4x4(const YAML::Node& n) {
  gtsam::Matrix4 m;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) m(r, c) = n[r][c].as<double>();
  return gtsam::Pose3(m);
}

struct DvlRow {
  double t;
  gtsam::Vector3 v;
  bool valid;
};

/// CSV "t,vx,vy,vz,valid"; a first line starting with a letter is a header. Rows must be in time order.
std::vector<DvlRow> loadDvlCsv(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read --dvl-file " + path);
  std::vector<DvlRow> rows;
  std::string line;
  size_t lineNo = 0;
  while (std::getline(in, line)) {
    ++lineNo;
    if (line.empty() || line[0] == '#') continue;
    if (std::isalpha(static_cast<unsigned char>(line[0]))) continue;
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> f;
    while (std::getline(ss, cell, ',')) f.push_back(std::stod(cell));
    if (f.size() < 4) throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": need t,vx,vy,vz[,valid]");
    const bool valid = f.size() < 5 || f[4] != 0.0;
    if (!rows.empty() && !(f[0] > rows.back().t))
      throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": DVL stamps must increase");
    rows.push_back({f[0], gtsam::Vector3(f[1], f[2], f[3]), valid});
  }
  return rows;
}

/// "t0:dur,t0:dur" -> windows [t0, t0 + dur) in seconds.
std::vector<std::pair<double, double>> parseBlackouts(const std::string& spec) {
  std::vector<std::pair<double, double>> w;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const auto c = item.find(':');
    if (c == std::string::npos) throw std::invalid_argument("--vision-blackout expects t0:dur[,t0:dur...]");
    const double t0 = std::stod(item.substr(0, c)), dur = std::stod(item.substr(c + 1));
    if (!(t0 >= 0.0 && dur > 0.0)) throw std::invalid_argument("--vision-blackout: need t0 >= 0 and dur > 0");
    w.emplace_back(t0, dur);
  }
  return w;
}

constexpr const char* kFrameLogHeader =
    "t,tracked,stereo,new_landmarks,landmark_factors,gated,candidates,vision_used,vision_healthy,consecutive_gated,"
    "recoveries,dvl_factors,dvl_gated,img_mean,img_std,n_features";

double stamp(const builtin_interfaces::msg::Time& t) { return t.sec + 1e-9 * t.nanosec; }

template <typename T>
T deserialize(const rosbag2_storage::SerializedBagMessage& m) {
  rclcpp::SerializedMessage sm(*m.serialized_data);
  T out;
  rclcpp::Serialization<T>().deserialize_message(&sm, &out);
  return out;
}

cv::Mat grey(const sensor_msgs::msg::Image& m) {
  if (m.encoding == "mono8") return cv::Mat(m.height, m.width, CV_8UC1, const_cast<uint8_t*>(m.data.data()), m.step).clone();
  if (m.encoding == "bgr8" || m.encoding == "rgb8") {
    cv::Mat c(m.height, m.width, CV_8UC3, const_cast<uint8_t*>(m.data.data()), m.step), g;
    cv::cvtColor(c, g, m.encoding == "bgr8" ? cv::COLOR_BGR2GRAY : cv::COLOR_RGB2GRAY);
    return g;
  }
  throw std::runtime_error("unsupported image encoding " + m.encoding);
}

cv::Mat grey(const sensor_msgs::msg::CompressedImage& m) {
  cv::Mat g = cv::imdecode(m.data, cv::IMREAD_GRAYSCALE);
  if (g.empty()) throw std::runtime_error("failed to decode compressed image (" + m.format + ")");
  return g;
}

struct Frame {
  double t;      ///< left image stamp shifted into IMU time
  double tRaw;   ///< left image header stamp (keys the track file)
  cv::Mat l, r;
};

// Learned-track ids are offset so they never collide with KLT ids in augment mode.
constexpr uint64_t kLearnedIdOffset = uint64_t{1} << 62;

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = parse(argc, argv);
    const YAML::Node intr = YAML::LoadFile(a.intrinsics), ext = YAML::LoadFile(a.imucam), imu = YAML::LoadFile(a.imu_yaml);
    if (!(a.resize > 0.0 && a.resize <= 1.0)) throw std::invalid_argument("--resize must be in (0, 1]");
    const CameraModel camL = scaled(camera(intr[a.left], a.left), a.resize);
    const CameraModel camR = scaled(camera(intr[a.right], a.right), a.resize);
    EstimatorConfig ec;
    ec.T_BC0 = pose4x4(ext[a.left]["T_cam_imu"]).inverse();
    ec.T_BC1 = pose4x4(ext[a.right]["T_cam_imu"]).inverse();
    if (a.stereo_from_intrinsics) {
      // Kalibr: camN's T_cn_cnm1 maps points from cam(N-1) into camN, i.e. it is cam(N-1)'s pose in camN.
      gtsam::Pose3 T_LR;
      if (intr[a.left]["T_cn_cnm1"] && a.right == "cam" + std::to_string(std::stoi(a.left.substr(3)) - 1)) {
        T_LR = pose4x4(intr[a.left]["T_cn_cnm1"]);
      } else if (intr[a.right]["T_cn_cnm1"] && a.left == "cam" + std::to_string(std::stoi(a.right.substr(3)) - 1)) {
        T_LR = pose4x4(intr[a.right]["T_cn_cnm1"]).inverse();
      } else {
        throw std::invalid_argument("--stereo-from-intrinsics: " + a.intrinsics + " has no T_cn_cnm1 linking " + a.left +
                                    " and " + a.right + " (adjacent cameraN names required)");
      }
      const gtsam::Pose3 d = ec.T_BC0.between(ec.T_BC1).between(T_LR);
      std::cerr << "stereo pose from " << a.intrinsics << ": differs from the cam-IMU chain by "
                << gtsam::Rot3::Logmap(d.rotation()).norm() * 180.0 / M_PI << " deg, " << d.translation().norm() * 1e3
                << " mm\n";
      ec.T_BC1 = ec.T_BC0 * T_LR;
    }
    const double shift = ext[a.left]["timeshift_cam_imu"] ? ext[a.left]["timeshift_cam_imu"].as<double>() : 0.0;
    ec.acc_noise_density = imu["accelerometer_noise_density"].as<double>();
    ec.gyr_noise_density = imu["gyroscope_noise_density"].as<double>();
    ec.acc_random_walk = imu["accelerometer_random_walk"].as<double>();
    ec.gyr_random_walk = imu["gyroscope_random_walk"].as<double>();
    ec.focal_px = camL.fx;
    ec.pixel_sigma = a.pixel_sigma;
    ec.lag_s = a.lag_s;
    ec.gate_sigma = a.gate_sigma;
    ec.huber_k = a.huber_k;
    ec.gravity_prior = a.gravity_prior != 0;
    ec.degeneracy_aware = a.degeneracy != 0;
    ec.recovery_keep_velocity = a.keep_velocity != 0;
    const bool useDvl = !a.dvl_file.empty() || !a.dvl_topic.empty();
    if (useDvl) {
      const EstimatorConfig& d = a.dvl_defaults;
      ec.dvl_enabled = true;
      ec.dvl_T_BD = pose4x4(YAML::LoadFile(a.dvl_extrinsic)["T_B_D"]);
      ec.dvl_sigma_abs_mps = d.dvl_sigma_abs_mps, ec.dvl_sigma_rel = d.dvl_sigma_rel;
      ec.dvl_time_offset_s = d.dvl_time_offset_s, ec.dvl_gate_quantile = d.dvl_gate_quantile;
      ec.dvl_gate_pred_sigma_mps = d.dvl_gate_pred_sigma_mps, ec.dvl_relock_after_s = d.dvl_relock_after_s;
      ec.dvl_max_speed_mps = d.dvl_max_speed_mps, ec.dvl_recovery_vel_sigma_mps = d.dvl_recovery_vel_sigma_mps;
    }
    ec.dense_cov_scale = a.dense_cov_scale;
    ec.dense_min_eig = a.dense_min_eig;
    if (a.dense && !(a.dense_interval_s > 0.0 && a.dense_cov_scale > 0.0 && a.dense_min_eig > 0.0))
      throw std::invalid_argument("--dense-interval, --dense-cov-scale and --dense-min-eig must be > 0");
    if (ec.T_BC0.between(ec.T_BC1).translation().x() <= 0)
      throw std::invalid_argument("right camera is not to the right of the left one; swap --left/--right");
    KltConfig kc;
    kc.max_features = a.max_features;
    kc.lcn = a.lcn == 1;
    kc.lcn_auto = a.lcn == 2;  // --lcn 2: decide from the dynamic range of the first frames
    kc.snr_sigma = a.snr_sigma != 0;
    kc.ncc_min = a.ncc_min;
    kc.lcn_min_floor = a.lcn_min_floor;
    kc.ncc_noise_aware = a.ncc_noise_aware != 0;
    kc.lcn_stereo = a.lcn_stereo != 0;
    kc.lcn_auto_max_range = a.lcn_auto_max_range;
    kc.base_sigma_px = a.pixel_sigma;
    kc.map_predict = a.map_predict != 0;
    kc.reacquire = a.reacquire != 0;
    KltStereoTracker tracker(kc, camL, camR, ec.T_BC0.between(ec.T_BC1));
    if (a.track_mode != "augment" && a.track_mode != "replace")
      throw std::invalid_argument("--track-mode must be augment or replace");
    std::unique_ptr<TrackFile> trackFile;
    if (!a.track_file.empty()) trackFile = std::make_unique<TrackFile>(a.track_file);
    uint64_t trackFileMisses = 0;
    Estimator est(ec);
    std::unique_ptr<DenseStereo> dense;
    if (a.dense) dense = std::make_unique<DenseStereo>(DenseStereoConfig(), camL, camR, ec.T_BC0.between(ec.T_BC1));
    IcpConfig icp;
    std::ofstream denseLog;
    if (!a.dense_log.empty()) {
      denseLog.open(a.dense_log);
      if (!denseLog) throw std::runtime_error("cannot write --dense-log " + a.dense_log);
      denseLog << std::setprecision(9) << std::fixed
               << "# t_ref t_cur ok reason inliers rmse_m | init T_ref_cur tx ty tz qx qy qz qw | icp T_ref_cur tx ty tz qx qy qz qw"
                  " | eigenvalues of H ascending (6)\n";
    }
    auto logPose = [&](const gtsam::Pose3& T) {
      const auto q = T.rotation().toQuaternion();
      denseLog << " " << T.x() << " " << T.y() << " " << T.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
    };
    icp.solve_abs_eig = ec.dense_min_eig;  // the ICP solves every direction the factor will keep
    struct Keyframe {
      double t;
      OrganisedCloud cloud;
    };
    std::optional<Keyframe> keyframe;
    uint64_t denseKeyframes = 0, denseOk = 0, denseNoRef = 0;
    uint64_t denseFail[5] = {0, 0, 0, 0, 0};  // indexed by IcpFailure
    double denseMs = 0;

    const std::vector<DvlRow> dvlRows = a.dvl_file.empty() ? std::vector<DvlRow>() : loadDvlCsv(a.dvl_file);
    size_t dvlNext = 0;
    const auto blackouts = parseBlackouts(a.vision_blackout);
    std::ofstream frameLog;
    if (!a.frame_log.empty()) {
      frameLog.open(a.frame_log);
      if (!frameLog) throw std::runtime_error("cannot write --frame-log " + a.frame_log);
      frameLog << std::setprecision(9) << std::fixed << kFrameLogHeader << "\n";
    }
    double firstImageT = NAN;
    uint64_t blackoutFrames = 0;

    rosbag2_cpp::Reader reader;
    reader.open(a.bag);
    enum class DvlMsg { None, Twist, TwistCov } dvlMsg = DvlMsg::None;
    if (!a.dvl_topic.empty()) {
      for (const auto& tm : reader.get_all_topics_and_types()) {
        if (tm.name != a.dvl_topic) continue;
        if (tm.type == "geometry_msgs/msg/TwistStamped") dvlMsg = DvlMsg::Twist;
        else if (tm.type == "geometry_msgs/msg/TwistWithCovarianceStamped") dvlMsg = DvlMsg::TwistCov;
        else throw std::invalid_argument("--dvl-topic " + a.dvl_topic + " has unsupported type " + tm.type);
      }
      if (dvlMsg == DvlMsg::None) throw std::invalid_argument("--dvl-topic " + a.dvl_topic + " is not in the bag");
    }
    std::ofstream out(a.out);
    out << std::setprecision(9) << std::fixed;
    std::map<double, cv::Mat> pendL, pendR;
    std::deque<Frame> frames;
    double lastImu = -1, p0 = NAN;
    uint64_t n = 0, noVision = 0;
    const auto& c = est.counters();
    double trackMs = 0, estMs = 0;
    double lastTrackT = NAN;  // time of the previous frame given to the tracker
    uint64_t sumPredicted = 0, sumRetried = 0, sumReacqTried = 0, sumReacquired = 0, sumReacqNoStereo = 0;
    auto tryFrames = [&]() {
      while (!frames.empty() && lastImu >= frames.front().t) {
        Frame f = std::move(frames.front());
        frames.pop_front();
        const auto c0 = std::chrono::steady_clock::now();
        std::vector<FeatureObs> obs;
        // Step 5: the estimator's IMU prediction for this frame, before the frame is added.
        MapPrediction pred;
        const MapPrediction* predPtr = nullptr;
        if (kc.map_predict || kc.reacquire) {
          if (const auto T_WC = est.predictLeftCamera(f.t)) {
            // Relative motion only if the estimator's latest state is the tracker's previous frame
            // (same double assigned on both sides, so exact comparison is valid).
            if (est.latestTime() == lastTrackT) pred.T_prev_cur = est.latestLeftCamera().between(*T_WC);
            for (const auto& [id, P] : est.liveLandmarks()) pred.landmarks_cur.emplace_back(id, T_WC->transformTo(P));
            predPtr = &pred;
          }
        }
        if (!trackFile || a.track_mode == "augment") {
          obs = tracker.track(f.l, f.r, predPtr);
          lastTrackT = f.t;
          const auto& ks = tracker.lastStats();
          sumPredicted += ks.predicted;
          sumRetried += ks.predict_retried;
          sumReacqTried += ks.reacq_tried;
          sumReacquired += ks.reacquired;
          sumReacqNoStereo += ks.reacq_no_stereo;
        }
        if (trackFile) {
          std::vector<FeatureObs> learned;
          if (!trackFile->read(f.tRaw, &learned)) ++trackFileMisses;
          for (auto& o : learned) {
            o.track_id += kLearnedIdOffset;
            obs.push_back(o);
          }
        }
        if (std::isnan(firstImageT)) firstImageT = f.t;
        for (const auto& [t0, dur] : blackouts) {
          if (f.t - firstImageT >= t0 && f.t - firstImageT < t0 + dur) {
            obs.clear();
            ++blackoutFrames;
            break;
          }
        }
        const auto c1 = std::chrono::steady_clock::now();
        // Step 6: on a keyframe, register its dense cloud to the previous keyframe's.
        std::optional<OrganisedCloud> cloud;
        DenseConstraint dc{0.0, gtsam::Pose3(), gtsam::Matrix66::Zero()};
        const DenseConstraint* dcPtr = nullptr;
        if (dense && (!keyframe || f.t - keyframe->t >= a.dense_interval_s)) {
          const auto d0 = std::chrono::steady_clock::now();
          cloud = dense->compute(f.l, f.r);
          ++denseKeyframes;
          if (keyframe) {
            const auto T_W_Cref = est.leftCameraAt(keyframe->t);
            const auto T_W_Ccur = est.predictLeftCamera(f.t);
            if (T_W_Cref && T_W_Ccur) {
              const gtsam::Pose3 init = T_W_Cref->between(*T_W_Ccur);
              const IcpResult r = registerClouds(keyframe->cloud, *cloud, init, icp);
              if (denseLog.is_open()) {
                denseLog << keyframe->t << " " << f.t << " " << r.ok << " " << toString(r.reason) << " " << r.inliers << " "
                         << r.rmse_m;
                logPose(init);
                logPose(r.T_ref_cur);
                const Eigen::SelfAdjointEigenSolver<gtsam::Matrix66> es(r.H);
                for (int k = 0; k < 6; ++k) denseLog << " " << es.eigenvalues()(k);
                denseLog << "\n";
              }
              if (r.ok) {
                ++denseOk;
                dc = {keyframe->t, r.T_ref_cur, r.H};
                dcPtr = &dc;
              } else {
                ++denseFail[static_cast<int>(r.reason)];
              }
            } else {
              ++denseNoRef;  // reference left the window or the estimator restarted: reseed from this frame
            }
          }
          denseMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d0).count();
        }
        FrameStats st;
        if (!est.addFrame(f.t, obs, &st, dcPtr)) continue;
        if (cloud) keyframe = Keyframe{f.t, std::move(*cloud)};
        const auto c2 = std::chrono::steady_clock::now();
        trackMs += std::chrono::duration<double, std::milli>(c1 - c0).count();
        estMs += std::chrono::duration<double, std::milli>(c2 - c1).count();
        noVision += !st.vision_used;
        if (frameLog.is_open()) {
          cv::Scalar mean, sd;
          cv::meanStdDev(f.l, mean, sd);
          const auto& ks = tracker.lastStats();
          frameLog << f.t << "," << ks.tracked << "," << ks.stereo << "," << st.new_landmarks << "," << st.landmark_factors << ","
                   << st.gated << "," << st.candidates << "," << st.vision_used << "," << st.vision_healthy << ","
                   << est.consecutiveGated() << "," << c.gate_recoveries << "," << st.dvl_factors << "," << st.dvl_gated << ","
                   << mean[0] << "," << sd[0] << "," << obs.size() << "\n";
        }
        const gtsam::Pose3 T = est.latestLeftCamera();
        const auto q = T.rotation().toQuaternion();
        out << f.t << " " << T.x() << " " << T.y() << " " << T.z() << " " << q.x() << " " << q.y() << " " << q.z() << " "
            << q.w() << "\n";
        if (++n % 200 == 0) {
          const auto& ks = tracker.lastStats();
          std::cout << n << " frames t=" << f.t << " tracked " << ks.tracked << " stereo " << ks.stereo << " new_lm "
                    << st.new_landmarks << " lm_factors " << st.landmark_factors << " gated " << st.gated
                    << " ncc_rej " << ks.ncc_rejected << " noise " << ks.noise_sigma << " mean_sigma " << ks.mean_sigma_px
                    << " | ms/frame track " << trackMs / n << " est " << estMs / n << std::endl;
        }
      }
    };
    while (reader.has_next()) {
      auto m = reader.read_next();
      const std::string& topic = m->topic_name;
      if (topic == a.imu_topic) {
        const auto s = deserialize<sensor_msgs::msg::Imu>(*m);
        const double t = stamp(s.header.stamp);
        est.addImu({t, gtsam::Vector3(s.linear_acceleration.x, s.linear_acceleration.y, s.linear_acceleration.z),
                    gtsam::Vector3(s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z)});
        lastImu = std::max(lastImu, t);
        // Sidecar DVL samples up to the IMU clock (after the extrinsic time offset), so the frames
        // released by tryFrames() see every DVL sample at or before their stamp.
        while (dvlNext < dvlRows.size() && dvlRows[dvlNext].t + ec.dvl_time_offset_s <= t) {
          est.addDvl(dvlRows[dvlNext].t, dvlRows[dvlNext].v, dvlRows[dvlNext].valid);
          ++dvlNext;
        }
        tryFrames();
      } else if (dvlMsg != DvlMsg::None && topic == a.dvl_topic) {
        if (dvlMsg == DvlMsg::Twist) {
          const auto d = deserialize<geometry_msgs::msg::TwistStamped>(*m);
          est.addDvl(stamp(d.header.stamp), gtsam::Vector3(d.twist.linear.x, d.twist.linear.y, d.twist.linear.z), true);
        } else {
          const auto d = deserialize<geometry_msgs::msg::TwistWithCovarianceStamped>(*m);
          const auto& l = d.twist.twist.linear;
          const auto& cv6 = d.twist.covariance;  // row-major 6x6; the linear block is the upper-left 3x3
          gtsam::Matrix3 cov;
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) cov(i, j) = cv6[6 * i + j];
          est.addDvl(stamp(d.header.stamp), gtsam::Vector3(l.x, l.y, l.z), true, cov);
        }
      } else if (!a.pressure_topic.empty() && topic == a.pressure_topic) {
        const auto p = deserialize<sensor_msgs::msg::FluidPressure>(*m);
        if (std::isnan(p0)) p0 = p.fluid_pressure;
        // Depth relative to the first reading; the estimator ties it to the body z at that time.
        est.addDepth(stamp(p.header.stamp), (p.fluid_pressure - p0) * a.pressure_scale / (a.water_density * kGravity));
      } else if (topic == a.left_topic || topic == a.right_topic) {
        const bool compressed = topic.size() > 11 && topic.compare(topic.size() - 11, 11, "/compressed") == 0;
        double t;
        cv::Mat g;
        if (compressed) {
          const auto im = deserialize<sensor_msgs::msg::CompressedImage>(*m);
          t = stamp(im.header.stamp);
          g = grey(im);
        } else {
          const auto im = deserialize<sensor_msgs::msg::Image>(*m);
          t = stamp(im.header.stamp);
          g = grey(im);
        }
        if (a.resize != 1.0) cv::resize(g, g, cv::Size(), a.resize, a.resize, cv::INTER_LINEAR);
        auto& mine = topic == a.left_topic ? pendL : pendR;
        auto& other = topic == a.left_topic ? pendR : pendL;
        auto it = other.lower_bound(t - kStereoPairTolS);
        if (it != other.end() && std::abs(it->first - t) < kStereoPairTolS) {
          const bool isLeft = topic == a.left_topic;
          const double tl = isLeft ? t : it->first;
          frames.push_back({tl + shift, tl, isLeft ? g : it->second, isLeft ? it->second : g});
          other.erase(other.begin(), std::next(it));
        } else {
          mine[t] = g;
          while (!mine.empty() && mine.begin()->first < t - 1.0) mine.erase(mine.begin());
        }
      }
    }
    tryFrames();
    std::cout << "lcn: " << (tracker.lastStats().lcn_active ? "on" : "off") << " (auto median dynamic range "
              << tracker.lastStats().auto_range << ")" << std::endl;
    if (trackFile) std::cout << "track file: " << a.track_file << " (" << a.track_mode << "), frames missing " << trackFileMisses << std::endl;
    if (kc.map_predict || kc.reacquire)
      std::cout << "step5: predicted " << sumPredicted << " (retried " << sumRetried << "), reacquisition tried "
                << sumReacqTried << ", accepted " << sumReacquired << ", dropped without stereo " << sumReacqNoStereo
                << std::endl;
    if (dense) {
      const uint64_t failed = denseFail[1] + denseFail[2] + denseFail[3] + denseFail[4];
      std::cout << "step6: keyframes " << denseKeyframes << " registrations ok/failed " << denseOk << "/" << failed
                << " (non-finite " << denseFail[1] << ", too-few-inliers " << denseFail[2] << ", low-fraction "
                << denseFail[3] << ", diverged " << denseFail[4] << ", no reference " << denseNoRef << ") factors added "
                << c.dense_factors << ", rejected-by-gate " << c.dense_gated << ", dropped-stale " << c.dense_stale
                << ", invalid " << c.dense_invalid << ", mean kept rank "
                << (c.dense_factors ? static_cast<double>(c.dense_rank_sum) / c.dense_factors : 0.0) << ", ms per keyframe "
                << (denseKeyframes ? denseMs / denseKeyframes : 0.0) << std::endl;
    }
    std::cout << "step3: gravity factors " << c.gravity_factors << ", degenerate frames " << c.degenerate_frames
              << " (directions " << c.degenerate_dirs << ")" << std::endl;
    std::cout << "done: " << n << " frames, " << noVision << " without vision; imu dropped " << c.imu_dropped
              << ", imu gaps " << c.imu_gaps << ", depth dropped " << c.depth_dropped << ", solver failures "
              << c.solver_failures << ", gate recoveries " << c.gate_recoveries << "; ms/frame track " << (n ? trackMs / n : 0) << " est " << (n ? estMs / n : 0);
    if (ec.dvl_enabled)
      std::cout << "; dvl_received " << c.dvl_received << " dvl_invalid " << c.dvl_invalid << " dvl_gated " << c.dvl_gated
                << " dvl_factors " << c.dvl_factors << " dvl_relocks " << c.dvl_relocks << " dvl_gaps " << c.dvl_gaps
                << " dvl_longest_gap_s " << c.dvl_longest_gap_s << " dvl_recoveries_kept " << c.dvl_recoveries_kept;
    if (!blackouts.empty()) std::cout << "; blackout_frames " << blackoutFrames;
    std::cout << " -> " << a.out << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "run_bag: " << e.what() << std::endl;
    return 1;
  }
}
