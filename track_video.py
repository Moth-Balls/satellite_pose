import argparse

import cv2
import numpy as np
from ultralytics import YOLO

video_path = "roe2_synthetic.mp4"
weights_path = "models/yolo26s_pose_dark.pt"
kps_path = "kps.npy" # keypoints file. Model is trained with this as well so don't modify
output_path = "output_vids/output.mp4"

# camera data (obtained from SPEED dataset)
fx = 3020.006
fy = 3020.006
cx = 960.0
cy = 600.0
camera_width = 1920

frame_dt = None
device = "cpu" # "mps" on a Mac, 0 for an NVIDIA GPU, "cpu" otherwise
imgsz = 960
brightness = 0.9
kp_conf = 0.75 # keypoint confidence
axis_len = 0.6
show = True
reinit_after = 10

# measurement pre-checks
min_inliers = 3 # of 8 keypoints within 8 px
max_rms_px = 30.0 # reprojection RMS
max_range_jump = 0.3
max_step_deg = 10.0
reinit_n = 3 # amount of frames before reinit

# kalman filter params
init_rot_deg = 3.0 # rotation, degrees
init_pos_frac = (0.02, 0.02, 0.06) # position x, y, z as a fraction of depth
init_spin = 1.0 # spin, rad/s 
init_vel = 0.5 # velocity, m/s

# Q matrix
q_rot = 1e-3 # rotation, rad
q_pos = 2e-3 # position, m
q_spin = 0.3 # spin, rad/s
q_vel = 0.05 # velocity, m/s

r_rot_deg = 3.0 # rotation per axis 
r_pos_frac = (0.003, 0.003, 0.03) # position x, y, z 
gate_chi2 = 30.0 # chi-square gate

# run YOLO on a frame
def detect_keypoints(model, frame):
  gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
  gray = cv2.convertScaleAbs(gray, alpha=brightness, beta=0) # brighten frames
  result = model.predict(gray, imgsz=imgsz, conf=0.05, device=device, verbose=False)[0]
  if len(result.boxes) == 0:
    return None
  best = int(result.boxes.conf.argmax())
  return result.keypoints.xy[best].cpu().numpy(), result.keypoints.conf[best].cpu().numpy()

# PnP with RANSAC
def solve_pose(kps3d, kps2d, K):
  if len(kps3d) < 4:
    return None
  obj = kps3d.astype(np.float64)
  img = kps2d.astype(np.float64)
  ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj, img, K, None, flags=cv2.SOLVEPNP_AP3P,
                                               reprojectionError=8.0, iterationsCount=500)
  if not ok or inliers is None or len(inliers) < 4:
    return None
  inliers = inliers.ravel()
  rvec, tvec = cv2.solvePnPRefineLM(obj[inliers], img[inliers], K, None, rvec, tvec)
  R = cv2.Rodrigues(rvec)[0]
  t = tvec.ravel()
  return R, t

def rot_vec_to_R(w):
  w_array = np.asarray(w, float).reshape(3, 1)
  return cv2.Rodrigues(w_array)[0]

def R_to_rotvec(R):
  return cv2.Rodrigues(R)[0].ravel()

# rotation matrix -> quaternion [w, x, y, z]
def R_to_quat(R):
  rv = R_to_rotvec(R)
  angle = np.linalg.norm(rv)
  if angle < 1e-12:
    return np.array([1.0, 0.0, 0.0, 0.0])
  q_w = np.cos(angle / 2)
  q_xyz = np.sin(angle / 2) * rv / angle
  return np.concatenate([[q_w], q_xyz])

# generic extended kalman filter: covariance bookkeeping and the gated update
class KalmanFilter:
  def __init__(self, P, gate):
    self.P = P # state covariance
    self.gate = gate # chi-square gate on the innovation
    self.nis = None # normalized innovation squared of the last accepted update

  # move the state forward by dt seconds
  def predict(self, dt):
    F = self.transition(dt)
    Q = self.process_noise(dt)
    self.P = F @ self.P @ F.T + Q
    self.propagate(dt)

  # correct the state with a measurement; returns True if it passed the gate
  def update(self, *meas):
    error = self.innovation(*meas)
    H = self.measurement_matrix()
    noise = self.measurement_noise()
    S = H @ self.P @ H.T + noise
    gate_stat = error @ np.linalg.solve(S, error)
    if gate_stat > self.gate:
      return False
    gain = self.P @ H.T @ np.linalg.inv(S)
    d = gain @ error
    self.correct(d)
    P_new = (np.eye(len(self.P)) - gain @ H) @ self.P
    self.P = 0.5 * (P_new + P_new.T)
    self.nis = float(gate_stat)
    return True

# satellite pose filter: rotation R, position t, spin w, velocity v
class PoseFilter(KalmanFilter):
  def __init__(self, R, t):
    self.R = R.copy()
    self.t = t.copy()
    self.w = np.zeros(3)
    self.v = np.zeros(3)
    depth = self.t[2]
    rot_cov = np.radians(init_rot_deg) ** 2
    pos_cov = (np.array(init_pos_frac) * depth) ** 2
    w_cov = init_spin ** 2
    v_cov = init_vel ** 2
    P = np.diag(np.concatenate([np.repeat(rot_cov, 3), pos_cov, np.repeat(w_cov, 3), np.repeat(v_cov, 3)]))
    super().__init__(P, gate_chi2)

  # F for the error state (rotation, position, spin, velocity)
  def transition(self, dt):
    dR = rot_vec_to_R(self.w * dt)
    F = np.eye(12)
    F[0:3, 0:3] = dR.T
    F[0:3, 6:9] = dt * np.eye(3)
    F[3:6, 9:12] = dt * np.eye(3)
    return F

  def process_noise(self, dt):
    return np.diag(np.repeat([q_rot, q_pos, q_spin, q_vel], 3) ** 2 * dt)

  def propagate(self, dt):
    self.R = self.R @ rot_vec_to_R(self.w * dt)
    self.t = self.t + self.v * dt

  # rotation error as a rotation vector, then position error
  def innovation(self, R_meas, t_meas):
    return np.concatenate([R_to_rotvec(self.R.T @ R_meas), t_meas - self.t])

  # pose (rotation and position) is measured directly
  def measurement_matrix(self):
    return np.eye(6, 12)

  # position noise scales with the current depth
  def measurement_noise(self):
    depth = max(self.t[2], 0.1)
    rot_noise = np.radians(r_rot_deg) ** 2
    pos_noise = (np.array(r_pos_frac) * depth) ** 2
    return np.diag(np.concatenate([np.repeat(rot_noise, 3), pos_noise]))

  def correct(self, d):
    self.R = self.R @ rot_vec_to_R(d[0:3])
    self.t = self.t + d[3:6]
    self.w = self.w + d[6:9]
    self.v = self.v + d[9:12]

# angle of the rotation between two rotation matrices
def angle_deg(R_a, R_b):
  c = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
  return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

def project(kps3d, R, t, K):
  pc = kps3d @ R.T + t
  uv = pc @ K.T
  return uv[:, :2] / uv[:, 2:3]

# consecutive poses agree with each other
def consistent(buf):
  for (Ra, ta), (Rb, tb) in zip(buf, buf[1:]):
    if angle_deg(Ra, Rb) > max_step_deg:
      return False
    if abs(np.linalg.norm(tb) - np.linalg.norm(ta)) > 0.1 * np.linalg.norm(ta):
      return False
  return True

# start the filter on the newest pose with the spin rate from the last two poses
def init_with_spin(buf, dt):
  (R0, t0), (R1, t1) = buf[-2], buf[-1]
  best = None
  for w in (R_to_rotvec(R1 @ R0.T) / dt,     # camera-frame spin
            R_to_rotvec(R0.T @ R1) / dt):    # body-frame spin
    f = PoseFilter(R0, t0)
    f.w = w
    f.predict(dt)
    err = angle_deg(f.R, R1)
    if best is None or err < best[0]:
      best = (err, w)
  f = PoseFilter(R1, t1)
  f.w = best[1]
  return f

# draw keypoints, the pose axes, and the pose values on a copy of the frame
def draw(frame, K, det, used, filt):
  img = frame.copy()
  scale = max(img.shape[1] / 1280.0, 0.6)
  if det is not None:
    for (x, y), ok in zip(det[0], used):
      color = (0, 255, 0) if ok else (0, 0, 255)
      cv2.drawMarker(img, (int(x), int(y)), color, cv2.MARKER_TILTED_CROSS, int(10 * scale), 1)
  if filt is None:
    lines = ["no pose yet"]
  else:
    rvec = cv2.Rodrigues(filt.R)[0]
    cv2.drawFrameAxes(img, K, None, rvec, filt.t.reshape(3, 1), axis_len, int(3 * scale))
    q = R_to_quat(filt.R)
    lines = [f"t = [{filt.t[0]:6.2f} {filt.t[1]:6.2f} {filt.t[2]:6.2f}] m",
             f"q = [{q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f} {q[3]:6.3f}]",
             f"spin {np.degrees(np.linalg.norm(filt.w)):6.1f} deg/s"]
  for i, text in enumerate(lines):
    pos = (12, int(28 * scale) * (i + 1))
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), 1)
  return img

# command-line options
def parse_args():
  parser = argparse.ArgumentParser(
    description="Track a satellite's pose in a video (YOLO keypoints, PnP, Kalman filter).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--video", default=video_path, help="input video")
  parser.add_argument("--weights", default=weights_path, help="YOLO pose weights")
  parser.add_argument("--kps", default=kps_path, help="3D keypoints file (.npy)")
  parser.add_argument("--output", default=output_path, help="annotated output video")
  parser.add_argument("--no-save", action="store_true", help="do not write the output video")
  parser.add_argument("--device", default=str(device), help='"mps" on a Mac, 0 for an NVIDIA GPU, "cpu" otherwise')
  parser.add_argument("--imgsz", type=int, default=imgsz, help="YOLO input size")
  parser.add_argument("--brightness", type=float, default=brightness, help="frame brightness scale before detection")
  parser.add_argument("--kp-conf", type=float, default=kp_conf, help="minimum keypoint confidence")
  parser.add_argument("--no-show", action="store_true", help="do not open the display window")
  return parser.parse_args()

if __name__ == "__main__":
  # the functions above read these settings as module globals, so overwrite them here
  args = parse_args()
  video_path = args.video
  weights_path = args.weights
  kps_path = args.kps
  output_path = None if args.no_save else args.output
  device = int(args.device) if args.device.isdigit() else args.device
  imgsz = args.imgsz
  brightness = args.brightness
  kp_conf = args.kp_conf
  show = show and not args.no_show

  model = YOLO(weights_path)
  kps3d = np.load(kps_path)

  cap = cv2.VideoCapture(video_path)
  assert cap.isOpened(), f"cannot open video {video_path}"
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  dt = frame_dt or 1 / fps

  s = width / camera_width
  K = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1.0]])

  writer = None
  if output_path:
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

  filt = None
  rejects = 0
  buf = [] # recent consecutive measurements that passed the pre-checks
  nis_values = [] # innovation statistic of every accepted update
  frame_idx = 0
  while True:
    ok, frame = cap.read()
    if not ok:
      break

    # measure the pose in this frame
    det = detect_keypoints(model, frame)
    used = []
    pose = None
    n_inl, rms = 0, np.inf
    if det is not None:
      used = det[1] >= kp_conf
      kps_subset = kps3d[used]
      det_subset = det[0][used]
      pose = solve_pose(kps_subset, det_subset, K)
    if pose is not None:
      reproj = np.linalg.norm(det_subset - project(kps_subset, pose[0], pose[1], K), axis=1)
      n_inl = int((reproj <= 8.0).sum())
      rms = float(np.sqrt(np.mean(reproj ** 2)))

    # feed it to the filter
    if filt is not None:
      filt.predict(dt)

    good = pose is not None and n_inl >= min_inliers and rms <= max_rms_px
    buf = (buf + [pose])[-reinit_n:] if good else []

    # a pose that failed the pre-checks is skipped and the filter coasts
    if good and filt is None:
      if len(buf) >= 2 and consistent(buf):
        filt = init_with_spin(buf, dt)
    elif good:
      r_pred, r_meas = np.linalg.norm(filt.t), np.linalg.norm(pose[1])
      accepted = False
      if abs(r_meas - r_pred) <= max_range_jump * r_pred:
        accepted = filt.update(pose[0], pose[1])
      if accepted:
        nis_values.append(filt.nis)
        rejects = 0
      else:
        rejects += 1
        if rejects >= reinit_after and len(buf) == reinit_n and consistent(buf):
          filt = init_with_spin(buf, dt)
          rejects = 0

    # print the pose
    if filt is None:
      print(f"frame {frame_idx:5d}  no pose")
    else:
      q = R_to_quat(filt.R)
      print(f"frame {frame_idx:5d}  t=[{filt.t[0]:5.2f} {filt.t[1]:5.2f} {filt.t[2]:5.2f}] m  "
            f"q=[{q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f} {q[3]:6.3f}]  "
            f"spin {np.degrees(np.linalg.norm(filt.w)):5.1f} deg/s")

    # draw, save and show
    out = draw(frame, K, det, used, filt)
    if writer is not None:
      writer.write(out)
    if show:
      cv2.imshow("satellite tracking", cv2.resize(out, None, fx=0.6, fy=0.6))
      key = cv2.waitKey(1) & 0xFF
      if key == ord(" "):  # pause until space or q
        key = cv2.waitKey(0) & 0xFF
      if key == ord("q"):
        break
    frame_idx += 1

  cap.release()
  if writer is not None:
    writer.release()
    print(f"annotated video: {output_path}")
  cv2.destroyAllWindows()
  if nis_values:
    print(f"innovation statistic over {len(nis_values)} accepted updates: mean {np.mean(nis_values):.1f}")