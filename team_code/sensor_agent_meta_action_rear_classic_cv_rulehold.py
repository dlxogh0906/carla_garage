"""
sensor_agent_meta_action_rear_classic_cv.py — TF++ + Meta-action VLA + Classic CV (전방 + 후방)

sensor_agent_meta_action_3rd_try 방법론 (TTC=3.0s, 20step) +
ClassicCVEnhancer 보정 후 전방·후방 카메라를 모두 VLM 입력으로 사용.

변경 내용 (classic_cv 전방 전용 대비):
  - rgb_rear 센서 추가 (yaw=180°)
  - 후방 카메라도 ClassicCVEnhancer 보정 적용 (BGR→BGR)
  - MetaActionRearVLAPlanner: 전방+후방 이미지를 함께 VLM에 전달
  - 대시보드: meta_action_rear_dashboard (전방+후방 inset 레이아웃)
  - TTC threshold / 재추론 간격은 3rd_try 동일 (3.0s / 20step)
"""

import os
from copy import deepcopy

import cv2
import carla
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np
import math

from leaderboard.autoagents import autonomous_agent
from model import LidarCenterNet
from config import GlobalConfig
from data import CARLA_Data
from nav_planner import RoutePlanner
from nav_planner import extrapolate_waypoint_route

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from scipy.optimize import fsolve

from scenario_logger import ScenarioLogger
import transfuser_utils as t_u

import sys as _sys
_b2d_enh_path = os.path.join(os.path.dirname(__file__), '..', 'Bench2Drive')
if _b2d_enh_path not in _sys.path:
    _sys.path.insert(0, _b2d_enh_path)
from image_enhancement_module.classic_cv_enhancer import ClassicCVEnhancer

import pathlib
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import ujson  # Like json but faster
import gzip

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


# Leaderboard function that selects the class used as agent.
def get_entry_point():
  return 'SensorAgent'


def strtobool(v):
  return str(v).lower() in ('yes', 'y', 'true', 't', '1', 'True')


class SensorAgent(autonomous_agent.AutonomousAgent):
  """
    Main class that runs the agents with the run_step function
    """

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    """Sets up the agent. route_index is for logging purposes"""
    torch.cuda.empty_cache()
    self.IS_BENCH2DRIVE = strtobool(os.environ.get('IS_BENCH2DRIVE', 'False'))
    print('IS_BENCH2DRIVE: ', self.IS_BENCH2DRIVE)
    self.track = autonomous_agent.Track.MAP if os.environ.get(
        'CHALLENGE_TRACK_CODENAME') == 'MAP' else autonomous_agent.Track.SENSORS
    if self.IS_BENCH2DRIVE:
      self.config_path = path_to_conf_file.split('+')[0]
    else:
      self.config_path = path_to_conf_file

    self.step = -1
    self.initialized = False
    self.device = torch.device('cuda:0')

    # Load the config saved during training
    with open(os.path.join(self.config_path, 'config.json'), 'rt', encoding='utf-8') as f:
      json_config = f.read()

    loaded_config = jsonpickle.decode(json_config)

    # Generate new config for the case that it has new variables.
    self.config = GlobalConfig()
    # Overwrite all properties that were set in the saved config.
    self.config.__dict__.update(loaded_config.__dict__)

    # For models supporting different output modalities we select which one to use here.
    # 0: Waypoints
    # 1: Path + Target Speed

    self.uncertainty_weight = int(os.environ.get('UNCERTAINTY_WEIGHT', 1))
    print('Uncertainty weighting?: ', self.uncertainty_weight)
    self.tuned_aim_distance = int(os.environ.get('TUNED_AIM_DISTANCE', 0))
    print('TUNED_AIM_DISTANCE for wp rep?: ', self.tuned_aim_distance)
    direct = os.environ.get('DIRECT', 1)
    self.config.inference_direct_controller = int(direct)
    print('Direct control prediction?: ', direct)
    self.stop_after_meter = int(os.environ.get('STOP_AFTER_METER', -1))
    print('STOP_AFTER_METER: ', self.stop_after_meter)

    # If set to true, will generate visualizations at SAVE_PATH
    self.config.debug = int(os.environ.get('DEBUG_CHALLENGE', 0)) == 1

    self.compile = int(os.environ.get('COMPILE', 0)) == 1

    self.config.brake_uncertainty_threshold = float(
        os.environ.get('UNCERTAINTY_THRESHOLD', self.config.brake_uncertainty_threshold))
    print('Brake uncertainty threshold: ', self.config.brake_uncertainty_threshold)

    # Classification networks are known to be overconfident which leads to them braking a bit too late in our case.
    # Reducing the driving speed slightly counteracts that.
    if int(os.environ.get('SLOWER', 0)):
      print(f'Reduce target speeds during evaluation by factor {self.config.slower_factor}.')
      self.inference_target_speeds = [self.config.slower_factor * speed for speed in self.config.target_speeds]
    else:
      print('No speed reduction during inference.')
      self.inference_target_speeds = self.config.target_speeds

    if self.config.tp_attention:
      self.tp_attention_buffer = []

    # Stop signs can be occluded with our camera setup. This buffer remembers them until cleared.
    # Very useful on the LAV benchmark
    self.stop_sign_controller = int(os.environ.get('STOP_CONTROL', 1))
    print('Use stop sign controller:', self.stop_sign_controller)
    if self.stop_sign_controller:
      # There can be max 1 stop sign affecting the ego
      self.stop_sign_buffer = deque(maxlen=1)
      self.clear_stop_sign = 0  # Counter if we recently cleared a stop sign

    # Load model files
    self.nets = []
    self.model_count = 0  # Counts how many models are in our ensemble
    for file in os.listdir(self.config_path):
      if file.endswith('.pth') and file.startswith('model'):
        self.model_count += 1
        print(os.path.join(self.config_path, file))
        net = LidarCenterNet(self.config)
        if self.config.sync_batch_norm:
          # Model was trained with Sync. Batch Norm.
          # Need to convert it otherwise parameters will load wrong.
          net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        state_dict = torch.load(os.path.join(self.config_path, file), map_location=self.device)
        net.load_state_dict(state_dict, strict=True)
        net.cuda(device=self.device)
        net.eval()

        if self.config.compile or self.compile:
          net = torch.compile(net, mode=self.config.compile_mode)

        self.nets.append(net)

    self.stuck_detector = 0
    self.force_move = 0

    self.bb_buffer = deque(maxlen=1)
    self.commands = deque(maxlen=2)
    self.commands.append(4)
    self.commands.append(4)
    self.target_point_prev = [1e5, 1e5, 1e5]

    # Filtering
    self.ego_model = EgoModel(dt=self.config.carla_frame_rate)
    self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
    # Still uses the leaderboard 1.0 bicycle model for the unscented kalman filter
    self.ukf = UKF(dim_x=4,
                   dim_z=4,
                   fx=bicycle_model_forward,
                   hx=measurement_function_hx,
                   dt=self.config.carla_frame_rate,
                   points=self.points,
                   x_mean_fn=state_mean,
                   z_mean_fn=measurement_mean,
                   residual_x=residual_state_x,
                   residual_z=residual_measurement_h)

    # State noise, same as measurement because we
    # initialize with the first measurement later
    self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
    # Measurement noise
    self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
    self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
    # Used to set the filter state equal the first measurement
    self.filter_initialized = False
    # Stores the last filtered positions of the ego vehicle. Need at least 2 for LiDAR 10 Hz realignment
    self.state_log = deque(maxlen=max((self.config.lidar_seq_len * self.config.data_save_freq), 2))

    #Temporal LiDAR
    self.lidar_buffer = deque(maxlen=self.config.lidar_seq_len * self.config.data_save_freq)

    self.lidar_last = None

    # Forced stopping
    if self.stop_after_meter > 0:
      self.meters_travelled = 0

    self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)

    # Classic CV image enhancement (항상 활성화, checkpoint 불필요)
    use_classic_cv = int(os.environ.get('USE_CLASSIC_CV', 1))
    if use_classic_cv:
      self.enhancer = ClassicCVEnhancer()
      print('[ClassicCV] Image enhancement enabled (front + rear).')
    else:
      self.enhancer = None

    # Dedicated visualization output path — per-route subfolder to avoid overwrite
    enh_vis = os.environ.get('ENH_VIS_PATH', None)
    enh_vis_max_routes = int(os.environ.get('ENH_VIS_MAX_ROUTES', 0))
    if enh_vis and self.enhancer is not None:
      enh_vis_dir = pathlib.Path(enh_vis)
      enh_vis_dir.mkdir(parents=True, exist_ok=True)
      enable_vis = True
      if enh_vis_max_routes > 0:
        counter_file = enh_vis_dir / '.route_counter'
        route_count = int(counter_file.read_text().strip()) + 1 if counter_file.exists() else 1
        counter_file.write_text(str(route_count))
        enable_vis = route_count <= enh_vis_max_routes
        if not enable_vis:
          print(f'[ClassicCV] Route {route_count} exceeds ENH_VIS_MAX_ROUTES={enh_vis_max_routes}, skipping visualization.')
      if enable_vis:
        save_name = path_to_conf_file.split('+')[-1] if '+' in path_to_conf_file else (str(route_index) if route_index else 'route')
        self.enh_vis_path = enh_vis_dir / save_name
        self.enh_vis_path.mkdir(parents=True, exist_ok=True)
        print(f'[ClassicCV] Saving visualizations to: {self.enh_vis_path}')
      else:
        self.enh_vis_path = None
    else:
      self.enh_vis_path = None

    # ----------------------------------------------------------------
    # Meta-action VLA: Qwen3-VL — TTC 기반 8-class 메타-액션 (전방+후방)
    # ----------------------------------------------------------------
    import sys  # pylint: disable=import-outside-toplevel
    _b2d_path = os.path.join(os.path.dirname(__file__), '..', 'Bench2Drive')
    if _b2d_path not in sys.path:
      sys.path.insert(0, _b2d_path)

    from meta_action_rear_vla_rulehold import MetaActionRearVLAPlanner  # pylint: disable=import-outside-toplevel
    meta_every_n = int(os.environ.get('META_EVERY_N_STEPS', 20))
    meta_ttc_threshold = float(os.environ.get('META_TTC_THRESHOLD', 3.0))
    meta_model = os.environ.get('META_MODEL', 'Qwen/Qwen3-VL-8B-Instruct')
    self._meta_route_name = (
        path_to_conf_file.split('+')[-1]
        if '+' in path_to_conf_file
        else (str(route_index) if route_index else 'route')
    )
    self._meta_dash_every_n = max(1, int(os.environ.get('META_DASHBOARD_EVERY_N_STEPS', 3)))
    self.meta_action = MetaActionRearVLAPlanner(
        self.device,
        ttc_threshold=meta_ttc_threshold,
        inference_every_n=meta_every_n,
        model_name=meta_model,
    )
    self._meta_rgb_np = None       # 전방 카메라 full-res RGB (ClassicCV 보정 후, VLA 입력)
    self._meta_rgb_rear_np = None  # 후방 카메라 full-res RGB (ClassicCV 보정 후, VLA 입력)
    self._dashboard_rgb_np = None       # 전방 카메라 full-res RGB (원본, dashboard 전용)
    self._dashboard_rgb_rear_np = None  # 후방 카메라 full-res RGB (원본, dashboard 전용)
    self._meta_rule_every_n = max(1, int(os.environ.get('META_RULE_EVERY_N_STEPS', 5)))
    self._meta_rule_confidence_threshold = float(os.environ.get('META_RULE_CONFIDENCE_THRESH', 0.75))
    self._meta_rule_result_max_age = max(1, int(os.environ.get('META_RULE_MAX_AGE_STEPS', 20)))
    self._meta_rule_enable_stop_sign = int(os.environ.get('META_RULE_ENABLE_STOP_SIGN', 0)) == 1
    self._meta_rule_hold_enabled = int(os.environ.get('META_RULE_HOLD_ENABLED', 1)) == 1
    self._meta_rule_hold_poll_steps = max(1, int(os.environ.get('META_RULE_HOLD_POLL_STEPS', 5)))
    self._meta_rule_hold_green_confirmations_required = max(
        1, int(os.environ.get('META_RULE_HOLD_GREEN_CONFIRMATIONS', 2)))
    self._meta_rule_hold_safety_steps = max(0, int(os.environ.get('META_RULE_HOLD_SAFETY_STEPS', 150)))
    self._meta_rule_hold_post_release_cooldown = max(
        0, int(os.environ.get('META_RULE_HOLD_POST_RELEASE_COOLDOWN', 50)))
    self._meta_stop_sign_release_steps = max(0, int(os.environ.get('META_STOP_SIGN_RELEASE_STEPS', 60)))
    self._meta_tl_prestop_enabled = int(os.environ.get('META_TL_PRESTOP_ENABLED', 1)) == 1
    self._meta_tl_prestop_scale = float(os.environ.get('META_TL_PRESTOP_SCALE', 0.45))
    self._meta_tl_prestop_distance = float(os.environ.get('META_TL_PRESTOP_DISTANCE', 45.0))
    self._meta_rule_hold_active = False
    self._meta_rule_hold_type = 'none'
    self._meta_rule_hold_since_step = -1
    self._meta_rule_hold_green_confirmations = 0
    self._meta_rule_hold_last_green_key = None
    self._meta_rule_hold_no_stop_votes = 0
    self._meta_rule_hold_last_no_stop_key = None
    self._meta_rule_hold_reason = ''
    self._meta_rule_hold_post_release_step = -1
    self._meta_stop_sign_release_until_step = -1
    self._meta_rule_active = False
    self._meta_rule_scale = 1.0
    self._meta_tl_prestop_active = False
    self._meta_rule_summary = self._summarize_rule_bboxes([])
    self._meta_rule_result = {
        'rule_intervene': False,
        'rule_type': 'none',
        'traffic_light_state': 'unknown',
        'stop_sign_visible': False,
        'relevant_to_ego': False,
        'rule_confidence': 0.0,
        'rule_speed_scale': 1.0,
        'rule_reason': '',
        'request_step': None,
        'prompt_mode': 'traffic_rule',
    }
    self._dashboard_vis = None
    try:
      from meta_action_rear_dashboard import render_meta_action_rear_dashboard  # pylint: disable=import-outside-toplevel
      self._dashboard_vis = render_meta_action_rear_dashboard
      print('[MetaDash] Rear dashboard visualization enabled.')
    except Exception as e:  # pylint: disable=broad-except
      print(f'[MetaDash] Rear dashboard import failed: {e}')

    meta_dash_root = os.environ.get('META_DASHBOARD_PATH', None)
    if meta_dash_root and self._dashboard_vis is not None:
      save_name = path_to_conf_file.split('+')[-1] if '+' in path_to_conf_file else 'route'
      self._meta_dash_path = pathlib.Path(meta_dash_root) / save_name
      self._meta_dash_path.mkdir(parents=True, exist_ok=True)
      print(f'[MetaDash] Saving dashboard to: {self._meta_dash_path}')
      print(f'[MetaDash] Capturing frame and explanation every {self._meta_dash_every_n} steps.')
    else:
      self._meta_dash_path = None

    self.orig_vis_path = None

    # Path to where visualizations and other debug output gets stored.
    # Leaderboard may not pass route_index for custom route configs, so derive
    # a stable route folder from path_to_conf_file as a fallback. Without this,
    # Qwen runtime summaries stay empty because qwen_intervention.jsonl is never
    # opened.
    self.save_path = os.environ.get('SAVE_PATH', None)
    if route_index is None:
      route_index = path_to_conf_file.split('+')[-1] if '+' in path_to_conf_file else None

    # Logger that generates logs used for infraction replay in the results_parser.
    if self.save_path is not None and route_index is not None:
      self.save_path = pathlib.Path(self.save_path) / route_index
      pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

      self.lon_logger = ScenarioLogger(
          save_path=self.save_path,
          route_index=route_index,
          logging_freq=self.config.logging_freq,
          log_only=True,
          route_only=False,  # with vehicles
          roi=self.config.logger_region_of_interest,
      )
    else:
      self.save_path = None

    self.metric_info = {}
    self._qwen_log_f = None
    self._last_logged_qwen_request_id = None
    if self.save_path is not None:
      self._qwen_log_f = open(self.save_path / 'qwen_intervention.jsonl', 'a', encoding='utf-8')

  def _init(self):
    try:
      locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
      lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
      earth_radius_equa = 6378137.0  # Constant from CARLA leaderboard GPS simulation

      def equations(variables):
        x, y = variables
        eq1 = (lon * math.cos(x * math.pi / 180.0) - (locx * x * 180.0) / (math.pi * earth_radius_equa) -
               math.cos(x * math.pi / 180.0) * y)
        eq2 = (math.log(math.tan(
            (lat + 90.0) * math.pi / 360.0)) * earth_radius_equa * math.cos(x * math.pi / 180.0) + locy -
               math.cos(x * math.pi / 180.0) * earth_radius_equa * math.log(math.tan((90.0 + x) * math.pi / 360.0)))
        return [eq1, eq2]

      initial_guess = [0.0, 0.0]
      solution = fsolve(equations, initial_guess)
      self.lat_ref, self.lon_ref = solution[0], solution[1]
    except Exception as e:
      print(e, flush=True)
      self.lat_ref, self.lon_ref = 0.0, 0.0

    # During setup() not everything is available yet, so this _init is a second setup in run_step()
    if self.save_path is not None:
      # Privileged map access for logging and visualizations. Turned off during normal evaluation.
      from srunner.scenariomanager.carla_data_provider import CarlaDataProvider  # pylint: disable=locally-disabled, import-outside-toplevel
      from nav_planner import interpolate_trajectory  # pylint: disable=locally-disabled, import-outside-toplevel
      self.world_map = CarlaDataProvider.get_map()
      trajectory = [item[0].location for item in self._global_plan_world_coord]
      self.dense_route, _ = interpolate_trajectory(self.world_map, trajectory)  # privileged

      self._waypoint_planner = RoutePlanner(self.config.log_route_planner_min_distance,
                                            self.config.route_planner_max_distance, self.lat_ref, self.lon_ref)
      self._waypoint_planner.set_route(self.dense_route, True)

      vehicle = CarlaDataProvider.get_hero_actor()
      self.lon_logger.ego_vehicle = vehicle
      self.lon_logger.world = vehicle.get_world()

      self.nets[0].init_visualization()

    self._route_planner = RoutePlanner(self.config.route_planner_min_distance, self.config.route_planner_max_distance,
                                       self.lat_ref, self.lon_ref)
    self._route_planner.set_route(self._global_plan, True)

    self.initialized = True

  def sensors(self):
    sensors = [{
        'type': 'sensor.camera.rgb',
        'x': self.config.camera_pos[0],
        'y': self.config.camera_pos[1],
        'z': self.config.camera_pos[2],
        'roll': self.config.camera_rot_0[0],
        'pitch': self.config.camera_rot_0[1],
        'yaw': self.config.camera_rot_0[2],
        'width': self.config.camera_width,
        'height': self.config.camera_height,
        'fov': self.config.camera_fov,
        'id': 'rgb_front'
    }, {
        'type': 'sensor.camera.rgb',
        'x': self.config.camera_pos[0],
        'y': self.config.camera_pos[1],
        'z': self.config.camera_pos[2],
        'roll': self.config.camera_rot_0[0],
        'pitch': self.config.camera_rot_0[1],
        'yaw': 180.0,  # 후방 카메라
        'width': self.config.camera_width,
        'height': self.config.camera_height,
        'fov': self.config.camera_fov,
        'id': 'rgb_rear'
    }, {
        'type': 'sensor.other.imu',
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'sensor_tick': self.config.carla_frame_rate,
        'id': 'imu'
    }, {
        'type': 'sensor.other.gnss',
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'sensor_tick': 0.01,
        'id': 'gps'
    }, {
        'type': 'sensor.speedometer',
        'reading_frequency': self.config.carla_fps,
        'id': 'speed'
    }]
    # Don't set up LiDAR for camera only approaches
    if self.config.backbone not in ('aim'):
      sensors.append({
          'type': 'sensor.lidar.ray_cast',
          'x': self.config.lidar_pos[0],
          'y': self.config.lidar_pos[1],
          'z': self.config.lidar_pos[2],
          'roll': self.config.lidar_rot[0],
          'pitch': self.config.lidar_rot[1],
          'yaw': self.config.lidar_rot[2],
          'id': 'lidar'
      })

    return sensors

  @torch.inference_mode()  # Turns off gradient computation
  def tick(self, input_data):
    """Pre-processes sensor data and runs the Unscented Kalman Filter"""
    rgb = []
    for camera_pos in ['front']:
      rgb_cam = 'rgb_' + camera_pos
      camera = input_data[rgb_cam][1][:, :, :3]

      # Also add jpg artifacts at test time, because the training data was saved as jpg.
      _, compressed_image_i = cv2.imencode('.jpg', camera)
      camera = cv2.imdecode(compressed_image_i, cv2.IMREAD_UNCHANGED)
      dashboard_camera = camera.copy()
      self._dashboard_rgb_np = cv2.cvtColor(dashboard_camera, cv2.COLOR_BGR2RGB)

      # ClassicCV enhancement on BGR before color conversion
      if self.enhancer is not None:
        original_camera = camera.copy()
        camera = self.enhancer.enhance(camera)  # BGR→BGR
        if self.enh_vis_path is not None and self.step % 5 == 0:
          side_by_side = np.concatenate([original_camera, camera], axis=1)
          cv2.imwrite(str(self.enh_vis_path / f'step_{self.step:05d}.jpg'), side_by_side)

      rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)

      # Store ClassicCV-enhanced full-res RGB for Meta-action VLA (전방)
      self._meta_rgb_np = rgb_pos.copy()

      rgb_pos = t_u.crop_array(self.config, rgb_pos)

      # Switch to pytorch channel first order
      rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
      rgb.append(rgb_pos)
    rgb = np.concatenate(rgb, axis=1)
    rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)

    # 후방 카메라 처리 — ClassicCV 보정 후 VLM 전달
    if 'rgb_rear' in input_data:
      rear_cam = input_data['rgb_rear'][1][:, :, :3]
      _, compressed_rear = cv2.imencode('.jpg', rear_cam)
      rear_cam = cv2.imdecode(compressed_rear, cv2.IMREAD_UNCHANGED)
      dashboard_rear = rear_cam.copy()
      self._dashboard_rgb_rear_np = cv2.cvtColor(dashboard_rear, cv2.COLOR_BGR2RGB)
      if self.enhancer is not None:
        rear_cam = self.enhancer.enhance(rear_cam)  # BGR→BGR
      self._meta_rgb_rear_np = cv2.cvtColor(rear_cam, cv2.COLOR_BGR2RGB)

    gps_pos = self._route_planner.convert_gps_to_carla(input_data['gps'][1])
    speed = input_data['speed'][1]['speed']
    compass = t_u.preprocess_compass(input_data['imu'][1][-1])

    result = {
        'rgb': rgb,
        'compass': compass,
    }

    if self.config.backbone not in ('aim'):
      result['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

    if not self.filter_initialized:
      # apply ukf only to x and y coordinates, append z coordinate afterwards
      self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
      self.filter_initialized = True

    self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
    self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
    filtered_state = self.ukf.x
    self.state_log.append(filtered_state)
    result['gps'] = filtered_state[0:2]

    waypoint_route = self._route_planner.run_step(np.append(filtered_state[0:2], gps_pos[2]))

    if len(waypoint_route) > 2:
      target_point, far_command = waypoint_route[1]
      target_point_next, _ = waypoint_route[2]
    elif len(waypoint_route) > 1:
      target_point, far_command = waypoint_route[1]
      target_point_next = target_point
    else:
      target_point, far_command = waypoint_route[0]
      target_point_next = target_point

    if (target_point != self.target_point_prev).all():
      self.target_point_prev = target_point
      self.commands.append(far_command.value)

    one_hot_command = t_u.command_to_one_hot(self.commands[-2])
    result['command'] = torch.from_numpy(one_hot_command[np.newaxis]).to(self.device, dtype=torch.float32)

    ego_target_point = t_u.inverse_conversion_2d(target_point[:2], result['gps'], result['compass'])  # original

    ego_target_point = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)

    result['target_point'] = ego_target_point

    if self.config.two_tp_input:
      ego_target_point_next = t_u.inverse_conversion_2d(target_point_next[:2], result['gps'], result['compass'])
      ego_target_point_next = torch.from_numpy(ego_target_point_next[np.newaxis]).to(self.device, dtype=torch.float32)
      result['target_point_next'] = ego_target_point_next

    result['speed'] = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)

    if self.save_path is not None:
      pass
      waypoint_route = self._waypoint_planner.run_step(np.append(result['gps'], gps_pos[2]))
      waypoint_route = extrapolate_waypoint_route(waypoint_route, self.config.route_points)
      route = np.array([[node[0][0], node[0][1]] for node in waypoint_route])[:self.config.route_points]
      self.lon_logger.log_step(route)

    return result

  @torch.inference_mode()  # Turns off gradient computation
  def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=locally-disabled, unused-argument
    self.step += 1

    if not self.initialized:
      self._init()
      control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
      self.control = control
      tick_data = self.tick(input_data)
      if self.config.backbone not in ('aim'):
        self.lidar_last = deepcopy(tick_data['lidar'])
      return control

    # Need to run this every step for GPS filtering
    tick_data = self.tick(input_data)

    lidar_indices = []
    for i in range(self.config.lidar_seq_len):
      lidar_indices.append(i * self.config.data_save_freq)

    #Current position of the car
    ego_x = self.state_log[-1][0]
    ego_y = self.state_log[-1][1]
    ego_theta = self.state_log[-1][2]

    ego_x_last = self.state_log[-2][0]
    ego_y_last = self.state_log[-2][1]
    ego_theta_last = self.state_log[-2][2]

    # We only get half a LiDAR at every time step. Aligns the last half into the current coordinate frame.
    if self.config.backbone not in ('aim'):
      lidar_last = self.align_lidar(self.lidar_last, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

    # Updates stop boxes by vehicle movement converting past predictions into the current frame.
    if self.stop_sign_controller:
      self.update_stop_box(self.stop_sign_buffer, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

    if self.config.backbone not in ('aim'):
      lidar_current = deepcopy(tick_data['lidar'])
      lidar_full = np.concatenate((lidar_current, lidar_last), axis=0)

      self.lidar_buffer.append(lidar_full)

    if self.config.backbone not in ('aim'):
      # We wait until we have sufficient LiDARs
      if len(self.lidar_buffer) < (self.config.lidar_seq_len * self.config.data_save_freq):
        self.lidar_last = deepcopy(tick_data['lidar'])
        tmp_control = carla.VehicleControl(0.0, 0.0, 1.0)
        self.control = tmp_control

        return tmp_control

    if self.config.backbone in ('aim'):  # Image only method
      # Dummy data
      lidar_bev = torch.zeros((1, 1 + int(self.config.use_ground_plane), self.config.lidar_resolution_height,
                               self.config.lidar_resolution_width)).to(self.device, dtype=torch.float32)
    else:
      # Voxelize LiDAR and stack temporal frames
      lidar_bev = []
      # prepare LiDAR input
      for i in lidar_indices:
        lidar_point_cloud = deepcopy(self.lidar_buffer[-(i + 1)])

        # For single frame there is no point in realignment. The state_log index will also differ.
        if self.config.realign_lidar and self.config.lidar_seq_len > 1:
          # Position of the car when the LiDAR was collected
          curr_x = self.state_log[i][0]
          curr_y = self.state_log[i][1]
          curr_theta = self.state_log[i][2]

          # Voxelize to BEV for NN to process
          lidar_point_cloud = self.align_lidar(lidar_point_cloud, curr_x, curr_y, curr_theta, ego_x, ego_y, ego_theta)

        lidar_histogram = self.data.lidar_to_histogram_features(lidar_point_cloud,
                                                                use_ground_plane=self.config.use_ground_plane)

        lidar_histogram = torch.from_numpy(lidar_histogram).unsqueeze(0).to(self.device, dtype=torch.float32)
        lidar_bev.append(lidar_histogram)

        lidar_bev = torch.cat(lidar_bev, dim=1)

    if self.config.backbone not in ('aim'):
      self.lidar_last = deepcopy(tick_data['lidar'])

    # prepare velocity input
    gt_velocity = tick_data['speed']
    velocity = gt_velocity.reshape(1, 1)  # used by transfuser

    compute_debug_output = self.config.debug and (self.save_path is not None)

    # new checkpoint lookahead: calculate which checkpoint to use for control
    speed = gt_velocity.item()

    if self.stop_after_meter > 0:
      dt = self.config.carla_frame_rate
      self.meters_travelled = self.meters_travelled + speed * dt

    # forward pass
    pred_wps = []
    pred_target_speeds = []
    pred_checkpoints = []
    bounding_boxes = []
    wp_selected = None
    for i in range(self.model_count):
      if self.config.backbone in ('transFuser', 'aim', 'bev_encoder'):
        pred_wp, \
        pred_target_speed, \
        pred_checkpoint, \
        pred_semantic, \
        pred_bev_semantic, \
        pred_depth, \
        pred_bb_features,\
        attention_weights,\
        pred_wp_1,\
        selected_path = self.nets[i].forward(
          rgb=tick_data['rgb'],
          lidar_bev=lidar_bev,
          target_point=tick_data['target_point'],
          target_point_next=tick_data['target_point_next'] if self.config.two_tp_input else None,
          ego_vel=velocity,
          command=tick_data['command'])
        # Only convert bounding boxes when they are used.
        if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ('aim') or
                                         self.stop_sign_controller):
          pred_bounding_box = self.nets[i].convert_features_to_bb_metric(pred_bb_features)
        else:
          pred_bounding_box = None
      else:
        raise ValueError('The chosen vision backbone does not exist. The options are: transFuser, aim, bev_encoder')

      if self.config.use_wp_gru:
        if self.config.multi_wp_output:
          wp_selected = 0
          if F.sigmoid(selected_path)[0].item() > 0.5:
            wp_selected = 1
            pred_wps.append(pred_wp_1)
          else:
            pred_wps.append(pred_wp)
        else:
          pred_wps.append(pred_wp)
      if self.config.use_controller_input_prediction:
        pred_target_speeds.append(F.softmax(pred_target_speed[0], dim=0))
        pred_checkpoints.append(pred_checkpoint[0])

      bounding_boxes.append(pred_bounding_box)

    # Average the predictions from ensembles
    if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ('aim') or
                                     self.stop_sign_controller):
      # We average bounding boxes by using non-maximum suppression on the set of all detected boxes.
      bbs_vehicle_coordinate_system = t_u.non_maximum_suppression(bounding_boxes, self.config.iou_treshold_nms)

      self.bb_buffer.append(bbs_vehicle_coordinate_system)
    else:
      bbs_vehicle_coordinate_system = None

    if self.stop_sign_controller:
      stop_for_stop_sign = self.stop_sign_controller_step(gt_velocity.item())

    if self.config.tp_attention:
      self.tp_attention_buffer.append(attention_weights[2])

    if self.config.use_wp_gru:
      self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)

    # calculate target speed scalar from model predictions
    if self.config.use_controller_input_prediction:
      pred_target_speed_ensemble = torch.stack(pred_target_speeds,
                                               dim=0).mean(dim=0)  # average across ensemble models' prediction

      if self.uncertainty_weight:
        uncertainty = pred_target_speed_ensemble.detach().cpu().numpy()
        if uncertainty[0] > self.config.brake_uncertainty_threshold:
          pred_target_speed_scalar = self.inference_target_speeds[0]
        else:
          pred_target_speed_scalar = sum(uncertainty * self.inference_target_speeds)
      else:
        pred_target_speed_index = torch.argmax(pred_target_speed_ensemble)
        pred_target_speed_scalar = self.inference_target_speeds[pred_target_speed_index]

    # ----------------------------------------------------------------
    # Meta-action VLA: Qwen3-VL — TTC 기반 8-class 메타-액션 (전방+후방)
    #
    # 1. bbs_vehicle_coordinate_system으로 TTC 계산
    # 2. TTC < ttc_threshold → VLM 비동기 트리거 (전방+후방 이미지 전달)
    # 3. TTC 위험 상황에서만 캐싱된 multiplier 적용
    # ----------------------------------------------------------------
    _ttc = self._compute_ttc(bbs_vehicle_coordinate_system, speed)
    _ttc_danger = _ttc < self.meta_action.ttc_threshold
    _speed_before_vlm = pred_target_speed_scalar
    self._meta_rule_summary = self._summarize_rule_bboxes(bbs_vehicle_coordinate_system)
    traffic_light_candidate = self._meta_rule_summary.get('bbox_traffic_light_count', 0) > 0
    stop_sign_candidate = (
        self._meta_rule_enable_stop_sign
        and self._meta_rule_summary.get('bbox_stop_sign_count', 0) > 0
    )
    rule_candidate = traffic_light_candidate or stop_sign_candidate
    rule_hold_probe = (
        self._meta_rule_hold_active
        and self.step % self._meta_rule_hold_poll_steps == 0
    )

    if self._meta_rgb_np is not None and (rule_candidate or rule_hold_probe):
      trigger = 'traffic_rule_hold' if rule_hold_probe else 'traffic_rule'
      self.meta_action.request_guidance(
          self._meta_rgb_np,
          self._meta_rgb_rear_np,
          self.step,
          prompt_mode='traffic_rule',
          trigger=trigger,
          ttc=_ttc,
          ego_speed=speed,
          tfpp_target_speed=pred_target_speed_scalar,
          rule_context=self._format_rule_context(self._meta_rule_summary),
      )
    elif _ttc_danger and self._meta_rgb_np is not None:
      self.meta_action.request_guidance(
          self._meta_rgb_np,
          self._meta_rgb_rear_np,
          self.step,
          prompt_mode='speed',
          trigger='ttc',
          ttc=_ttc,
          ego_speed=speed,
      )

    latest_result = self.meta_action.get_latest_result()
    if latest_result.get('prompt_mode') == 'traffic_rule':
      self._meta_rule_result = latest_result
    self._meta_rule_active = self._update_meta_rule_hold_state(speed)
    if not self._meta_rule_active:
      self._meta_rule_active = self._should_apply_meta_rule_result(speed)
    self._meta_rule_scale = (
        0.0
        if self._meta_rule_hold_active
        else float(self._meta_rule_result.get('rule_speed_scale', 1.0))
    )
    self._meta_tl_prestop_active = self._should_meta_tl_prestop(self._meta_rule_summary)

    if self.config.use_controller_input_prediction:
      _multiplier = 1.0
      if _ttc_danger:
        _multiplier = min(_multiplier, self.meta_action.get_speed_multiplier())
      if self._meta_tl_prestop_active:
        _multiplier = min(_multiplier, self._meta_tl_prestop_scale)
      if self._meta_rule_active:
        _multiplier = min(_multiplier, self._meta_rule_scale)
      pred_target_speed_scalar = pred_target_speed_scalar * _multiplier
    else:
      _multiplier = 1.0

    # Visualize the output of the last model
    if compute_debug_output:
      if self.config.use_controller_input_prediction:
        prob_target_speed = F.softmax(pred_target_speed, dim=1)
      else:
        prob_target_speed = pred_target_speed

      self.nets[0].visualize_model(
          self.save_path,
          self.step,
          tick_data['rgb'],
          lidar_bev,
          tick_data['target_point'],
          pred_wp,
          target_point_next=tick_data['target_point_next'] if self.config.two_tp_input else None,
          pred_semantic=pred_semantic,
          pred_bev_semantic=pred_bev_semantic,
          pred_depth=pred_depth,
          pred_checkpoint=pred_checkpoint,
          pred_speed=prob_target_speed,
          pred_target_speed_scalar=pred_target_speed_scalar,
          pred_bb=deepcopy(bbs_vehicle_coordinate_system) if bbs_vehicle_coordinate_system is not None else None,
          gt_speed=gt_velocity,
          gt_wp=pred_wp_1,
          wp_selected=wp_selected)

    if self.config.inference_direct_controller and self.config.use_controller_input_prediction:
      pred_checkpoints = torch.stack(pred_checkpoints, dim=0).mean(dim=0).detach().cpu().numpy()
      steer, throttle, brake = self.nets[0].control_pid_direct(pred_checkpoints, pred_target_speed_scalar, gt_velocity)

      # ── 대시보드 시각화 (설정된 간격, META_DASHBOARD_PATH가 있을 때만) ────
      # 실제 설명 모드에서는 각 저장 프레임마다 Qwen3-VL 캡션을 생성한다.
      if (
          self._dashboard_vis is not None
          and self._meta_dash_path is not None
          and self.step % self._meta_dash_every_n == 0
      ):
        self._save_meta_dashboard(
            pred_checkpoints=pred_checkpoints,
            bbs=bbs_vehicle_coordinate_system,
            lidar_bev=lidar_bev,
            pred_bev_semantic=pred_bev_semantic,
            speed_probs=pred_target_speed_ensemble,
            target_speeds=self.inference_target_speeds,
            ttc=_ttc,
            ttc_danger=_ttc_danger,
            multiplier=_multiplier,
            speed_before=_speed_before_vlm,
            speed_after=pred_target_speed_scalar,
            ego_speed=speed,
            steer=steer,
            throttle=throttle,
            brake=brake,
        )

    elif self.config.use_wp_gru and not self.config.inference_direct_controller:
      steer, throttle, brake = self.nets[0].control_pid(self.pred_wp,
                                                        gt_velocity,
                                                        tuned_aim_distance=bool(self.tuned_aim_distance))
    else:
      raise ValueError('An output representation was chosen that was not trained.')

    # 0.1 is just an arbitrary low number to threshold when the car is stopped
    if gt_velocity < 0.1:
      self.stuck_detector += 1
    else:
      self.stuck_detector = 0

    # Restart mechanism in case the car got stuck. Not used a lot anymore but doesn't hurt to keep it.
    if self.stuck_detector > self.config.stuck_threshold:
      self.force_move = self.config.creep_duration

    if self.force_move > 0:
      emergency_stop = False
      if self.config.backbone not in ('aim'):
        # safety check
        safety_box = deepcopy(self.lidar_buffer[-1])

        # z-axis
        safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
        safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]

        # y-axis
        safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
        safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]

        # x-axis
        safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
        safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
        emergency_stop = (len(safety_box) > 0)  # Checks if the List is empty

      if not emergency_stop:
        print('Detected agent being stuck. Step: ', self.step)
        throttle = max(self.config.creep_throttle, throttle)
        brake = False
        self.force_move -= 1
      else:
        print('Creeping stopped by safety box. Step: ', self.step)
        throttle = 0.0
        brake = True
        self.force_move = self.config.creep_duration

    if self.stop_sign_controller:
      if stop_for_stop_sign:
        throttle = 0.0
        brake = True

    if self.stop_after_meter > 0 and self.meters_travelled > self.stop_after_meter:
      print(f'Stopping after {self.stop_after_meter} meters.')
      throttle = 0.0
      brake = True

    control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
    self._log_qwen_step(
        _ttc,
        _ttc_danger,
        _speed_before_vlm,
        pred_target_speed_scalar,
        _multiplier,
        latest_result,
        control,
    )

    if self.IS_BENCH2DRIVE:
      # TODO doesn't seem to work
      metric_info = self.get_metric_info()
      self.metric_info[self.step] = metric_info
      if self.save_path is not None and self.step % 1 == 0:
        with open(self.save_path / 'metric_info.json', 'w') as outfile:
          ujson.dump(self.metric_info, outfile, indent=4)

    # CARLA will not let the car drive in the initial frames.
    # We set the action to brake so that the filter does not get confused.
    if self.step < self.config.inital_frames_delay:
      self.control = carla.VehicleControl(0.0, 0.0, 1.0)
    else:
      self.control = control

    return control

  def stop_sign_controller_step(self, ego_speed):
    """Checks whether the car is intersecting with one of the detected stop signs"""
    if self.clear_stop_sign > 0:
      self.clear_stop_sign -= 1

    if len(self.bb_buffer) < 1:
      return False
    stop_sign_stop_predicted = False
    extent = carla.Vector3D(self.config.ego_extent_x, self.config.ego_extent_y, self.config.ego_extent_z)
    origin = carla.Location(x=0.0, y=0.0, z=0.0)

    car_box = carla.BoundingBox(origin, extent)

    for bb in self.bb_buffer[-1]:
      if bb[7] == 3:  # Stop sign detected
        self.stop_sign_buffer.append(bb)

    if len(self.stop_sign_buffer) > 0:
      # Check if we need to stop
      stop_box = self.stop_sign_buffer[0]
      stop_origin = carla.Location(x=stop_box[0], y=stop_box[1], z=0.0)
      stop_extent = carla.Vector3D(stop_box[2], stop_box[3], 1.0)
      stop_carla_box = carla.BoundingBox(stop_origin, stop_extent)
      stop_carla_box.rotation = carla.Rotation(0.0, np.rad2deg(stop_box[4]), 0.0)

      if t_u.check_obb_intersection(stop_carla_box, car_box) and self.clear_stop_sign <= 0:
        if ego_speed > 0.01:
          stop_sign_stop_predicted = True
        else:
          # We have cleared the stop sign
          stop_sign_stop_predicted = False
          self.stop_sign_buffer.pop()
          # Stop signs don't come in herds, so we know we don't need to clear one for a while.
          self.clear_stop_sign = 100

    if len(self.stop_sign_buffer) > 0:
      # Remove boxes that are too far away
      if np.linalg.norm(self.stop_sign_buffer[0][:2]) > abs(self.config.max_x):
        self.stop_sign_buffer.pop()

    return stop_sign_stop_predicted

  def _summarize_rule_bboxes(self, bbs) -> dict:
    summary = {
        'bbox_traffic_light_count': 0,
        'bbox_stop_sign_count': 0,
        'bbox_vehicle_count': 0,
        'bbox_pedestrian_count': 0,
        'bbox_emergency_vehicle_count': 0,
        'nearest_traffic_light_id': None,
        'nearest_traffic_light_x': float('nan'),
        'nearest_traffic_light_y': float('nan'),
        'nearest_stop_sign_id': None,
        'nearest_stop_sign_x': float('nan'),
        'nearest_stop_sign_y': float('nan'),
    }
    if bbs is None:
      return summary

    nearest_tl = None
    nearest_stop = None
    for idx, bb in enumerate(bbs):
      if len(bb) < 8:
        continue
      class_id = int(bb[7])
      x = float(bb[0])
      y = float(bb[1])
      if class_id == 0:
        summary['bbox_vehicle_count'] += 1
      elif class_id == 1:
        summary['bbox_pedestrian_count'] += 1
      elif class_id == 2:
        summary['bbox_traffic_light_count'] += 1
        if x >= -2.0 and abs(y) <= 8.0 and (nearest_tl is None or x < nearest_tl[0]):
          nearest_tl = (x, y, idx)
      elif class_id == 3:
        summary['bbox_stop_sign_count'] += 1
        if x >= -2.0 and abs(y) <= 8.0 and (nearest_stop is None or x < nearest_stop[0]):
          nearest_stop = (x, y, idx)
      elif class_id == 4:
        summary['bbox_emergency_vehicle_count'] += 1

    if nearest_tl is not None:
      summary['nearest_traffic_light_x'] = nearest_tl[0]
      summary['nearest_traffic_light_y'] = nearest_tl[1]
      summary['nearest_traffic_light_id'] = nearest_tl[2]
    if nearest_stop is not None:
      summary['nearest_stop_sign_x'] = nearest_stop[0]
      summary['nearest_stop_sign_y'] = nearest_stop[1]
      summary['nearest_stop_sign_id'] = nearest_stop[2]
    return summary

  @staticmethod
  def _format_rule_context(summary: dict) -> str:
    lines = [
        f"traffic_light_count={int(summary.get('bbox_traffic_light_count', 0) or 0)}",
        f"stop_sign_count={int(summary.get('bbox_stop_sign_count', 0) or 0)}",
    ]
    if summary.get('nearest_traffic_light_id') is not None:
      lines.append(
          'nearest_traffic_light='
          f"id={summary.get('nearest_traffic_light_id')} "
          f"x={float(summary.get('nearest_traffic_light_x', float('nan'))):.1f}m "
          f"y={float(summary.get('nearest_traffic_light_y', float('nan'))):.1f}m"
      )
    if summary.get('nearest_stop_sign_id') is not None:
      lines.append(
          'nearest_stop_sign='
          f"id={summary.get('nearest_stop_sign_id')} "
          f"x={float(summary.get('nearest_stop_sign_x', float('nan'))):.1f}m "
          f"y={float(summary.get('nearest_stop_sign_y', float('nan'))):.1f}m"
      )
    return '\n'.join(lines)

  def _update_meta_rule_hold_state(self, ego_speed: float) -> bool:
    """Keep red/yellow light stops latched until Qwen confirms green."""
    if not self._meta_rule_hold_enabled:
      self._reset_meta_rule_hold()
      return False

    result = self._meta_rule_result
    if self._meta_rule_hold_active:
      hold_type = self._meta_rule_hold_type
      if hold_type == 'stop_sign' and ego_speed < 0.2:
        self._meta_stop_sign_release_until_step = self.step + self._meta_stop_sign_release_steps
        self._reset_meta_rule_hold()
        return False

      if hold_type in ('red_light', 'yellow_light'):
        if self._is_green_meta_release_confirmation(result):
          result_key = self._meta_rule_result_key(result)
          if result_key != self._meta_rule_hold_last_green_key:
            self._meta_rule_hold_last_green_key = result_key
            self._meta_rule_hold_green_confirmations += 1
            self._meta_rule_hold_no_stop_votes = 0
            self._meta_rule_hold_last_no_stop_key = None
          if self._meta_rule_hold_green_confirmations >= self._meta_rule_hold_green_confirmations_required:
            print(f'[MetaActionRearRuleHold] release: green confirmed ({hold_type})')
            self._meta_rule_hold_post_release_step = self.step
            self._reset_meta_rule_hold()
            return False
        elif self._is_high_conf_meta_rule_stop(result):
          self._meta_rule_hold_green_confirmations = 0
          self._meta_rule_hold_last_green_key = None
          self._meta_rule_hold_no_stop_votes = 0
          self._meta_rule_hold_last_no_stop_key = None
          self._meta_rule_hold_type = str(result.get('rule_type', hold_type)).lower()
          self._meta_rule_hold_reason = str(result.get('rule_reason', result.get('reason', '')))
        else:
          no_stop_key = result.get('request_step')
          if no_stop_key is not None and no_stop_key != self._meta_rule_hold_last_no_stop_key:
            self._meta_rule_hold_last_no_stop_key = no_stop_key
            self._meta_rule_hold_no_stop_votes += 1
            print(
                '[MetaActionRearRuleHold] no-stop vote '
                f'{self._meta_rule_hold_no_stop_votes}/2 '
                f'tl={result.get("traffic_light_state", "unknown")} step={no_stop_key}'
            )
          if ego_speed < 0.5 and self._meta_rule_hold_no_stop_votes >= 2:
            print(f'[MetaActionRearRuleHold] release: no-stop votes ({hold_type})')
            self._meta_rule_hold_post_release_step = self.step
            self._reset_meta_rule_hold()
            return False

      if (
          self._meta_rule_hold_safety_steps > 0
          and self._meta_rule_hold_since_step >= 0
          and self.step - self._meta_rule_hold_since_step > self._meta_rule_hold_safety_steps
      ):
        print(
            '[MetaActionRearRuleHold] safety release: '
            f'{self._meta_rule_hold_type} age={self.step - self._meta_rule_hold_since_step}'
        )
        self._meta_rule_hold_post_release_step = self.step
        self._reset_meta_rule_hold()
        return False

      return True

    if not self._is_high_conf_meta_rule_stop(result):
      return False

    rule_type = str(result.get('rule_type', 'none')).lower()
    if rule_type == 'stop_sign' and self.step < self._meta_stop_sign_release_until_step:
      return False

    self._meta_rule_hold_active = True
    self._meta_rule_hold_type = rule_type
    self._meta_rule_hold_since_step = self.step
    self._meta_rule_hold_green_confirmations = 0
    self._meta_rule_hold_last_green_key = None
    self._meta_rule_hold_no_stop_votes = 0
    self._meta_rule_hold_last_no_stop_key = None
    self._meta_rule_hold_reason = str(result.get('rule_reason', result.get('reason', '')))
    print(f'[MetaActionRearRuleHold] activate: {rule_type} step={self.step}')
    return True

  def _reset_meta_rule_hold(self) -> None:
    self._meta_rule_hold_active = False
    self._meta_rule_hold_type = 'none'
    self._meta_rule_hold_since_step = -1
    self._meta_rule_hold_green_confirmations = 0
    self._meta_rule_hold_last_green_key = None
    self._meta_rule_hold_no_stop_votes = 0
    self._meta_rule_hold_last_no_stop_key = None
    self._meta_rule_hold_reason = ''

  def _is_high_conf_meta_rule_stop(self, result: dict) -> bool:
    if (
        self._meta_rule_hold_post_release_cooldown > 0
        and self._meta_rule_hold_post_release_step >= 0
        and self.step - self._meta_rule_hold_post_release_step < self._meta_rule_hold_post_release_cooldown
    ):
      return False
    if not bool(result.get('rule_intervene', False)):
      return False
    if not bool(result.get('relevant_to_ego', False)):
      return False
    rule_type = str(result.get('rule_type', 'none')).lower()
    if rule_type not in ('red_light', 'yellow_light', 'stop_sign'):
      return False
    if rule_type == 'stop_sign' and not self._meta_rule_enable_stop_sign:
      return False
    try:
      confidence = float(result.get('rule_confidence', 0.0))
      request_step = int(result.get('request_step'))
      rule_scale = float(result.get('rule_speed_scale', 1.0))
    except (TypeError, ValueError):
      return False
    if confidence < self._meta_rule_confidence_threshold:
      return False
    if rule_scale >= 0.95:
      return False
    age = self.step - request_step
    return 0 <= age <= self._meta_rule_result_max_age

  def _is_green_meta_release_confirmation(self, result: dict) -> bool:
    try:
      confidence = float(result.get('rule_confidence', 0.0))
      request_step = int(result.get('request_step'))
    except (TypeError, ValueError):
      return False
    age = self.step - request_step
    return (
        0 <= age <= self._meta_rule_result_max_age
        and str(result.get('traffic_light_state', 'unknown')).lower() == 'green'
        and bool(result.get('relevant_to_ego', False))
        and confidence >= self._meta_rule_confidence_threshold
    )

  @staticmethod
  def _meta_rule_result_key(result: dict):
    return (
        result.get('request_step'),
        str(result.get('traffic_light_state', 'unknown')),
        str(result.get('rule_type', 'none')),
        str(result.get('raw_response', ''))[:120],
    )

  def _should_apply_meta_rule_result(self, ego_speed: float) -> bool:
    if not self._is_high_conf_meta_rule_stop(self._meta_rule_result):
      return False
    rule_type = str(self._meta_rule_result.get('rule_type', 'none')).lower()
    if rule_type == 'stop_sign' and self.step < self._meta_stop_sign_release_until_step:
      return False
    if rule_type == 'stop_sign' and ego_speed < 0.2:
      self._meta_stop_sign_release_until_step = self.step + self._meta_stop_sign_release_steps
      return False
    return True

  def _should_meta_tl_prestop(self, rule_summary: dict) -> bool:
    if not self._meta_tl_prestop_enabled:
      return False
    if self._meta_rule_active or self._meta_rule_hold_active:
      return False
    if int(rule_summary.get('bbox_traffic_light_count', 0) or 0) <= 0:
      return False
    try:
      light_x = float(rule_summary.get('nearest_traffic_light_x', 999.0))
    except (TypeError, ValueError):
      light_x = 999.0
    if not np.isfinite(light_x) or light_x < -2.0 or light_x > self._meta_tl_prestop_distance:
      return False
    if (
        self._meta_rule_hold_post_release_cooldown > 0
        and self._meta_rule_hold_post_release_step >= 0
        and self.step - self._meta_rule_hold_post_release_step < self._meta_rule_hold_post_release_cooldown
    ):
      return False
    try:
      request_step = int(self._meta_rule_result.get('request_step'))
      confidence = float(self._meta_rule_result.get('rule_confidence', 0.0))
      age = self.step - request_step
    except (TypeError, ValueError):
      return True
    no_intervene = not bool(self._meta_rule_result.get('rule_intervene', False))
    if no_intervene and confidence >= self._meta_rule_confidence_threshold and 0 <= age <= self._meta_rule_result_max_age:
      return False
    return True

  def _compute_ttc(self, bbs, ego_speed: float) -> float:
    """
    TTC = bb[0] / ego_speed (앞 차 정지 가정, 보수적 추정)

    bb 좌표계: x=전방(양수=앞), y=측면
    차량/보행자/긴급차(class 0,1,4)만 고려.
    차선 밖(abs(y) > 2.0m) 객체 제외.

    Returns:
        TTC (초). 위험 없으면 999.0.
    """
    if ego_speed < 0.5 or bbs is None or len(bbs) == 0:
      return 999.0

    min_ttc = 999.0
    for bb in bbs:
      if len(bb) < 8:
        continue
      class_id = int(bb[7])
      if class_id not in (0, 1, 4):  # 차량, 보행자, 긴급차량만
        continue
      x = float(bb[0])   # 전방 거리
      y = float(bb[1])   # 측방 거리
      if x < 0.5:         # 후방 또는 너무 가까운 객체 제외
        continue
      if abs(y) > 2.0:    # 차선 밖 객체 제외
        continue
      ttc = x / ego_speed
      if ttc < min_ttc:
        min_ttc = ttc

    return min_ttc

  def _save_meta_dashboard(
      self,
      pred_checkpoints,
      bbs,
      ttc: float,
      ttc_danger: bool,
      multiplier: float,
      speed_before: float,
      speed_after: float,
      ego_speed: float,
      lidar_bev=None,
      pred_bev_semantic=None,
      speed_probs=None,
      target_speeds=None,
      steer: float = 0.0,
      throttle: float = 0.0,
      brake: float = 0.0,
  ) -> None:
    """대시보드 이미지를 META_DASHBOARD_PATH/<route>/ 에 저장 (rear 대시보드 형식)."""
    if self._dashboard_vis is None or self._meta_dash_path is None:
      return
    if self._meta_rgb_np is None:
      return

    try:
      save_path_png = self._meta_dash_path / f'{self.step:05d}.png'

      front_dashboard = self._dashboard_rgb_np if self._dashboard_rgb_np is not None else self._meta_rgb_np
      rear_dashboard = (
          self._dashboard_rgb_rear_np if self._dashboard_rgb_rear_np is not None else self._meta_rgb_rear_np
      )

      if self._meta_rule_active or self._meta_rule_hold_active:
        rule_type = str(self._meta_rule_result.get('rule_type', self._meta_rule_hold_type)).replace('_', ' ')
        action_name = 'stop'
        action_idx = 2
        action_reason = str(self._meta_rule_result.get('rule_reason') or f'Rule hold is active for {rule_type}.')
      elif ttc_danger:
        action_name = self.meta_action.get_action_name()
        action_idx = self.meta_action.get_action_idx()
        action_reason = self.meta_action.get_action_reason()
      else:
        action_name = 'proceed'
        action_idx = 0
        action_reason = 'Ego corridor is clear, allowing the ego vehicle to proceed.'

      dashboard_vlm_reason = ''
      if os.environ.get('META_DASHBOARD_VLM_EXPLANATION', '1').strip() != '0':
        dashboard_vlm_reason = self.meta_action.explain_dashboard_frame(
            front_dashboard,
            rear_dashboard,
            self.step,
            action_name=action_name,
            ttc=ttc,
            ego_speed=ego_speed,
        )
        if dashboard_vlm_reason:
          action_reason = dashboard_vlm_reason

      dashboard_bbs = bbs
      if dashboard_bbs is None and len(self.bb_buffer) > 0:
        dashboard_bbs = self.bb_buffer[-1]

      frame_data = {
          'image': front_dashboard,
          'rear_image': rear_dashboard,
          'step': self.step,
          'ego_speed': ego_speed,
          'ttc': ttc if ttc < 999.0 else 999.0,
          'ttc_threshold': self.meta_action.ttc_threshold,
          'intervention': ttc_danger,
          'action_name': action_name,
          'action_idx': action_idx,
	          'action_reason': action_reason,
	          'dashboard_vlm_reason': dashboard_vlm_reason,
	          'multiplier': multiplier,
	          'vlm_ran': bool(dashboard_vlm_reason) or self.step == self.meta_action._last_trigger_step,
	          'rule_active': self._meta_rule_active,
	          'rule_type': str(self._meta_rule_result.get('rule_type', 'none')),
	          'rule_confidence': float(self._meta_rule_result.get('rule_confidence', 0.0)),
	          'rule_relevant': bool(self._meta_rule_result.get('relevant_to_ego', False)),
	          'rule_speed_scale': self._meta_rule_scale,
	          'traffic_light_state': str(self._meta_rule_result.get('traffic_light_state', 'unknown')),
	          'stop_sign_visible': bool(self._meta_rule_result.get('stop_sign_visible', False)),
	          'rule_reason': str(self._meta_rule_result.get('rule_reason', '')),
	          'rule_hold_active': self._meta_rule_hold_active,
	          'rule_hold_type': self._meta_rule_hold_type,
	          'rule_hold_green_confirmations': self._meta_rule_hold_green_confirmations,
	          'tl_prestop_active': self._meta_tl_prestop_active,
	          'bbox_traffic_light_count': self._meta_rule_summary.get('bbox_traffic_light_count', 0),
	          'bbox_stop_sign_count': self._meta_rule_summary.get('bbox_stop_sign_count', 0),
	          'nearest_traffic_light_id': self._meta_rule_summary.get('nearest_traffic_light_id'),
	          'nearest_traffic_light_x': self._meta_rule_summary.get('nearest_traffic_light_x', float('nan')),
	          'nearest_traffic_light_y': self._meta_rule_summary.get('nearest_traffic_light_y', float('nan')),
	          'tfpp_speed_mps': speed_before,
	          'final_speed_mps': speed_after,
          'steer': steer,
          'throttle': throttle,
          'brake': brake,
          'pred_checkpoints': pred_checkpoints,
          'pred_boxes': dashboard_bbs,
          'lidar_bev': lidar_bev,
          'pred_bev_semantic': pred_bev_semantic,
          'speed_probs': speed_probs,
          'target_speeds': target_speeds,
          'history_key': str(self._meta_dash_path),
          'reset_history': self.step <= 5,
      }

      self._dashboard_vis(frame_data, save_path=str(save_path_png))

    except Exception as e:  # pylint: disable=broad-except
      print(f'[MetaDash] visualization error at step={self.step}: {e}')

  def bb_detected_in_front_of_vehicle(self, ego_speed):
    if len(self.bb_buffer) < 1:  # We only start after we have 4 time steps.
      return False

    collision_predicted = False

    extent = carla.Vector3D(self.config.ego_extent_x, self.config.ego_extent_y, self.config.ego_extent_z)

    # Safety box
    bremsweg = ((ego_speed.cpu().numpy().item() * 3.6) / 10.0)**2 / 2.0  # Bremsweg formula for emergency break
    safety_x = np.clip(bremsweg + 1.0, a_min=2.0, a_max=4.0)  # plus one meter is the car.

    center_safety_box = carla.Location(x=safety_x, y=0.0, z=1.0)

    safety_bounding_box = carla.BoundingBox(center_safety_box, extent)
    safety_bounding_box.rotation = carla.Rotation(0.0, 0.0, 0.0)

    for bb in self.bb_buffer[-1]:
      # We just give them some arbitrary height. Does not matter
      bb_extent_z = 1.0
      loc_local = carla.Location(bb[0], bb[1], 0.0)
      extent_det = carla.Vector3D(bb[2], bb[3], bb_extent_z)
      bb_local = carla.BoundingBox(loc_local, extent_det)
      bb_local.rotation = carla.Rotation(0.0, np.rad2deg(bb[4]).item(), 0.0)

      if t_u.check_obb_intersection(safety_bounding_box, bb_local):
        collision_predicted = True

    return collision_predicted

  def align_lidar(self, lidar, x, y, orientation, x_target, y_target, orientation_target):
    pos_diff = np.array([x_target, y_target, 0.0]) - np.array([x, y, 0.0])
    rot_diff = t_u.normalize_angle(orientation_target - orientation)

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                [np.sin(orientation_target),
                                 np.cos(orientation_target), 0.0], [0.0, 0.0, 1.0]])
    pos_diff = rotation_matrix.T @ pos_diff

    return t_u.algin_lidar(lidar, pos_diff, rot_diff)

  def update_stop_box(self, boxes, x, y, orientation, x_target, y_target, orientation_target):
    pos_diff = np.array([x_target, y_target]) - np.array([x, y])
    rot_diff = t_u.normalize_angle(orientation_target - orientation)

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target)],
                                [np.sin(orientation_target), np.cos(orientation_target)]])
    pos_diff = rotation_matrix.T @ pos_diff

    # Rotation matrix in local coordinate system
    local_rot_matrix = np.array([[np.cos(rot_diff), -np.sin(rot_diff)], [np.sin(rot_diff), np.cos(rot_diff)]])

    for _, box_pred in enumerate(boxes):
      box_pred[:2] = (local_rot_matrix.T @ (box_pred[:2] - pos_diff).T).T
      box_pred[4] = t_u.normalize_angle(box_pred[4] - rot_diff)

  def destroy(self, results=None):  # pylint: disable=locally-disabled, unused-argument
    """
    Gets called after a route finished.
    The leaderboard client doesn't properly clear up the agent after the route finishes so we need to do it here.
    Also writes logging files to disk.
    """
    if self.save_path is not None:
      self.lon_logger.dump_to_json()
      if len(self.nets[0].speed_histogram) > 0:
        with gzip.open(self.save_path / 'target_speeds.json.gz', 'wt', encoding='utf-8') as f:
          ujson.dump(self.nets[0].speed_histogram, f, indent=4)

      if self.config.tp_attention:
        if len(self.tp_attention_buffer) > 0:
          print('Average TP attention: ', sum(self.tp_attention_buffer) / len(self.tp_attention_buffer))
          with gzip.open(self.save_path / 'tp_attention.json.gz', 'wt', encoding='utf-8') as f:
            ujson.dump(self.tp_attention_buffer, f, indent=4)

        del self.tp_attention_buffer
      if self._qwen_log_f is not None:
        self._qwen_log_f.close()
        self._qwen_log_f = None

    del self.nets
    del self.config
    del self.metric_info

  def _log_qwen_step(self, ttc, ttc_danger, speed_before, speed_after, multiplier, meta_result, control):
    if self._qwen_log_f is None:
      return
    request_id = meta_result.get('request_id')
    if request_id is None or request_id == self._last_logged_qwen_request_id:
      return

    prompt_mode = str(meta_result.get('prompt_mode') or 'speed')
    action_idx = int(meta_result.get('action_idx', self.meta_action.get_action_idx()))
    action_name = str(meta_result.get('action_name', self.meta_action.get_action_name()))
    entry = {
        'experiment': 'rear_classiccv_rulehold_8meta_action_frontrear',
        'image_enhancer': 'classic_cv' if self.enhancer is not None else 'off',
        'step': int(self.step),
        'ttc': round(float(ttc), 3) if ttc < 999.0 else 999.0,
        'is_risky': bool(ttc_danger),
        'vlm_called': True,
        'vlm_trigger': meta_result.get('request_trigger') or ('ttc' if ttc_danger else 'none'),
        'prompt_mode': prompt_mode,
        'action_idx': action_idx,
        'action_name': action_name,
        'qwen_intervene': action_idx != 0,
        'speed_scale': round(float(multiplier), 4),
        'tfpp_target_speed': round(float(speed_before), 3),
        'final_target_speed': round(float(speed_after), 3),
        'qwen_raw_response': str(meta_result.get('raw_response', ''))[:200],
        'qwen_request_step': meta_result.get('request_step'),
        'qwen_request_trigger': meta_result.get('request_trigger'),
        'rule_type': meta_result.get('rule_type'),
        'traffic_light_state': meta_result.get('traffic_light_state'),
        'rule_intervene': meta_result.get('rule_intervene'),
        'rule_speed_scale': meta_result.get('rule_speed_scale'),
        'control_throttle': round(float(getattr(control, 'throttle', 0.0)), 4),
        'control_brake': round(float(getattr(control, 'brake', 0.0)), 4),
        'control_steer': round(float(getattr(control, 'steer', 0.0)), 4),
    }
    if meta_result.get('benchmark') is not None:
      bench_key = 'rule_benchmark' if prompt_mode == 'traffic_rule' else 'qwen_benchmark'
      entry[bench_key] = meta_result['benchmark']
    self._last_logged_qwen_request_id = request_id
    self._qwen_log_f.write(ujson.dumps(entry) + '\n')
    self._qwen_log_f.flush()


# Filter Functions
def bicycle_model_forward(x, dt, steer, throttle, brake):
  # Kinematic bicycle model.
  # Numbers are the tuned parameters from World on Rails
  front_wb = -0.090769015
  rear_wb = 1.4178275

  steer_gain = 0.36848336
  brake_accel = -4.952399
  throt_accel = 0.5633837

  locs_0 = x[0]
  locs_1 = x[1]
  yaw = x[2]
  speed = x[3]

  if brake:
    accel = brake_accel
  else:
    accel = throt_accel * throttle

  wheel = steer_gain * steer

  beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
  next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
  next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
  next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
  next_speed = speed + accel * dt
  next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

  next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

  return next_state_x


def measurement_function_hx(vehicle_state):
  '''
    For now we use the same internal state as the measurement state
    :param vehicle_state: VehicleState vehicle state variable containing
                          an internal state of the vehicle from the filter
    :return: np array: describes the vehicle state as numpy array.
                       0: pos_x, 1: pos_y, 2: rotatoion, 3: speed
    '''
  return vehicle_state


def state_mean(state, wm):
  '''
    We use the arctan of the average of sin and cos of the angle to calculate
    the average of orientations.
    :param state: array of states to be averaged. First index is the timestep.
    :param wm:
    :return:
    '''
  x = np.zeros(4)
  sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
  sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
  x[0] = np.sum(np.dot(state[:, 0], wm))
  x[1] = np.sum(np.dot(state[:, 1], wm))
  x[2] = math.atan2(sum_sin, sum_cos)
  x[3] = np.sum(np.dot(state[:, 3], wm))

  return x


def measurement_mean(state, wm):
  '''
  We use the arctan of the average of sin and cos of the angle to
  calculate the average of orientations.
  :param state: array of states to be averaged. First index is the
  timestep.
  '''
  x = np.zeros(4)
  sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
  sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
  x[0] = np.sum(np.dot(state[:, 0], wm))
  x[1] = np.sum(np.dot(state[:, 1], wm))
  x[2] = math.atan2(sum_sin, sum_cos)
  x[3] = np.sum(np.dot(state[:, 3], wm))

  return x


def residual_state_x(a, b):
  y = a - b
  y[2] = t_u.normalize_angle(y[2])
  return y


def residual_measurement_h(a, b):
  y = a - b
  y[2] = t_u.normalize_angle(y[2])
  return y


class EgoModel:
  """
      Kinematic bicycle model describing the motion of a car given it's state and
      action. Tuned parameters are taken from World on Rails.
      """

  def __init__(self, dt, ego_vehicle_model=True):
    self.dt = dt  # the following numbers are optimized for dt=1./20. = 20 FPS

    self.ego_vehicle_model = ego_vehicle_model

    # Kinematic bicycle model. Numbers are the tuned parameters from World
    # on Rails
    self.front_wb = -0.090769015
    self.rear_wb = 1.4178275
    self.steer_gain = 0.36848336
    self.brake_accel = -4.952399
    self.throt_accel = 0.5633837

    # Numbers are tuned parameters for the polynomial equations below using
    # a dataset where the car drives on a straight highway, accelerates to
    # 80 km/h and brakes to 0 km/h
    self.throt_values = np.array([
        9.63873001e-01, 4.37535692e-04, -3.80192912e-01, 1.74950069e+00, 9.16787414e-02, -7.05461530e-02,
        -1.05996152e-03, 6.71079346e-04
    ])
    self.brake_values = np.array([
        9.31711370e-03, 8.20967431e-02, -2.83832427e-03, 5.06587474e-05, -4.90357228e-07, 2.44419284e-09,
        -4.91381935e-12
    ])

  def forward(self, locs, yaws, spds, acts):
    # Kinematic bicycle model. Numbers are the tuned parameters from World
    # on Rails
    steer = acts[..., 0:1].item()
    throt = acts[..., 1:2].item()
    brake = acts[..., 2:3].astype(np.uint8)

    wheel = self.steer_gain * steer

    beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
    yaws = yaws.item()
    spds = spds.item()
    next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
    next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
    next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt

    if self.ego_vehicle_model:
      if brake:
        spds = spds * 3.6
        features = np.array([spds, spds**2, spds**3, spds**4, spds**5, spds**6, spds**7]).T

        next_spds = (features @ self.brake_values).item() / 3.6
      else:
        throttle = np.clip(throt, 0., 1.0)
        # for a throttle value < 0.3 the car doesn't accelerate and the polynomial model below breaks
        if throttle < 0.3:
          next_spds = spds
        else:
          spds = spds * 3.6
          features = np.array([
              spds, spds**2, throttle, throttle**2, spds * throttle, spds * throttle**2, spds**2 * throttle,
              spds**2 * throttle**2
          ]).T

          next_spds = (features @ self.throt_values).item() / 3.6
    else:
      if brake:
        next_spds = spds + self.brake_accel * self.dt
      else:
        next_spds = spds + self.throt_accel * self.dt

    next_spds = max(0, next_spds)

    next_locs = np.array([next_locs_0, next_locs_1, locs[2]])
    next_yaws = np.array(next_yaws)
    next_spds = np.array(next_spds)

    return next_locs, next_yaws, next_spds
