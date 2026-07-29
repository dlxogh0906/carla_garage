"""
sensor_agent_meta_action.py â TF++ + Meta-action VLA (Qwen2.5-VL)

sensor_agent.py ìë³¸ ë¬´ìì . ì´ íì¼ììë§ ë³ê²½.
ë³ê²½ ë´ì©:
  1. MetaActionVLAPlanner (meta_action_vla.py) ì´ê¸°í
  2. TTC ê³ì° (bb[0] / ego_speed, ì ì§ ê°ì  ë³´ìì  ì¶ì )
  3. TTC < ttc_threshold ììë§ VLM í¸ë¦¬ê±° (TF++ ì ì ì£¼í ë³´ì¡´)
  4. 8ê° ë©í-ì¡ì â ìë multiplier ì ì©
  5. Qwen dashboard ìê°í ì ì¥ (qwen_dashboard.py ì¬ì©)
"""

import os
import sys
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
#from allweathernet_enhancer import build_enhancer
import transfuser_utils as t_u

_GARAGE_EXT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _GARAGE_EXT_SRC not in sys.path:
  sys.path.insert(0, _GARAGE_EXT_SRC)

from garage_ext.vlm_intervention.qwen_input import (
    annotate_qwen_image,
    build_object_context,
    format_object_table,
    format_gap_context,
    format_path_summary,
    format_rule_context,
    summarize_gap_context,
    summarize_bbox_classes,
    summarize_objects,
)
from garage_ext.vlm_intervention.ttc import compute_sensitive_ttc, get_front_distance

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

    # Optional AllWeatherNet image enhancement
    use_awn = int(os.environ.get('USE_ALLWEATHERNET', 0))
    if use_awn:
      self.enhancer = build_enhancer(self.device)
      if self.enhancer is None:
        print('[AWN] Enhancement disabled — checkpoint missing.')
    else:
      self.enhancer = None

    # Dedicated visualization output path â per-route subfolder to avoid overwrite
    awn_vis = os.environ.get('AWN_VIS_PATH', None)
    awn_vis_max_routes = int(os.environ.get('AWN_VIS_MAX_ROUTES', 0))
    if awn_vis and self.enhancer is not None:
      awn_vis_dir = pathlib.Path(awn_vis)
      awn_vis_dir.mkdir(parents=True, exist_ok=True)
      enable_vis = True
      if awn_vis_max_routes > 0:
        counter_file = awn_vis_dir / '.route_counter'
        route_count = int(counter_file.read_text().strip()) + 1 if counter_file.exists() else 1
        counter_file.write_text(str(route_count))
        enable_vis = route_count <= awn_vis_max_routes
        if not enable_vis:
          print(f'[AWN] Route {route_count} exceeds AWN_VIS_MAX_ROUTES={awn_vis_max_routes}, skipping visualization.')
      if enable_vis:
        save_name = path_to_conf_file.split('+')[-1] if '+' in path_to_conf_file else (str(route_index) if route_index else 'route')
        self.awn_vis_path = awn_vis_dir / save_name
        self.awn_vis_path.mkdir(parents=True, exist_ok=True)
        print(f'[AWN] Saving visualizations to: {self.awn_vis_path}')
      else:
        self.awn_vis_path = None
    else:
      self.awn_vis_path = None

    # ----------------------------------------------------------------
    # Meta-action VLA: Qwen2.5-VL â TTC ê¸°ë° 8-class ë©í-ì¡ì íëë
    # ----------------------------------------------------------------
    import sys  # pylint: disable=import-outside-toplevel
    _b2d_path = os.path.join(os.path.dirname(__file__), '..', 'Bench2Drive')
    if _b2d_path not in sys.path:
      sys.path.insert(0, _b2d_path)

    from meta_action_vla import MetaActionVLAPlanner  # pylint: disable=import-outside-toplevel
    self._meta_route_name = (
        path_to_conf_file.split('+')[-1]
        if '+' in path_to_conf_file
        else (str(route_index) if route_index else 'route')
    )
    meta_every_n = int(os.environ.get('META_EVERY_N_STEPS', 20))
    meta_ttc_threshold = float(os.environ.get('META_TTC_THRESHOLD', 3.0))
    self.meta_action = MetaActionVLAPlanner(
        self.device,
        ttc_threshold=meta_ttc_threshold,
        inference_every_n=meta_every_n,
    )
    self._meta_rgb_np = None   # original full-res RGB for VLA inference
    self._meta_lateral_thresh = float(os.environ.get('META_LATERAL_THRESH', 2.0))
    self._meta_front_max_distance = float(os.environ.get('META_FRONT_MAX_DISTANCE', 80.0))
    self._meta_max_objects = max(1, int(os.environ.get('META_MAX_OBJECTS', 8)))
    self._meta_save_inputs = int(os.environ.get('META_SAVE_INPUTS', 1)) == 1
    self._meta_ttc_history = deque(maxlen=20)
    self._meta_rule_every_n = max(1, int(os.environ.get('META_RULE_EVERY_N_STEPS', 5)))
    self._meta_rule_confidence_threshold = float(os.environ.get('META_RULE_CONFIDENCE_THRESH', 0.75))
    self._meta_rule_result_max_age = max(1, int(os.environ.get('META_RULE_MAX_AGE_STEPS', 20)))
    self._meta_rule_enable_stop_sign = int(os.environ.get('META_RULE_ENABLE_STOP_SIGN', 0)) == 1
    self._meta_rule_hold_enabled = int(os.environ.get('META_RULE_HOLD_ENABLED', 1)) == 1
    self._meta_rule_hold_poll_steps = max(1, int(os.environ.get('META_RULE_HOLD_POLL_STEPS', 5)))
    self._meta_rule_hold_green_confirmations_required = max(
        1, int(os.environ.get('META_RULE_HOLD_GREEN_CONFIRMATIONS', 2)))
    self._meta_rule_hold_max_steps = max(0, int(os.environ.get('META_RULE_HOLD_MAX_STEPS', 0)))
    self._meta_rule_hold_no_stop_votes_required = max(
        1, int(os.environ.get('META_RULE_HOLD_NO_STOP_VOTES', 2)))
    self._meta_rule_hold_safety_steps = max(0, int(os.environ.get('META_RULE_HOLD_SAFETY_STEPS', 80)))
    self._meta_rule_hold_post_release_cooldown = max(
        0, int(os.environ.get('META_RULE_HOLD_POST_RELEASE_COOLDOWN', 30)))
    self._meta_stop_sign_release_steps = max(0, int(os.environ.get('META_STOP_SIGN_RELEASE_STEPS', 60)))
    self._meta_tl_prestop_enabled = int(os.environ.get('META_TL_PRESTOP_ENABLED', 1)) == 1
    self._meta_tl_prestop_scale = float(os.environ.get('META_TL_PRESTOP_SCALE', 0.45))
    self._meta_tl_prestop_distance = float(os.environ.get('META_TL_PRESTOP_DISTANCE', 45.0))
    self._meta_gap_enabled = int(os.environ.get('META_GAP_CRITIC_ENABLED', 1)) == 1
    self._meta_gap_result_max_age = max(1, int(os.environ.get('META_GAP_MAX_AGE_STEPS', 40)))
    self._meta_gap_confidence_threshold = float(os.environ.get('META_GAP_CONFIDENCE_THRESH', 0.55))
    self._meta_gap_lookahead = float(os.environ.get('META_GAP_LOOKAHEAD_DISTANCE', 45.0))
    self._meta_gap_side_distance = float(os.environ.get('META_GAP_SIDE_DISTANCE', 14.0))
    self._meta_gap_visual_probe = int(os.environ.get('META_GAP_VISUAL_PROBE', 1)) == 1
    self._meta_gap_no_candidate_release_steps = max(
      0, int(os.environ.get('META_GAP_RELEASE_NO_CANDIDATE_STEPS', 12)))
    self._meta_gap_stop_release_steps = max(
      0, int(os.environ.get('META_GAP_STOP_RELEASE_STEPS', 35)))
    self._meta_gap_stop_release_scale = float(os.environ.get('META_GAP_STOP_RELEASE_SCALE', 0.35))
    self._meta_gap_route_only = int(os.environ.get('META_GAP_ROUTE_ONLY', 1)) == 1
    self._meta_gap_min_scale = float(os.environ.get('META_GAP_MIN_SCALE', 0.85))
    self._meta_gap_strong_confidence_threshold = float(os.environ.get('META_GAP_STRONG_CONFIDENCE_THRESH', 0.82))
    self._meta_gap_stop_confidence_threshold = float(os.environ.get('META_GAP_STOP_CONFIDENCE_THRESH', 0.92))
    self._meta_gap_immediate_x = float(os.environ.get('META_GAP_IMMEDIATE_X', 12.0))
    self._meta_gap_immediate_y = float(os.environ.get('META_GAP_IMMEDIATE_Y', 6.0))
    self._meta_speed_semantic_enabled = int(os.environ.get('META_SPEED_SEMANTIC_ENABLED', 1)) == 1
    self._meta_speed_result_max_age = max(1, int(os.environ.get('META_SPEED_MAX_AGE_STEPS', 12)))
    self._meta_recovery_enabled = int(os.environ.get('META_RECOVERY_ENABLED', 1)) == 1
    self._meta_recovery_stuck_steps = max(1, int(os.environ.get('META_RECOVERY_STUCK_STEPS', 25)))
    self._meta_recovery_duration_steps = max(1, int(os.environ.get('META_RECOVERY_DURATION_STEPS', 35)))
    self._meta_recovery_target_speed = float(os.environ.get('META_RECOVERY_TARGET_SPEED', 1.5))
    self._meta_recovery_throttle = float(os.environ.get('META_RECOVERY_THROTTLE', 0.45))
    self._meta_recovery_requires_motion = int(os.environ.get('META_RECOVERY_REQUIRES_MOTION', 1)) == 1
    self._meta_recovery_motion_frames_required = max(
      1, int(os.environ.get('META_RECOVERY_MOTION_FRAMES', 5)))
    self._meta_turn_caution_enabled = int(os.environ.get('META_TURN_CAUTION_ENABLED', 1)) == 1
    self._meta_turn_caution_speed = float(os.environ.get('META_TURN_CAUTION_SPEED', 3.0))
    self._meta_turn_caution_lateral = float(os.environ.get('META_TURN_CAUTION_LATERAL', 2.5))
    self._meta_escape_reverse_enabled = int(os.environ.get('META_ESCAPE_REVERSE_ENABLED', 1)) == 1
    self._meta_escape_stuck_steps = max(1, int(os.environ.get('META_ESCAPE_STUCK_STEPS', 90)))
    self._meta_escape_reverse_steps = max(1, int(os.environ.get('META_ESCAPE_REVERSE_STEPS', 14)))
    self._meta_escape_throttle = float(os.environ.get('META_ESCAPE_THROTTLE', 0.35))
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
    self._meta_tl_prestop_active = False
    self._meta_rule_active = False
    self._meta_rule_scale = 1.0
    self._meta_gap_active = False
    self._meta_gap_scale = 1.0
    self._meta_gap_summary = {}
    self._meta_gap_stop_hold_steps = 0
    self._meta_gap_no_candidate_steps = 0
    self._meta_speed_active = False
    self._meta_recovery_active = False
    self._meta_recovery_until_step = -1
    self._meta_recovery_motion_frames = 0
    self._meta_recovery_ever_moving = False
    self._meta_turn_caution_active = False
    self._meta_escape_reverse_active = False
    self._meta_escape_reverse_until_step = -1
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
    self._meta_speed_result = {
        'intervene': False,
        'risk_level': 'low',
        'speed_scale': 1.0,
        'reason': '',
        'request_step': None,
        'prompt_mode': 'speed',
    }
    self._meta_gap_result = {
        'intervene': False,
        'risk_level': 'low',
        'speed_scale': 1.0,
        'reason': '',
        'request_step': None,
        'prompt_mode': 'gap',
        'gap_decision': 'unknown',
        'clear_to_enter': True,
        'cross_traffic': False,
        'gap_confidence': 0.0,
    }
    self._meta_vlm_called_this_step = False
    self._meta_vlm_trigger_this_step = 'none'
    self._meta_last_qwen_image = None
    self._meta_object_count = 0
    self._meta_primary_object_id = None
    self._meta_object_summary = summarize_objects([])
    self._meta_rule_summary = summarize_bbox_classes([])
    self._meta_planner_braking_cue = False
    self._meta_ttc_source = 'none'
    self._qwen_dashboard_interval = max(
        1, int(os.environ.get('EXT_DASHBOARD_INTERVAL', os.environ.get('META_DASHBOARD_INTERVAL', 4))))
    self._qwen_dashboard_render = None
    try:
      from garage_ext.visualization.qwen_dashboard import render_qwen_dashboard  # pylint: disable=import-outside-toplevel
      self._qwen_dashboard_render = render_qwen_dashboard
      print('[QwenDash] Dashboard visualization enabled.')
    except Exception as e:  # pylint: disable=broad-except
      print(f'[QwenDash] Dashboard import failed: {e}')
    self._dashboard_vis = None  # legacy dashboard disabled; QwenDash renders both dashboard outputs.

    # META_DASHBOARD_PATH: ORIG_VIS_PATH ë°©ìê³¼ ëì¼íê² routeë³ ìë¸í´ë ìì±
    # leaderboard_evaluatorê° route_indexë¥¼ ëê¸°ì§ ìì¼ë¯ë¡ path_to_conf_fileë¡ ë£¨í¸ëª ì¶ì¶
    meta_dash_root = os.environ.get('META_DASHBOARD_PATH', None)
    if meta_dash_root:
      save_name = path_to_conf_file.split('+')[-1] if '+' in path_to_conf_file else 'route'
      self._meta_dash_path = pathlib.Path(meta_dash_root) / save_name
      self._meta_dash_path.mkdir(parents=True, exist_ok=True)
      print(f'[QwenDash] Saving meta dashboard to: {self._meta_dash_path}')
    else:
      self._meta_dash_path = None

    self._meta_input_path = None
    if self._meta_save_inputs and self._meta_dash_path is not None:
      self._meta_input_path = self._meta_dash_path / 'vlm_input'
      self._meta_input_path.mkdir(parents=True, exist_ok=True)

    self.orig_vis_path = None  # meta_action ììë orig_vis ë¯¸ì¬ì©

    # Path to where visualizations and other debug output gets stored
    self.save_path = os.environ.get('SAVE_PATH', None)

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

  def _init(self):
    # The CARLA leaderboard does not expose the lat lon reference value of the GPS which make it impossible to use the
    # GPS because the scale is not known. In the past this was not an issue since the reference was constant 0.0
    # But town 13 has a different value in CARLA 0.9.15. The following code, adapted from Bench2DriveZoo estimates the
    # lat, lon reference values by abusing the fact that the leaderboard exposes the route plan also in CARLA
    # coordinates. The GPS plan is compared to the CARLA coordinate plan to estimate the reference point / scale
    # of the GPS. It seems to work reasonably well, so we use this workaround for now.
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

      rgb_pos = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)

      # Store original full-res RGB (used by Meta-action VLA)
      self._meta_rgb_np = rgb_pos.copy()

      # AllWeatherNet enhancement (applied before crop to preserve full resolution)
      if self.enhancer is not None:
        original_rgb = rgb_pos.copy()
        rgb_pos = self.enhancer.enhance(rgb_pos)
        # Save side-by-side visualization every 10 steps
        if self.awn_vis_path is not None and self.step % 5 == 0:
          side_by_side = np.concatenate([original_rgb, rgb_pos], axis=1)
          side_by_side_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)
          cv2.imwrite(str(self.awn_vis_path / f'step_{self.step:05d}.jpg'), side_by_side_bgr)

      rgb_pos = t_u.crop_array(self.config, rgb_pos)

      # Switch to pytorch channel first order
      rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
      rgb.append(rgb_pos)
    rgb = np.concatenate(rgb, axis=1)
    rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)

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
    result['command_value'] = int(self.commands[-2])

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
    if speed > 1.0:
      self._meta_recovery_motion_frames += 1
      if self._meta_recovery_motion_frames >= self._meta_recovery_motion_frames_required:
        self._meta_recovery_ever_moving = True

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

    stop_for_stop_sign = False
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

    pred_checkpoints_for_control = None
    if self.config.use_controller_input_prediction and pred_checkpoints:
      pred_checkpoints_for_control = torch.stack(pred_checkpoints, dim=0).mean(dim=0).detach().cpu().numpy()

    # ----------------------------------------------------------------
    # Meta-action VLA: Qwen3-VL with annotated multimodal input.
    #
    # The VLA now receives the same enriched signal shape as the Qwen
    # experiment: front RGB + corridor/BEV overlay, object table, rule
    # candidates, path summary, and recent TTC history.
    # ----------------------------------------------------------------
    objects, primary_object_id = build_object_context(
        bbs_vehicle_coordinate_system,
        lateral_thresh=self._meta_lateral_thresh,
        max_objects=self._meta_max_objects,
        max_primary_distance=self._meta_front_max_distance,
    )
    object_summary = summarize_objects(objects)
    rule_summary = summarize_bbox_classes(
        bbs_vehicle_coordinate_system,
        lateral_thresh=self._meta_lateral_thresh,
    )
    self._meta_object_count = len(objects)
    self._meta_primary_object_id = primary_object_id
    self._meta_object_summary = object_summary
    self._meta_rule_summary = rule_summary
    front_distance = get_front_distance(
        bbs_vehicle_coordinate_system,
        lateral_thresh=self._meta_lateral_thresh,
        max_distance=self._meta_front_max_distance,
    )
    _bbox_ttc = self._compute_ttc(bbs_vehicle_coordinate_system, speed)
    self._meta_planner_braking_cue = self._has_meta_planner_braking_cue(
        speed,
        pred_target_speed_scalar,
    )
    _sensitive_ttc, _ttc_source = compute_sensitive_ttc(
        front_distance,
        speed,
        tfpp_target_speed=pred_target_speed_scalar,
        control_brake=0.0,
        has_front_objects=object_summary.get('front_object_count', 0) > 0,
    )
    if _bbox_ttc < _sensitive_ttc:
      _ttc = _bbox_ttc
      _ttc_source = 'bbox'
    else:
      _ttc = _sensitive_ttc
    self._meta_ttc_source = _ttc_source
    _ttc_danger = _ttc < self.meta_action.ttc_threshold
    _speed_before_vlm = pred_target_speed_scalar
    command_value = int(tick_data.get('command_value', self.commands[-2]))
    gap_summary = summarize_gap_context(
        objects,
        path_points=pred_checkpoints_for_control,
        command_value=command_value,
        route_name=self._meta_route_name,
        lateral_thresh=self._meta_lateral_thresh,
        max_forward_m=self._meta_gap_lookahead,
        max_side_m=self._meta_gap_side_distance,
    )
    self._meta_gap_summary = gap_summary

    self._meta_ttc_history.append({
        'step': self.step,
        'ego_speed': speed,
        'front_distance': front_distance,
        'ttc': _ttc,
        'source': _ttc_source,
    })

    self._meta_vlm_called_this_step = False
    self._meta_vlm_trigger_this_step = 'none'

    traffic_light_candidate = rule_summary.get('bbox_traffic_light_count', 0) > 0
    stop_sign_candidate = (
        self._meta_rule_enable_stop_sign
        and rule_summary.get('bbox_stop_sign_count', 0) > 0
    )
    rule_candidate = (
        traffic_light_candidate
        or stop_sign_candidate
    )
    rule_hold_probe = (
        self._meta_rule_hold_active
        and self.step % self._meta_rule_hold_poll_steps == 0
    )
    rule_probe = rule_candidate or rule_hold_probe
    gap_probe = (
        self._meta_gap_enabled
        and self._is_meta_gap_route_allowed(gap_summary)
        and self._has_meta_gap_evidence(gap_summary)
        and (
            bool(gap_summary.get('should_probe', False))
            or (
                self._meta_gap_visual_probe
                and bool(gap_summary.get('visual_probe_suggested', False))
            )
        )
        and not self._meta_rule_hold_active
    )
    speed_probe = (
        _ttc_danger
        and speed > 0.5
        and (
            object_summary.get('front_object_count', 0) > 0
            or self._meta_planner_braking_cue
        )
    )

    prompt_mode = None
    if rule_probe:
      prompt_mode = 'traffic_rule'
    elif gap_probe:
      prompt_mode = 'gap'
    elif speed_probe:
      prompt_mode = 'speed'

    qwen_image = None
    if prompt_mode is not None and self._meta_rgb_np is not None:
      qwen_image = annotate_qwen_image(
          self._meta_rgb_np,
          objects,
          primary_object_id,
          ego_speed=speed,
          front_distance=front_distance,
          ttc=_ttc,
          tfpp_target_speed=pred_target_speed_scalar,
          rule_summary=rule_summary,
          lateral_thresh=self._meta_lateral_thresh,
          include_bev=(prompt_mode == 'gap'),
      )
      self._meta_last_qwen_image = qwen_image

    if prompt_mode is not None and qwen_image is not None:
      if rule_hold_probe:
        trigger = 'traffic_rule_hold'
      elif rule_candidate:
        trigger = 'traffic_rule'
      elif gap_probe:
        trigger = 'intersection_gap'
      else:
        trigger = 'speed_ttc'
      self._meta_vlm_trigger_this_step = trigger
      context = self._build_meta_prompt_context(
          prompt_mode=prompt_mode,
          speed=speed,
          front_distance=front_distance,
          ttc=_ttc,
          ttc_source=_ttc_source,
          tfpp_target_speed=pred_target_speed_scalar,
          objects=objects,
          rule_summary=rule_summary,
          gap_summary=gap_summary,
          pred_checkpoints_for_control=pred_checkpoints_for_control,
      )
      context['trigger'] = trigger
      submitted = self.meta_action.request_guidance(
          qwen_image,
          self.step,
          context=context,
          prompt_mode=prompt_mode,
      )
      self._meta_vlm_called_this_step = submitted
      if submitted:
        self._save_meta_input_image(qwen_image)

    latest_result = self.meta_action.get_latest_result()
    if latest_result.get('prompt_mode') == 'traffic_rule':
      self._meta_rule_result = latest_result
    elif latest_result.get('prompt_mode') == 'gap':
      self._meta_gap_result = latest_result
    else:
      self._meta_speed_result = latest_result

    self._meta_rule_active = self._update_meta_rule_hold_state(speed)
    if not self._meta_rule_active:
      self._meta_rule_active = self._should_apply_meta_rule_result(speed)
    self._meta_rule_scale = (
        0.0
        if self._meta_rule_hold_active
        else float(self._meta_rule_result.get('rule_speed_scale', 1.0))
    )
    self._meta_speed_active = self._should_apply_meta_speed_result(_ttc_danger)
    self._meta_gap_active = self._should_apply_meta_gap_result(gap_summary)
    self._meta_gap_scale = (
        float(self._meta_gap_result.get('speed_scale', 1.0))
        if self._meta_gap_active
        else 1.0
    )
    gap_decision = str(self._meta_gap_result.get('gap_decision', 'unknown')).lower()
    gap_candidate_count = int(gap_summary.get('gap_candidate_count', 0) or 0)
    if self._meta_gap_active and gap_decision == 'stop':
      self._meta_gap_stop_hold_steps += 1
    else:
      self._meta_gap_stop_hold_steps = 0
    if self._meta_gap_active and gap_candidate_count <= 0:
      self._meta_gap_no_candidate_steps += 1
    else:
      self._meta_gap_no_candidate_steps = 0
    if (
        self._meta_gap_active
        and self._meta_gap_stop_release_steps > 0
        and self._meta_gap_stop_hold_steps >= self._meta_gap_stop_release_steps
    ):
      self._meta_gap_scale = max(self._meta_gap_scale, self._meta_gap_stop_release_scale)
      self._meta_gap_result['speed_scale'] = self._meta_gap_scale
      self._meta_gap_result['action'] = 'cautious_proceed'
      self._meta_gap_result['gap_decision'] = 'cautious_go'
      print(
          '[MetaActionVLA] gap stop soften: '
          f'step={self.step} hold={self._meta_gap_stop_hold_steps} '
          f'scale={self._meta_gap_scale:.2f}'
      )
    if (
        self._meta_gap_active
        and self._meta_gap_no_candidate_release_steps > 0
        and self._meta_gap_no_candidate_steps >= self._meta_gap_no_candidate_release_steps
    ):
      self._meta_gap_scale = max(self._meta_gap_scale, self._meta_gap_stop_release_scale)
      self._meta_gap_result['speed_scale'] = self._meta_gap_scale
      self._meta_gap_result['action'] = 'cautious_proceed'
      self._meta_gap_result['gap_decision'] = 'cautious_go'
      print(
          '[MetaActionVLA] gap deadlock soften: '
          f'step={self.step} hold={self._meta_gap_stop_hold_steps} '
          f'no_candidate={self._meta_gap_no_candidate_steps} scale={self._meta_gap_scale:.2f}'
      )

    self._meta_tl_prestop_active = self._should_meta_tl_prestop(rule_summary)
    self._meta_turn_caution_active = self._should_meta_turn_caution(gap_summary)
    self._meta_recovery_active = self._should_meta_recovery(speed, stop_for_stop_sign)
    if self.config.use_controller_input_prediction:
      if self._meta_rule_active:
        _multiplier = min(self._meta_rule_scale, 1.0)
      elif self._meta_gap_active:
        _multiplier = min(self._meta_gap_scale, 1.0)
      elif self._meta_speed_active:
        _multiplier = float(self._meta_speed_result.get('speed_scale', 1.0))
      else:
        _multiplier = 1.0
      if self._meta_tl_prestop_active:
        _multiplier = min(_multiplier, self._meta_tl_prestop_scale)
      pred_target_speed_scalar = pred_target_speed_scalar * _multiplier
      if self._meta_turn_caution_active and not self._meta_recovery_active:
        pred_target_speed_scalar = min(pred_target_speed_scalar, self._meta_turn_caution_speed)
      if self._meta_recovery_active:
        pred_target_speed_scalar = max(pred_target_speed_scalar, self._meta_recovery_target_speed)
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
          pred_bb=bbs_vehicle_coordinate_system,
          gt_speed=gt_velocity,
          gt_wp=pred_wp_1,
          wp_selected=wp_selected)

    if self.config.inference_direct_controller and self.config.use_controller_input_prediction:
      if pred_checkpoints_for_control is None:
        pred_checkpoints = torch.stack(pred_checkpoints, dim=0).mean(dim=0).detach().cpu().numpy()
      else:
        pred_checkpoints = pred_checkpoints_for_control
      steer, throttle, brake = self.nets[0].control_pid_direct(pred_checkpoints, pred_target_speed_scalar, gt_velocity)

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

    self._meta_escape_reverse_active = self._should_meta_escape_reverse()
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

    if self._meta_rule_hold_active:
      throttle = 0.0
      brake = True

    if self._meta_escape_reverse_active:
      throttle = max(float(throttle), self._meta_escape_throttle)
      brake = False
      steer = -float(steer)
    elif self._meta_recovery_active and not self._meta_rule_hold_active and not self._is_active_meta_signal_stop():
      throttle = max(float(throttle), self._meta_recovery_throttle)
      brake = False
      if self.force_move > 0:
        self.force_move -= 1

    if self.stop_after_meter > 0 and self.meters_travelled > self.stop_after_meter:
      print(f'Stopping after {self.stop_after_meter} meters.')
      throttle = 0.0
      brake = True

    control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
    if self._meta_escape_reverse_active:
      control.reverse = True

    self._save_qwen_dashboard(
        pred_checkpoints=pred_checkpoints if self.config.use_controller_input_prediction else None,
        bbs=bbs_vehicle_coordinate_system,
        ttc=_ttc,
        ttc_danger=_ttc_danger,
        multiplier=_multiplier,
        speed_before=_speed_before_vlm,
        speed_after=pred_target_speed_scalar,
        front_distance=front_distance,
        ego_speed=speed,
        control=control,
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
            print(f'[MetaActionVLA] rule_hold release: green confirmed ({hold_type})')
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
                '[MetaActionVLA] rule_hold no-stop vote '
                f'{self._meta_rule_hold_no_stop_votes}/{self._meta_rule_hold_no_stop_votes_required} '
                f'tl={result.get("traffic_light_state", "unknown")} step={no_stop_key}'
            )
          if (
              ego_speed < 0.5
              and self._meta_rule_hold_no_stop_votes >= self._meta_rule_hold_no_stop_votes_required
          ):
            print(f'[MetaActionVLA] rule_hold release: no-stop votes ({hold_type})')
            self._meta_rule_hold_post_release_step = self.step
            self._reset_meta_rule_hold()
            return False

      if (
          self._meta_rule_hold_max_steps > 0
          and self._meta_rule_hold_since_step >= 0
          and self.step - self._meta_rule_hold_since_step > self._meta_rule_hold_max_steps
      ):
        print(f'[MetaActionVLA] rule_hold max-step release: {self._meta_rule_hold_type}')
        self._meta_rule_hold_post_release_step = self.step
        self._reset_meta_rule_hold()
        return False

      if (
          self._meta_rule_hold_safety_steps > 0
          and self._meta_rule_hold_since_step >= 0
          and self.step - self._meta_rule_hold_since_step > self._meta_rule_hold_safety_steps
      ):
        print(f'[MetaActionVLA] rule_hold safety release: {self._meta_rule_hold_type}')
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
    print(f'[MetaActionVLA] rule_hold activate: {rule_type} step={self.step}')
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

  def _is_meta_gap_route_allowed(self, gap_summary: dict) -> bool:
    if not self._meta_gap_route_only:
      return True
    route_lower = str(self._meta_route_name or '').lower()
    return 'nonsignalizedjunction' in route_lower or bool(gap_summary.get('route_junction', False))

  @staticmethod
  def _has_meta_planner_braking_cue(
      ego_speed: float,
      tfpp_target_speed: float,
  ) -> bool:
    try:
      ego = float(ego_speed)
      target = float(tfpp_target_speed)
    except (TypeError, ValueError):
      return False
    if not np.isfinite(ego) or ego < 1.0 or not np.isfinite(target):
      return False
    target = max(0.0, target)
    speed_drop = ego - target
    if target <= 0.2 and ego >= 2.0:
      return True
    if speed_drop >= 3.0 and target <= ego * 0.65:
      return True
    return False

  def _should_apply_meta_speed_result(self, ttc_danger: bool) -> bool:
    if not self._meta_speed_semantic_enabled:
      return False
    if not ttc_danger:
      return False
    if not bool(self._meta_speed_result.get('intervene', False)):
      return False
    if not bool(self._meta_speed_result.get('path_blocked', False)):
      return False
    try:
      requested_scale = float(self._meta_speed_result.get('speed_scale', 1.0))
      request_step = int(self._meta_speed_result.get('request_step'))
    except (TypeError, ValueError):
      return False
    if requested_scale >= 0.95:
      return False
    if self._meta_ttc_source != 'bbox' and not self._meta_planner_braking_cue:
      return False
    risk_level = str(self._meta_speed_result.get('risk_level', 'low')).lower()
    if risk_level not in {'medium', 'high', 'critical'}:
      return False
    age = self.step - request_step
    return 0 <= age <= self._meta_speed_result_max_age

  def _meta_gap_immediate_conflict(self, gap_summary: dict) -> bool:
    if int(gap_summary.get('immediate_gap_candidate_count', 0) or 0) > 0:
      return True
    try:
      nearest_x = float(gap_summary.get('nearest_gap_x', float('nan')))
      nearest_y = float(gap_summary.get('nearest_gap_y', float('nan')))
    except (TypeError, ValueError):
      return False
    return (
        np.isfinite(nearest_x)
        and np.isfinite(nearest_y)
        and -2.0 <= nearest_x <= self._meta_gap_immediate_x
        and abs(nearest_y) <= self._meta_gap_immediate_y
    )

  def _has_meta_gap_evidence(self, gap_summary: dict) -> bool:
    if int(gap_summary.get('gap_candidate_count', 0) or 0) <= 0:
      return False
    if self._meta_gap_immediate_conflict(gap_summary):
      return True
    return int(gap_summary.get('moving_gap_candidate_count', 0) or 0) > 0

  def _should_apply_meta_gap_result(self, gap_summary: dict) -> bool:
    """Apply a recent intersection-gap decision even when TTC is not low."""
    if not self._meta_gap_enabled:
      return False
    if not self._is_meta_gap_route_allowed(gap_summary):
      return False
    if not self._has_meta_gap_evidence(gap_summary):
      return False
    if self._meta_rule_active or self._meta_rule_hold_active:
      return False
    if not (
        bool(gap_summary.get('should_probe', False))
        or (
            self._meta_gap_visual_probe
            and bool(gap_summary.get('visual_probe_suggested', False))
        )
    ):
      return False

    result = self._meta_gap_result
    if result.get('prompt_mode') != 'gap':
      return False
    try:
      request_step = int(result.get('request_step'))
      scale = float(result.get('speed_scale', 1.0))
      confidence = float(result.get('gap_confidence', result.get('confidence', 0.0)))
    except (TypeError, ValueError):
      return False
    age = self.step - request_step
    if age < 0 or age > self._meta_gap_result_max_age:
      return False
    if confidence < self._meta_gap_confidence_threshold:
      return False
    if scale >= 0.95 and bool(result.get('clear_to_enter', False)):
      return False
    if not bool(result.get('intervene', scale < 0.95)):
      return False
    gap_decision = str(result.get('gap_decision', 'unknown')).lower()
    immediate_conflict = self._meta_gap_immediate_conflict(gap_summary)
    if gap_decision == 'stop' or scale <= 0.05:
      if not immediate_conflict or confidence < self._meta_gap_stop_confidence_threshold:
        result['speed_scale'] = max(scale, self._meta_gap_min_scale)
        result['action'] = 'cautious_proceed'
        result['gap_decision'] = 'cautious_go'
    elif not immediate_conflict and confidence < self._meta_gap_strong_confidence_threshold:
      result['speed_scale'] = max(scale, self._meta_gap_min_scale)
      result['action'] = 'cautious_proceed'
      result['gap_decision'] = 'cautious_go'
    return scale < 0.95

  def _should_meta_recovery(self, ego_speed: float, stop_for_stop_sign: bool = False) -> bool:
    """Let the agent crawl out after a long stop/collision unless a signal hold is active."""
    if not self._meta_recovery_enabled:
      return False
    if self._meta_recovery_requires_motion and not self._meta_recovery_ever_moving:
      return False
    if stop_for_stop_sign:
      self._meta_recovery_until_step = -1
      return False
    if self._meta_rule_hold_active or self._is_active_meta_signal_stop():
      self._meta_recovery_until_step = -1
      return False
    if self._meta_escape_reverse_active:
      return True
    if self._meta_recovery_until_step >= self.step:
      return True
    if ego_speed >= 0.2:
      self._meta_recovery_until_step = -1
      return False
    if self.stuck_detector < self._meta_recovery_stuck_steps:
      return False

    self._meta_recovery_until_step = self.step + self._meta_recovery_duration_steps
    print(
        '[MetaActionVLA] recovery crawl activate: '
        f'step={self.step} stuck={self.stuck_detector} '
        f'target={self._meta_recovery_target_speed:.2f} throttle={self._meta_recovery_throttle:.2f}'
    )
    return True

  def _should_meta_turn_caution(self, gap_summary: dict) -> bool:
    if not self._meta_turn_caution_enabled:
      return False
    command = str(gap_summary.get('command_name', 'unknown')).lower()
    if command in ('left', 'right'):
      return True
    try:
      path_lateral = float(gap_summary.get('path_lateral_max', float('nan')))
    except (TypeError, ValueError):
      path_lateral = float('nan')
    return np.isfinite(path_lateral) and path_lateral >= self._meta_turn_caution_lateral

  def _should_meta_escape_reverse(self) -> bool:
    if not self._meta_escape_reverse_enabled:
      return False
    if self._meta_rule_hold_active or self._is_active_meta_signal_stop():
      self._meta_escape_reverse_until_step = -1
      return False
    if self._meta_escape_reverse_until_step >= self.step:
      return True
    if not self._meta_recovery_active:
      return False
    if self.stuck_detector < self._meta_escape_stuck_steps:
      return False
    self._meta_escape_reverse_until_step = self.step + self._meta_escape_reverse_steps
    print(
        '[MetaActionVLA] escape reverse activate: '
        f'step={self.step} stuck={self.stuck_detector} '
        f'duration={self._meta_escape_reverse_steps} throttle={self._meta_escape_throttle:.2f}'
    )
    return True

  def _is_active_meta_signal_stop(self) -> bool:
    if not self._meta_rule_active:
      return False
    rule_type = str(self._meta_rule_result.get('rule_type', 'none')).lower()
    return rule_type in ('red_light', 'yellow_light')

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
    tl_state = str(self._meta_rule_result.get('traffic_light_state', 'unknown')).lower()
    if no_intervene and confidence >= self._meta_rule_confidence_threshold and 0 <= age <= self._meta_rule_result_max_age:
      return False
    return True

  def _build_meta_prompt_context(
      self,
      prompt_mode: str,
      speed: float,
      front_distance: float,
      ttc: float,
      ttc_source: str,
      tfpp_target_speed: float,
      objects,
      rule_summary: dict,
      gap_summary: dict,
      pred_checkpoints_for_control,
  ) -> dict:
    context = {
        'step': self.step,
        'ego_speed': float(speed),
        'front_distance': float(front_distance),
        'ttc': float(ttc),
        'ttc_source': ttc_source,
        'tfpp_target_speed': float(tfpp_target_speed),
    }
    if prompt_mode == 'traffic_rule':
      context['rule_context'] = format_rule_context(rule_summary)
    elif prompt_mode == 'gap':
      context['gap_context'] = format_gap_context(gap_summary)
      context['path_summary'] = format_path_summary(pred_checkpoints_for_control)
    else:
      front_focus = [
          obj for obj in objects
          if obj.get('same_lane', False) or obj.get('primary', False)
      ]
      if front_focus:
        context['object_table'] = format_object_table(front_focus[:2])
    return context

  def _compute_ttc(self, bbs, ego_speed: float) -> float:
    """
    TTC = bb[0] / ego_speed (ì ì°¨ ì ì§ ê°ì , ë³´ìì  ì¶ì )

    bb ì¢íê³: x=ì ë°©(ìì=ì), y=ì¸¡ë©´
    ì°¨ë/ë³´íì/ê¸´ê¸ì°¨(class 0,1,4)ë§ ê³ ë ¤.
    ì°¨ì  ë°(abs(y) > 2.0m) ê°ì²´ ì ì¸.

    Returns:
        TTC (ì´). ìí ìì¼ë©´ 999.0.
    """
    if ego_speed < 0.5 or bbs is None or len(bbs) == 0:
      return 999.0

    min_ttc = 999.0
    for bb in bbs:
      if len(bb) < 8:
        continue
      class_id = int(bb[7])
      if class_id not in (0, 1, 4):  # ì°¨ë, ë³´íì, ê¸´ê¸ì°¨ëë§
        continue
      x = float(bb[0])   # ì ë°© ê±°ë¦¬
      y = float(bb[1])   # ì¸¡ë°© ê±°ë¦¬
      if x < 0.5:         # íë°© ëë ëë¬´ ê°ê¹ì´ ê°ì²´ ì ì¸
        continue
      if abs(y) > 2.0:    # ì°¨ì  ë° ê°ì²´ ì ì¸
        continue
      ttc = x / ego_speed
      if ttc < min_ttc:
        min_ttc = ttc

    return min_ttc

  def _save_meta_input_image(self, image_rgb: np.ndarray) -> None:
    if self._meta_input_path is None:
      return
    try:
      out = self._meta_input_path / f'{self.step:05d}.png'
      cv2.imwrite(str(out), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    except Exception as exc:  # pylint: disable=broad-except
      print(f'[MetaActionVLA] input save failed step={self.step}: {exc}')

  def _save_qwen_dashboard(
      self,
      pred_checkpoints,
      bbs,
      ttc: float,
      ttc_danger: bool,
      multiplier: float,
      speed_before: float,
      speed_after: float,
      front_distance: float,
      ego_speed: float,
      control,
  ) -> None:
    if self._qwen_dashboard_render is None or self.save_path is None:
      return
    if self.step % self._qwen_dashboard_interval != 0:
      return
    image_rgb = self._meta_rgb_np
    if image_rgb is None:
      return

    try:
      if self._meta_rule_active:
        display_result = self._meta_rule_result
      elif self._meta_gap_active:
        display_result = self._meta_gap_result
      elif self._meta_speed_active:
        display_result = self._meta_speed_result
      else:
        display_result = self._meta_speed_result
      applied_scale = 0.0 if self._meta_rule_hold_active else float(multiplier)
      if self._meta_rule_active:
        applied_scale = min(applied_scale, self._meta_rule_scale)
      risk_level = display_result.get('risk_level', 'low')
      if self._meta_rule_active:
        risk_level = 'critical'
      display_action_id = int(display_result.get('action_id', 0) or 0)
      display_action = str(display_result.get('action', 'proceed'))
      if self._meta_recovery_active:
        display_action_id = 4
        display_action = 'reverse_escape' if self._meta_escape_reverse_active else 'crawl'
      elif (
          not self._meta_rule_active
          and not self._meta_gap_active
          and not self._meta_speed_active
          and applied_scale >= 0.95
      ):
        display_action_id = 0
        display_action = 'proceed'
      elif self._meta_gap_active and applied_scale >= 0.2 and display_action == 'stop':
        display_action_id = 3
        display_action = 'yield'

      frame_data = {
          'image': image_rgb,
          'step': self.step,
          'action_id': display_action_id,
          'action': display_action,
          'ego_speed': float(ego_speed),
          'front_distance': float(front_distance),
          'ttc': float(ttc),
          'ttc_threshold': self.meta_action.ttc_threshold,
          'risky': bool(ttc_danger or self._meta_rule_active or self._meta_gap_active),
          'speed_scale': float(applied_scale),
          'guard_scale': 1.0,
          'risk_level': risk_level,
          'reason': display_result.get('reason', display_result.get('rule_reason', '')),
          'intervene': bool(self._meta_rule_active or applied_scale < 0.95),
          'vlm_called': self._meta_vlm_called_this_step,
          'vlm_ready': True,
          'vlm_trigger': self._meta_vlm_trigger_this_step,
          'semantic_active': self._meta_speed_active,
          'ttc_source': self._meta_ttc_source,
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
          'turn_caution_active': self._meta_turn_caution_active,
          'recovery_active': self._meta_recovery_active,
          'escape_reverse_active': self._meta_escape_reverse_active,
          'gap_active': self._meta_gap_active,
          'gap_decision': str(self._meta_gap_result.get('gap_decision', 'unknown')),
          'gap_confidence': float(self._meta_gap_result.get('gap_confidence', 0.0)),
          'gap_clear_to_enter': bool(self._meta_gap_result.get('clear_to_enter', True)),
          'gap_cross_traffic': bool(self._meta_gap_result.get('cross_traffic', False)),
          'gap_candidate_count': int(self._meta_gap_summary.get('gap_candidate_count', 0) or 0),
          'gap_nearest_id': self._meta_gap_summary.get('nearest_gap_id'),
          'gap_nearest_x': self._meta_gap_summary.get('nearest_gap_x', float('nan')),
          'gap_nearest_y': self._meta_gap_summary.get('nearest_gap_y', float('nan')),
          'gap_command': str(self._meta_gap_summary.get('command_name', 'unknown')),
          'gap_near_junction': bool(self._meta_gap_summary.get('near_junction', False)),
          'qwen_intervene': bool(self._meta_speed_result.get('intervene', False)),
          'qwen_requested_scale': float(self._meta_speed_result.get('speed_scale', 1.0)),
          'qwen_path_blocked': bool(self._meta_speed_result.get('path_blocked', False)),
          'qwen_hazard_type': str(self._meta_speed_result.get('hazard_type', 'none')),
          'tfpp_target_speed': float(speed_before),
          'final_target_speed': float(speed_after),
          'pred_boxes': bbs,
          'object_count': self._meta_object_count,
          'primary_object_id': self._meta_primary_object_id,
          'same_lane_count': self._meta_object_summary.get('same_lane_count', 0),
          'front_object_count': self._meta_object_summary.get('front_object_count', 0),
          'nearest_object_id': self._meta_object_summary.get('nearest_object_id'),
          'nearest_object_x': self._meta_object_summary.get('nearest_object_x', float('nan')),
          'nearest_object_y': self._meta_object_summary.get('nearest_object_y', float('nan')),
          'bbox_vehicle_count': self._meta_rule_summary.get('bbox_vehicle_count', 0),
          'bbox_pedestrian_count': self._meta_rule_summary.get('bbox_pedestrian_count', 0),
          'bbox_traffic_light_count': self._meta_rule_summary.get('bbox_traffic_light_count', 0),
          'bbox_stop_sign_count': self._meta_rule_summary.get('bbox_stop_sign_count', 0),
          'bbox_emergency_vehicle_count': self._meta_rule_summary.get('bbox_emergency_vehicle_count', 0),
          'nearest_traffic_light_id': self._meta_rule_summary.get('nearest_traffic_light_id'),
          'nearest_traffic_light_x': self._meta_rule_summary.get('nearest_traffic_light_x', float('nan')),
          'nearest_traffic_light_y': self._meta_rule_summary.get('nearest_traffic_light_y', float('nan')),
          'nearest_stop_sign_id': self._meta_rule_summary.get('nearest_stop_sign_id'),
          'nearest_stop_sign_x': self._meta_rule_summary.get('nearest_stop_sign_x', float('nan')),
          'nearest_stop_sign_y': self._meta_rule_summary.get('nearest_stop_sign_y', float('nan')),
          'history_key': str(self.save_path),
          'reset_history': self.step <= self._qwen_dashboard_interval,
          'control_throttle': getattr(control, 'throttle', None),
          'control_brake': getattr(control, 'brake', None),
          'control_steer': getattr(control, 'steer', None),
      }
      save_path_png = pathlib.Path(self.save_path) / 'dashboard' / f'{self.step:05d}.png'
      self._qwen_dashboard_render(frame_data, save_path=str(save_path_png))
      if self._meta_dash_path is not None:
        meta_save_path_png = self._meta_dash_path / f'{self.step:05d}.png'
        meta_frame_data = dict(frame_data)
        meta_frame_data['history_key'] = str(self._meta_dash_path)
        self._qwen_dashboard_render(meta_frame_data, save_path=str(meta_save_path_png))
    except Exception as e:  # pylint: disable=broad-except
      print(f'[QwenDash] visualization error at step={self.step}: {e}')

  def _save_meta_dashboard(
      self,
      pred_checkpoints,
      bbs,
      ttc: float,
      ttc_danger: bool,
      multiplier: float,
      speed_before: float,
      speed_after: float,
  ) -> None:
    """ëìë³´ë ì´ë¯¸ì§ë¥¼ META_DASHBOARD_PATH/<route>/ ì ì ì¥."""
    if self._dashboard_vis is None or self._meta_dash_path is None:
      return
    if self._meta_rgb_np is None:
      return

    try:
      save_path_png = self._meta_dash_path / f'{self.step:05d}.png'

      # TTC â danger score (0~1): TTC=0 â 1.0, TTC=ttc_threshold â 1.0, TTC=6s â 0.5
      ttc_thresh = self.meta_action.ttc_threshold
      danger_score = float(np.clip(ttc_thresh / max(ttc, 0.1), 0.0, 1.0)) if ttc < 999.0 else 0.0

      action_name = self.meta_action.get_action_name()
      mode = f'VLM:{action_name} (x{multiplier:.1f})' if ttc_danger else 'TF++ active'

      frame_data = {
          'image': self._meta_rgb_np,
          'tfpp_waypoints': pred_checkpoints,
          'final_waypoints': pred_checkpoints,
          'danger_score': danger_score,
          'raw_danger_score': danger_score,
          'vlm_json': {
              'ttc': round(ttc, 2) if ttc < 999.0 else None,
              'meta_action': action_name,
              'multiplier': multiplier,
          },
          'pred_boxes': bbs,
          'history_key': str(self._meta_dash_path),
          'reset_history': self.step <= 5,
          'intervention': ttc_danger,
          'mode': mode,
          'vlm_ran': self.step == self.meta_action._last_trigger_step,
          'vlm_suggested_action': action_name,
          'final_selected_action': action_name if ttc_danger else 'proceed (TF++)',
          'danger_components': {'ttc': danger_score},
          'tfpp_speed_mps': speed_before,
          'vlm_speed_mps': speed_after if ttc_danger else None,
          'final_speed_mps': speed_after,
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

    del self.nets
    del self.config
    del self.metric_info


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
