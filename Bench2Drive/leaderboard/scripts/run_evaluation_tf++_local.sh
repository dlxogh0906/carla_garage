export CARLA_ROOT=/mnt/2/carla
export WORK_DIR=/mnt/2/carla_garage/Bench2Drive
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:/mnt/2/carla_garage/team_code
export PYTHONPATH=$PYTHONPATH:/mnt/2/carla_garage/src
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# ---- 이미지 보정 스위치 ----
# 켜기:  아래 줄 주석 해제
# 끄기:  아래 줄 주석 처리 (#)
export GARAGE_EXT_CONFIG=/mnt/2/carla_garage/configs/experiments/classic_enhance.yaml

#!/bin/bash
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=${WORK_DIR}/leaderboard/data/bench2drive220
TEAM_AGENT=/mnt/2/carla_garage/src/garage_ext/agents/ext_sensor_agent.py
# Must set YOUR_CKPT_PATH to your actual checkpoint directory
TEAM_CONFIG=/mnt/2/pretrained_models/all_towns
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220_image_enhancer_0417   # 파일명 수정 ->  평가 .json(DS 나오는거)
PLANNER_TYPE=traj
ALGO=tfpp
SAVE_PATH=/mnt/2/carla_metric_result/carla_viz #시각화 경로

# ---- 시각화 스위치 ----
# 켜기:  아래 줄 주석 해제
# 끄기:  아래 줄 주석 처리 (#)
export DEBUG_CHALLENGE=1

if [ ! -d "${ALGO}_b2d_${PLANNER_TYPE}" ]; then
    mkdir ${ALGO}_b2d_${PLANNER_TYPE}
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} created. \033[0m"
else
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} already exists. \033[0m"
fi

# Check if the split_xml script needs to be executed
if [ ! -f "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag" ]; then
    echo -e "****************************\033[33m Attention \033[0m ****************************"
    echo -e "\033[33m Running split_xml.py \033[0m"
    TASK_NUM=4 # 8*H100, 1 task per gpu
    python ${WORK_DIR}/tools/split_xml.py $BASE_ROUTES $TASK_NUM $ALGO $PLANNER_TYPE
    touch "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
    echo -e "\033[32m Splitting complete. Flag file created. \033[0m"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi

echo -e "**************\033[36m Please Manually adjust GPU or TASK_ID \033[0m **************"
# Example, 8*H100, 1 task per gpu
GPU_RANK_LIST=(0 0 0)
TASK_LIST=(1 2 3)
echo -e "\033[32m GPU_RANK_LIST: $GPU_RANK_LIST \033[0m"
echo -e "\033[32m TASK_LIST: $TASK_LIST \033[0m"
echo -e "***********************************************************************************"

cd ${WORK_DIR}

length=${#GPU_RANK_LIST[@]}
for ((i=0; i<$length; i++ )); do
      PORT=$((BASE_PORT + i * 150))
      TM_PORT=$((BASE_TM_PORT + i * 150))
      ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
      CHECKPOINT_ENDPOINT="/mnt/2/carla_metric_result/${ALGO}_b2d_${PLANNER_TYPE}/${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
      mkdir -p "/mnt/2/carla_metric_result/${ALGO}_b2d_${PLANNER_TYPE}"
      GPU_RANK=${GPU_RANK_LIST[$i]}
      echo -e "\033[32m ALGO: $ALGO \033[0m"
      echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
      echo -e "\033[32m TASK_ID: $i \033[0m"
      echo -e "\033[32m PORT: $PORT \033[0m"
      echo -e "\033[32m TM_PORT: $TM_PORT \033[0m"
      echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
      echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
      echo -e "\033[32m bash ${WORK_DIR}/leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
      echo -e "***********************************************************************************"
      bash -e ${WORK_DIR}/leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > /mnt/2/carla_metric_result/${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.log &
      sleep 5
done
wait