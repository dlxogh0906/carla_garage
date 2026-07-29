# Classic CV Image Enhancement Module

이 폴더는 현재 TF++ 이미지 보정 실험에서 쓰는 `classic_cv` 보정 코드를 따로 모아둔 스냅샷이다. 실제 평가 실행은 원본 경로의 모듈을 사용한다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `classic_cv_enhancer.py` | 독립 실행/검토용 `ClassicCVEnhancer` 코드 |
| `base.py` | enhancer interface 설명용 protocol |
| `classic_enhance.yaml` | image-only 실험 설정 예시 |
| `README.md` | 코드 설명 문서 |

## 원본 실행 경로

실제 CARLA 평가에서 로드되는 원본 파일은 다음이다.

```text
carla_garage/src/garage_ext/modules/image_enhancer/classic.py
carla_garage/configs/experiments/classic_enhance.yaml
carla_garage/src/garage_ext/agents/ext_sensor_agent.py
carla_garage/run_tfpp_enhance_dev10.sh
```

평가 스크립트는 `GARAGE_EXT_CONFIG`로 `classic_enhance.yaml`을 넘긴다. 이 설정은 `image_enhancer=classic_cv`만 켜고, `vlm=null`, `risk=noop`, `safety=noop`으로 둔다. 따라서 Qwen/VLM 없이 TF++ 입력 이미지 보정만 적용된다.

## 입력과 출력

`ClassicCVEnhancer.enhance()`는 BGR `uint8` 이미지를 입력으로 받고, 같은 shape/dtype의 BGR `uint8` 이미지를 반환한다.

```python
from classic_cv_enhancer import ClassicCVEnhancer

enhancer = ClassicCVEnhancer()
enhanced_bgr = enhancer.enhance(frame_bgr)
```

CARLA sensor frame은 `frame[:, :, :3]`의 BGR 채널만 보정하고, 보정 결과를 다시 원래 frame에 넣는다.

## 처리 흐름

```text
input BGR frame
  -> frame statistics 분석
  -> mode 선택: low_light / over_exposed / haze / blurry / normal
  -> mode별 보정 pipeline 적용
  -> enhanced BGR frame 반환
```

## 모드 판별 기준

`_analyze()`는 다음 통계를 계산한다.

| 통계 | 의미 |
|---|---|
| `bm` | grayscale 평균 밝기 |
| `p10`, `p90` | 어두운 영역과 밝은 영역 percentile |
| `contrast` | grayscale 표준편차 |
| `sharpness` | Laplacian variance, 흐림 정도 |
| `sat_mean` | HSV saturation 평균 |
| `dc_mean` | dark channel 기반 haze 힌트 |

각 mode score가 `0.35` 이상이면 가장 높은 score의 mode를 선택하고, 아니면 `normal`로 둔다.

## 모드별 보정

| mode | 적용 보정 |
|---|---|
| `low_light` | gamma 밝기 보정, CLAHE, 약한 unsharp |
| `over_exposed` | gamma 어둡게 조정, highlight 압축, contrast stretch |
| `haze` | contrast stretch, CLAHE, saturation 증가, unsharp |
| `blurry` | unsharp sharpening |
| `normal` | 원본 copy 반환 |

## TF++ 실험에서의 위치

`ExtSensorAgent.run_step()`에서 upstream TF++를 호출하기 전에 이미지 보정을 적용한다.

```text
CARLA rgb_front
  -> classic_cv enhancer
  -> TF++ SensorAgent.run_step()
  -> control output
```

현재 `run_tfpp_enhance_dev10.sh`는 시각화 확인을 위해 rear camera도 compare-only로 추가할 수 있다. 이 rear image는 비교 PNG 저장용이며, TF++ 모델 입력에는 사용하지 않는다.

비교 이미지는 다음 형태로 저장된다.

```text
[front original][front enhanced]
[rear  original][rear  enhanced]
```

저장 위치:

```text
/mnt/2/carla_metric_result/tfpp_enhance_dev10/viz/<route>/enhance_compare/
```

## 실행

dev10 이미지 보정 only 실험:

```bash
/mnt/2/carla_garage/run_tfpp_enhance_dev10.sh
```

기본 출력:

```text
/mnt/2/carla_metric_result/tfpp_enhance_dev10/eval.json
/mnt/2/carla_metric_result/tfpp_enhance_dev10/eval.log
/mnt/2/carla_metric_result/tfpp_enhance_dev10/viz/
```

