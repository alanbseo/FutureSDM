# SDM 모델링 데이터 및 코드 구성 가이드

본 문서는 기후변화 및 토지이용 변화에 따른 종 분포 모델(Species Distribution Model, SDM) 구축과 관련된 전체 데이터 위치, 사용 용도 및 코드(스크립트)의 워크플로우를 정리한 문서입니다. 최신 모델 튜닝(Unconstrained) 및 전 지구적 온도 한계치 적용(Constrained) 파이프라인의 변경 사항이 반영되어 있습니다.

## 1. 사용되는 데이터 모음 및 사용 용도

### 1-1. 종 출현 데이터 (Presence Data)
종 분포 모델의 라벨(Target)로 사용되는 식물의 좌표 및 메타데이터입니다.
- **원본 위치**: `../Data/`
  - `Plant_metadata.xlsx`: 식물종의 메타데이터 (기후민감종, 멸종위기 등급 등)
  - `Plant_coordinates.csv`: 식물의 관측된 경도/위도 좌표 데이터
- **처리 후 저장**: `Plant_Spatial_Data_{Date}.gpkg` 형태의 공간 데이터 (GPKG)
- **용도**: MaxEnt 모델 학습 시 해당 종의 출현 지점 정보(Presence points)로 활용. 유효 좌표가 10개 미만인 종은 신뢰성 문제로 자동 제외됩니다.

### 1-2. 열적 한계 데이터 (Thermal Limits Data) [NEW]
- **데이터 위치**: `../Output/SDM/ThermalLimits/Species_ThermalLimits.json`
- **용도**: Constrained 모델 구동 시 사용됩니다. 각 종이 생존 가능한 최저 기온(Bio6) 및 최고 기온(Bio5)을 담고 있으며, 한국 내의 관측 기록뿐만 아니라 **GBIF(Global Biodiversity Information Facility)**의 전 세계 분포 데이터에서 추출한 생리적 극한 한계 온도를 우선적으로 기록해 둔 파일입니다.

### 1-3. 기후 및 환경 데이터 (Environmental Variables)
환경 예측 변수로 지형, 토양, 생물기후(Bioclim), 대기습도(VPD) 등이 사용됩니다.
- **원본 기후 데이터 (CMIP6 / WorldClim)**: `../Climate_Raw/`
- **전처리 후 변환 위치 (출력 디렉토리)**: `../Model/`
- **용도**: 종의 생태적 지위(Niche) 학습을 위한 연속형 공간 피처. 미래 SSP 시나리오별로 투영됩니다. (VPD, 일사량, 상대습도 등 파생 변수 포함)

### 1-4. 보조 모델링 데이터
- **통합 설정 파일**: `SDM/sdm_config.json`
  - 모델에 사용되는 환경 변수, 디렉토리 경로 정보, 하이퍼파라미터 설정(예: `clamp=False`, `beta_multipliers=1.5`) 및 핫/콜드 제약 적용 여부(`APPLY_HOT_CONSTRAINT`) 등을 전역적으로 관리합니다.

---

## 2. 코드 스크립트 위치 및 워크플로우

스크립트는 파이프라인 형태이며 `../Scripts/` 디렉토리에 위치해 있습니다.

### 단계 0: 데이터 전처리
* **`SDM_0_Preprocess_NIE_plant_data.py`**
  * **용도**: 원본 출현 데이터를 병합 및 정제하여 유효한 Point Geometry(GPKG) 공간 데이터를 생성합니다.
* **`SDM_1_2_PreprocessClimateData.py` & `SDM_1_3_CreateAnnualBIOCLIM.py`**
  * **용도**: 미래 기후 시나리오를 투영 가능하도록 포맷을 일치시키고(Reproject), 월별 데이터를 조합하여 19종의 생물기후(Bioclim) 변수를 자체 계산합니다.

### 단계 1: 모델 학습 (Model Tuning 적용)
* **`SDM_2_BuildSDM.py`**
  * **용도**: `elapid` 패키지를 이용해 MaxEnt 종 분포 모델을 학습시킵니다.
  * **핵심 업데이트**: 과거의 과적합(극단적 미래 기온에서 서식지가 무한 확장되는 오류)을 막기 위해 **Unconstrained 접근법(파라미터 튜닝)**이 적용되어 있습니다. 복잡한 Hinge 피처 대신 `Linear`, `Quadratic` 피처를 사용하여 부드러운 포물선 반응 곡선을 유도하고, `clamp=False`로 설정하여 미지의 기온 영역에서는 서식 적합도가 자연스럽게 감소(Extrapolate)하도록 최적화되었습니다.

### 단계 2: 모의 및 평가 (투영 및 마스킹)
* **`SDM_3_1_ProjectHabitatSuitability.py`**
  * **용도**: 훈련된 모델 및 미래 SSP 시나리오를 바탕으로 연속적인(0.0~1.0) 서식지 적합도를 투영합니다.
  * **핵심 업데이트 (Constrained Approach)**: 모델 설정(`APPLY_HOT_CONSTRAINT=True`)에 따라, `Species_ThermalLimits.json`을 읽어와 **GBIF 기반의 종별 절대 최고 기온 한계(Bio5 Max)**를 초과하는 미래 지역은 예측 확률과 무관하게 서식 확률을 강제로 0으로 마스킹하는 물리적 교정 작업이 수행됩니다.
* **`SDM_3_2_EvaluateHabitatSuitability.py`**
  * **용도**: TP10(10th Percentile Training Presence) 기법을 기본으로 하여 보수적인 임계값을 산출하고, 연속형 적합도를 **서식 가능(1) / 불가(0) 이진 맵(Binary map)**으로 변환합니다. 여기서 Gain, Loss, Stable 구역이 도출됩니다.

### 단계 3: 통계 및 요약 시각화
* **`SDM_4_SummaryStats.py`**
  * **용도**: 변환된 이진 맵을 바탕으로 각 시나리오별 총 면적 통계를 계산하여 `SDM_Analysis_AllSpecies_MaxEnt_[Unconstrained/Constrained].csv` 파일 및 시나리오 비교 승패 매트릭스 자료를 출력합니다.
* **`SDM_5_GroupRichnessMaps.py`**
  * **용도**: 멸종위기종, 민감종 등 그룹별 풍부도(Richness) 맵을 생성합니다.
  * **핵심 업데이트**: **[2020년 현재 풍부도]**, **[2050년 미래 풍부도]**, **[소멸(Loss) 풍부도]** 패널 3개로 출력되며, 직관적인 비교를 위해 2020년과 2050년 맵이 동일한 색상 스킴(`YlGn`) 및 시나리오 통합 스케일(`vmax`)을 공유하도록 렌더링됩니다.
* **`SDM_Plot_Aggregated_ResponseCurves.py`**
  * **용도**: 튜닝된 MaxEnt 모델의 환경 변수별 종 출현 확률(반응 곡선)을 개별 종 및 그룹별 평균(신뢰구간 포함)으로 도식화합니다.

### 보조 워크플로우 문서
* **`run_sdm_pipeline.md`** 경로: `.agents/workflows/run_sdm_pipeline.md`
  * **용도**: SDM 2단계(학습)부터 5단계(시각화)까지 터미널 명령어 라인으로 한 번에 자동 실행할 수 있도록 통합된 실행 가이드입니다.
