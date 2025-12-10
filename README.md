# 📊 4-Day Datathon Project



## 🚀 프로젝트 개요

**프로젝트 주제:** Mercari Price Suggestion Challenge 및 여러 e-commerce 데이터셋 분석을 통한 가격 예측 및 데이터 인사이트 발굴

**프로젝트 기간:** 4일

**프로젝트 목표:**

* 주어진 데이터셋을 분석하고 EDA 및 시각화를 통해 주요 인사이트 도출
* 머신러닝을 활용한 가격 예측 모델 구축
* 팀 단위로 가설 설정 → 검증 → 모델링 → 발표까지의 전체 데이터 프로젝트 사이클 경험

---

## 👥 역할 분담

| 역할                        | 담당자      | 업무 내용                        |
| ------------------------- | -------- | ---------------------------- |
| **PM / 팀 리더**             | 김지수      | 일정 관리, 팀 의견 조율, 발표자료 최종 취합   |
| **데이터 전처리**               | 김지수, 김영서 | 데이터 클리닝, 결측치/이상치 처리, 테이블 병합  |
| **EDA & 시각화 담당**          | 최재우      | 탐색적 데이터 분석, 다양한 시각화, 인사이트 도출 |
| **모델링(Machine Learning)** | 임수명, 서민영 | ML 모델 구현, 성능 평가, 하이퍼파라미터 튜닝  |

---

## 📁 데이터셋

### 💼 사용된 데이터

* **Mercari Price Suggestion Challenge Dataset**

  * 제품명, 카테고리, 브랜드, 설명 등을 기반으로 중고 가격을 예측하는 Kaggle 데이터셋


---

## 🧭 프로젝트 선택 배경

### 🔥 1순위: Brazilian E-Commerce Dataset

* 실무와 가장 유사한 복잡한 구조
* 고객 여정, 배송, 결제 등 다양한 분석 관점 제공

### ⭐ 2순위: Mercari Price Suggestion (선정 프로젝트)

* 명확한 목표(가격 예측)
* NLP + Tabular 데이터 혼합으로 학습 효과 큼

### 📱 3순위: Google Play Store Apps

* 난이도 낮고 안정적
* 단, 차별화된 문제 정의가 어려움

---

## 📆 Day별 수행 내용

### **Day 1 – 데이터 이해 및 주제 선정**

* 팀 아이스브레이킹 및 역할 분담
* 데이터 구조 파악 (컬럼별 의미 확인)
* 테이블 간 연관성 분석
* 분석 방향 / 가설 논의

### **Day 2 – 전처리 & Feature Engineering**

* 결측치 처리 전략 수립 및 실험
* 이상치 탐지 및 제거/대체
* 카테고리 변수 정리, 텍스트 전처리
* 모델링을 위한 Feature 생성

### **Day 3 – EDA & 인사이트 도출**

* 변수 간 상관관계 분석
* 가격 분포 및 텍스트 길이·브랜드·카테고리 영향 분석
* 팀 가설 검증을 위한 시각화 수행

### **Day 4 – 발표자료 제작 및 최종 제출**

* 발표용 슬라이드 제작 및 시각화 정리
* GitHub 업로드 및 코드 정리
* 발표 및 회고 진행

<details>
  <summary>Day 1 – 데이터 이해 및 주제 선정</summary>
  Day 1 내용 …
</details>

<details>
  <summary>Day 2 – 전처리 & Feature Engineering</summary>
  Day 2 내용 …
</details>

<details>
  <summary>Day 3 – EDA & 인사이트 도출</summary>
  Day 3 내용 …
</details>

<details>
  <summary>Day 4 – 발표자료 제작 및 최종 제출</summary>
  Day 4 내용 …
</details>

---

## 🔍 진행한 EDA 예시 (서술용)

* 가격의 분포가 극단적으로 치우쳐 있어 로그 변환 필요
* 브랜드 정보가 가격 예측에 큰 영향을 미침
* 텍스트 설명의 길이가 가격과 일정 수준의 양의 상관관계 존재
* 카테고리 정보가 매우 다단계 구조 → Grouping 필요

---

## 🤖 모델링

* 기본 모델: Linear Regression, RandomForestRegressor, LightGBM
* 성능 평가 지표: RMSE
* 텍스트 설명을 TF-IDF 기반으로 벡터화 후 Tabular feature와 결합
* Hyperparameter tuning (GridSearch / Optuna 등 시도 가능)

---

## 📊 결과

* Feature engineering 및 텍스트 반영 시 성능 향상 확인
* 카테고리/브랜드가 주요 변수로 작용
* 텍스트 정보 활용의 중요성 검증

---

## 🎤 발표 자료

* 분석 과정 및 인사이트 중심 구조로 구성
* 모델 평가 결과 및 한계점 공유
* 향후 개선 방향 제시

---

## 📦 Repository 구조 예시

```plaintext
├── data/                  # 데이터셋
├── notebooks/             # 분석 및 모델링 과정 ipynb 파일
├── src/                   # 전처리 / 모델 코드 분리 가능
├── images/                # 그래프, 시각화 이미지
├── README.md              # 프로젝트 설명 문서
└── presentation.pdf       # 발표 자료
```

---

## ✨ 회고

* 짧은 시간 동안 데이터 분석 전체 사이클을 경험할 수 있었음
* 역할 분담이 가능해 빠르게 실전 프로젝트 경험 확보
* 텍스트 데이터 다루는 능력 향상
* 비즈니스 관점에서 인사이트 도출의 중요성 깨달음

---

## 📮 Contact

필요한 내용이나 프로젝트 상세 설명 요청은 언제든지 문의해주세요!
