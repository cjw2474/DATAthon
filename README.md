# 지역 기반 고객 구매 패턴 분석

## 1) 프로젝트 소개

* Brazilian E-Commerce 데이터를 활용하여 지역별 고객 구매 행동과 소비패턴이 영향을 미치는지 분석
* 단순 현황 분석을 넘어, 지역 맞춤형 마케팅 전략 인사이트를 제안

## 2) 프로젝트 목표

* 주어진 데이터셋을 분석하고 EDA 및 시각화를 통해 주요 인사이트 도출
* 머신러닝을 활용한 가격 예측 모델 구축
* 팀 단위로 가설 설정 → 검증 → 모델링 → 발표까지의 전체 데이터 프로젝트 사이클 경험


## 3) 데이터셋

<img width="826" height="551" alt="image" src="https://github.com/user-attachments/assets/46489343-163b-4843-87d5-a67dc82bf397" />

### 데이터 출처: 브라질 Olist 이커머스 플랫폼 공개 데이터 (2016-2018년)
### 데이터 규모: 약 10만 건의 주문 정보
### 데이터 구조: 8개 테이블로 구성된 관계형 데이터셋
  * 핵심 테이블: 주문, 고객, 상품,결제, 리뷰 등
  * 연결 관계: order_id, product_id 등 고유 ID로 연결
  * 핵심 테이블: olist_customers_dataset를 중심으로 시작
### 순차적 합병:
  * order_id로 주문-상품, 주문-결제, 주문-리뷰 정보 연결
  * customer_id로 고객 정보 연결
  * seller_id로 판매자 정보 연결
  * product_id로 상품 정보 연결
#### 불필요한 컬럼 제거: 분석에 필요 없는 컬럼들(예: 상품 설명 길이, 리뷰 제목 등)을 삭제
#### 결측치(NaN) 처리: 주요 컬럼의 빈 값을 확인하고 결측치 개수가 많은 경우 다른 값을 대체(-99, Unkown)
#### 주문 상태 필터링: 'canceled''된 주문만 남겨 취소된 거래 데이터 확보


## 4) 가설 설정

#### 가설1
 * 브라질은 지리적으로 국토가 광대해 다양한 기후대가 분포하며, 지역별로 상이한 생활 방식이 형성되어있다. 따라서, 지역마다 서로 다른 고객 유형과 소비 형태가 나타날 것이다. 즉, 5개 지역의 군집 분석 결과가 각각 다를 것이다.

#### 가설2
 * 평균 구매 금액이 높은 지역과 낮은 지역은 특정 권역에 집중되어 있으며, 이들 간의 차이가 뚜렷하게 나타날 것이다.

#### 가설3
 * 대부분의 지역에서 저녁 6시부터 8시 사이에 구매 비율이 가장 높게 나타나고, 반대로 새벽 12시부터 5시 사이에는 구매 비율이 가장 낮을 것이다.



## Day별 수행 내용

<details>
  <summary>Day 1 – 1일차 (6/26): 팀 빌딩과 분석 방향 설정</summary>

### **목표**: 우리 팀이 어떤 데이터를 가지고, 어떤 질문에 답을 찾을 것인지 명확히 정의하기

### **데이터셋 선택 전략**

**Brazilian E-Commerce (Olist) - ⭐⭐⭐⭐⭐**

- **장점**: 여러 테이블을 `pd.merge()`로 합치는 연습에 최고
- **적합한 질문**: "어떤 고객이 만족도가 높을까?", "배송 시간 단축 방안은?"
- **추천 대상**: 도전을 좋아하고, 실무형 프로젝트를 원하는 팀

**Mercari Price - ⭐⭐⭐⭐**

- **장점**: 가격 예측이라는 명확한 목표
- **기술 스택**: 텍스트 데이터 전처리(TfidfVectorizer, CountVectorizer), 회귀 모델
- **추천 대상**: 머신러닝에 집중하고 싶은 팀

**Google Play Store - ⭐⭐⭐**

- **장점**: 데이터가 직관적이라 EDA에 집중하기 좋음
- **적합한 질문**: "성공하는 앱의 특징은 무엇일까?"
- **추천 대상**: 안정적인 결과를 원하는 팀

### **데이터 처음 만났을 때: Pandas 기본 4총사**

```python
# 반드시 해야 할 기본 탐색
df.head()           # 데이터 구조 확인
df.info()           # 데이터 타입, 결측치 개수 확인
df.describe()       # 숫자형 데이터의 통계 요약
df.isnull().sum()   # 컬럼별 결측치 개수 상세 확인

```

### **1일차 체크리스트**

### **팀 빌딩 (1시간)**

- [x]  팀원 소개 및 아이스브레이킹 완료
- [x]  각자의 강점 파악 (Python 실력, 통계 지식, 시각화 경험 등)
- [x]  팀 소통 채널 개설 완료 (카톡방, Discord 등)
- [x]  역할 분담 1차 논의

### **데이터셋 선정 (1.5시간)**

- [x]  3개 데이터셋의 설명을 다시 읽고 팀의 흥미와 역량에 맞는 데이터셋 1개 선정
- [x]  선정 이유와 기대효과를 팀원 모두가 공감할 수 있도록 정리

### 선정이유 및 기대효과

- 김지수
    - 선정 이유: 실무와 가장 유사한 형태의 데이터, 데이터 내용이 방대해 다양한 내용의 가설을 설정하고 증명해가는 과정이 흥미로울 것 같아 선택
    - 기대효과: 큰 데이터셋인만큼 흥미로운 인사이트들이 많이 나올 것 같음. e-commerce 업계의 실무 데이터를 간접적으로 다뤄볼 수 있다는 점이 기대됨. 또한, 팀 프로젝트이기 때문에 내가 생각치 못한 창의저인 아이디어가 많이 나올 것 같아 기대됨
- 임수명
    - 선정이유: 범주, 분류, 회귀 등 적용할 수 있는 알고리즘이 여러가지이기 때문에 선택사항이 많음
    - 기대효과: 다양한 가설을 통해 재미있는 인사이트가 나올 것으로 기대됨
- 최재우
    - 선정 이유: 실제 비즈니스 데이터를 사용해서 다양한 정보를 다루며 비즈니스 분석과 시각화를 하여 실력 향상에 도움이 될거라고 생각해서 선택
    - 기대효과: 지도 기반 인사이트 도출을 더 잘 활용할 수 있을것으로 기대됨
- 서민영:
    - 선정 이유: 관계형 데이터 베이스로 전처리 과정이 실무와 가장 유사할 것 같고, 다양한 지표가 존재하여 지표간의 연관성을 확인하기 좋아보여서 선택했다.
    - 기대 효과: 문제 상황을 정해보고 이를 검증하는 과정에서의 다양한 아이디어와 결과물

### **데이터 초기 탐색 (2시간)**

- [x]  `pd.read_csv()`로 모든 데이터를 불러오고 기본 4총사로 탐색 완료
- [x]  각 파일의 크기, 컬럼 수, 결측치 비율 파악
- [x]  **(Olist 선택 시)** 9개 파일의 관계를 그림으로 그려보거나, 어떤 키(key)로 연결되는지 논의
- [x]  **(중요!)** 어떤 컬럼들이 분석에 핵심이 될지 1차 추정

### **분석 주제 설정 (1시간)**

- [x]  데이터에서 답을 찾고 싶은 재미있는 질문 3~5개 브레인스토밍
- [x]  팀의 최종 분석 주제(Main Question) 1개 확정
    - **좋은 예**: "브라질 고객의 리뷰 점수에 가장 큰 영향을 미치는 요인은 무엇인가?"
    - **나쁜 예**: "데이터를 분석해보자" (너무 모호함)

### 데이터 초기 탐색 및 분석 주제 아이디어 공유

- 김지수
    
    ### customer
    
    - **지역과 시기에 따라 고객들의 구매 빈도가 높은 상품은 무엇일까?**
        - 지역에 따라 구매 빈도가 높은 상품 카테고리 / 상품명 (지역 특성에 따른 구매?)
        - 시기에 따라 구매 빈도가 높은 상품 카테고리 / 상품명 (시즌성 구매?)
        - 월별, 요일, 시간대별 매출액 비교
        - 기대효과: 시즌과 지역에 맞춰 상이한 마케팅 및 프로모션 진행 가능 → 구매 극대화
    - **배송 속도가 리뷰 점수에 얼마나 영향을 미칠까?**
        - 배송 지연이 일어날 수밖에 없는 상황이 분명 존재하는데, 지역과 구매 품목에 따라 배송 지연이 구매 만족도에 얼마나 큰 영향을 미칠까? (평균 별점)
        - 지역에 따른 배송 속도에도 차이가 있기 때문에 중북부/남부/동부/서부 등으로 지역 단위를 나누어 고객 만족도를 평가함
        - 며칠까지는 고객 리뷰 good, 며칠부터는 고객 리뷰 bad
        - 디테일하게는 배송 품목에 따라서도 고객 만족도 평가 가능
- 임수명
    - 고객 행동 패턴 분류(고객데이터셋 + 주문데이터셋 + 주문아이템)
        - 고객 지역, 주 + 주문 수 + 주문제품/주문항목/가격
        - K-Means 클러스터링
        
        > 활용방안: 지역별 구매자 특성 파악, 상품 추천 전략, 마케팅 대상 세분화
        > 
    - 배송 예정일 예측(주문데이터셋 + 주문아이템 + 제품 데이터셋)
        - 배송비, 가격, 배송일수, 주문 요일, 제품 무게
        - 실제 배송소요일수 → 예상 배송일
        - 회귀모델
        
        > 활용방안 : 예상 배송일 안내, 배송이 지연되는 원인 파악
        > 
        
- 최재우
    - 판매자별 배송 성능 비교
        - order_items + orders +order_review + sellers → 판매자별 평균 배송일, 리뷰 점수
            - 배송 이슈 조기 탐지 및 개선 가능
    - 지역별 구매 트렌드 분석
        - customers + geolocation → 주별, 도시별 구매 패턴
            - 재고/물류 운영 효율와, 마케팅 타겟팅 최적화 가능
- 서민영
  
  <img width="771" height="907" alt="image" src="https://github.com/user-attachments/assets/7b19fe9b-93f5-42ed-a63f-d6477cbf9c0d" />

  +) 시간대별 많이 팔리는 품목 확인 후, 사이트 메인 광고로 업로드 

### 데이터셋 테이블 컬럼명
<img width="667" height="1115" alt="image" src="https://github.com/user-attachments/assets/ede1f987-8cd3-4dc6-bdb0-8fae5ca66b30" />

### 주제 선정

**<지역 기반 고객 구매 패턴 분석>**

- 카테고리별 구매 비율 (지역 기준)
    - 지역마다 인기 있는 제품군 파악 → 지역 맞춤 마케팅
- 지역별 평균 구매 금액 (스케일링 필요)
    - 지역별 구매력 또는 고가 제품 선호도 비교
- 지역별 구매 시간대 패턴
    - 마케팅 캠페인 타이밍 최적화
- 품목별 취소율 (‘canceled’ 비율)
    - 문제가 잦은 제품 카테고리 또는 판매자 파악
- 지역별 평균 리뷰 점수
    - 고객 만족도 지역별 편차 확인
- 결제유형 / 할부 수 분포 (지역별)
    - 지역별 소비 습관 차이 이해 (카드 사용/할부 선호 등)
- 구매 수 대비 총 지출 금액 (1인당 평균)
    - 지역별 구매 빈도 vs 지출 간 균형 분석

</details>

<details>
  <summary>Day 2 – 2일차 (6/27): EDA와 Feature Engineering / 분석 전략 수립</summary>

### **목표**: 데이터를 깨끗하게 만들고, 분석에 필요한 새로운 변수를 생성하며 구체적인 계획 세우기

### **가설 설정의 기술**

**좋은 가설의 조건:**

- 검증 가능해야 함
- 구체적이어야 함
- 데이터로 답할 수 있어야 함

```python
# 좋은 가설의 예
"배송 예정일보다 실제 배송이 빠른 주문은 리뷰 점수가 높을 것이다."

# 나쁜 가설의 예
"배송과 리뷰는 관련이 있을 것이다." (너무 모호함)

```

### **결측치 처리 전략**

```python
# 결측치 현황 파악
missing_data = df.isnull().sum()
missing_percent = 100 * missing_data / len(df)
missing_table = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print(missing_table[missing_table['Missing Count'] > 0].sort_values('Missing Percentage', ascending=False))

```

**처리 방법:**

- **삭제 (`.dropna()`)**: 데이터가 충분히 많고, 결측치가 일부 행에 집중될 때
- **대체 (`.fillna()`)**:
    - 숫자형: 평균(mean), 중앙값(median), 최빈값(mode) - 이상치가 많다면 중앙값이 안정적
    - 범주형: 최빈값(mode)이나 "N/A" 같은 새로운 카테고리

### **Feature Engineering 아이디어**

기존 데이터에서 새로운 특징을 뽑아내는 가장 창의적인 단계!

**Olist 데이터셋 예시:**

```python
# 시간 관련 피처
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['day_of_week'] = df['order_purchase_timestamp'].dt.day_name()
df['hour'] = df['order_purchase_timestamp'].dt.hour
df['month'] = df['order_purchase_timestamp'].dt.month

# 배송 관련 피처
df['total_delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
df['is_fast_delivery'] = df['total_delivery_days'] <= 7

```

**Mercari 데이터셋 예시:**

```python
# 텍스트 관련 피처
df['name_word_count'] = df['name'].str.split().str.len()
df['description_char_count'] = df['item_description'].str.len()
df['has_brand'] = df['brand_name'].notna()

# 브랜드 추출 (정규표현식 활용)
popular_brands = ['Apple', 'Samsung', 'Nike', 'Adidas']
df['is_popular_brand'] = df['brand_name'].isin(popular_brands)

```

**Google Play Store 예시:**

```python
# 설치 수 정규화
df['Installs_cleaned'] = df['Installs'].str.replace('[+,]', '', regex=True).astype(int)

# 업데이트 최신성
df['Last Updated'] = pd.to_datetime(df['Last Updated'])
df['days_since_update'] = (pd.Timestamp.now() - df['Last Updated']).dt.days

# 평점 카테고리화
df['rating_category'] = pd.cut(df['Rating'], bins=[0, 3, 4, 5], labels=['Low', 'Medium', 'High'])

```

### 주제 선정

**<지역 기반 고객 구매 패턴 분석>**

- 카테고리별 구매 비율 (지역 기준)
    - 지역마다 인기 있는 제품군 파악 → 지역 맞춤 마케팅
- 지역별 평균 구매 금액 (스케일링 필요)
    - 지역별 구매력 또는 고가 제품 선호도 비교
- 지역별 구매 시간대 패턴
    - 마케팅 캠페인 타이밍 최적화
- 품목별 취소율 (‘canceled’ 비율)
    - 문제가 잦은 제품 카테고리 또는 판매자 파악
- 지역별 평균 리뷰 점수
    - 고객 만족도 지역별 편차 확인
- 결제유형 / 할부 수 분포 (지역별)
    - 지역별 소비 습관 차이 이해 (카드 사용/할부 선호 등)
- 구매 수 대비 총 지출 금액 (1인당 평균)
    - 지역별 구매 빈도 vs 지출 간 균형 분석

목적: 브라질의 고객들을 지역 및 구매 행동 데이터를 바탕으로 군집화하여, 지역별로 어떤 유형의 고객이 주로 분포하는지를 파악한다. 

기대효과: 셀러들에게 해당 정보를 제공하여 셀러들의 이익 창출을 도운다. / 셀러 유치 

### 가설 설정

1. 지역에 따라 고객 유형이 다를 것이다 
    1. 5개 지역의 군집 분석 결과가 각각 다를 것이다
2. 평균 구매 금액이 높은 고객은 특정 지역에 밀집되어 있을 것이다
    1. 부유층이 밀집되어 있는 곳이 있을 것이다

 3. 저녁 6시~8시 사이에 구매 비율이 가장 높을 것이다

1. 새벽 12시~5시 사이에 구매 비율이 가장 낮을 것이다

### 3일차, 4일차 세부 일정 수립

- 2일차: 데이터 전처리 & 시각화
    
    결측치 삭제
    
    컬럼, 만들어야 할 컬럼
    
    한 데이터셋으로
    
- 3일차: 시각화 & 모델링
- 4일차: 발표

### 전처리 정보

| **Dataset** | **Columns** | 병합  | 불필요한 컬럼 삭제 |
| --- | --- | --- | --- |
| customers | customer_state | O | O |
| Products | Category_name | O | O |
| products_trans | category_name_translation | O | O |
| payments | payment_value | O | O |
|  | payment_type |  |  |
|  | payment_installments |  |  |
| order | order_purchase_timestamp | O | O |
|  | order_status |  |  |
| review | review_score |  | O |
- 연결 컬럼

order_id

product_id

seller_id

customer_id

zip_code_prefix

### **2일차 체크리스트**

### **가설 설정 및 계획 수립 (1시간)**

- [x]  메인 주제와 관련된 구체적인 가설 2~3개 설정
- [x]  각 가설을 검증하기 위해 필요한 분석 방법 계획
- [ ]  3일차, 4일차 세부 일정 수립

### **데이터 통합 및 전처리 (3시간)**

- [ ]  **(Olist 선택 시)** 필요한 테이블들을 `pd.merge()`를 이용해 하나의 분석용 데이터프레임으로 통합

```python
# 예시: Olist 테이블 통합
main_df = orders.merge(order_items, on='order_id') \
                .merge(customers, on='customer_id') \
                .merge(sellers, on='seller_id') \
                .merge(products, on='product_id')

```

- [ ]  결측치 처리 전략을 팀원과 논의하고 코드로 구현
- [ ]  Box Plot 등을 이용해 이상치를 확인하고 처리 방안 논의/구현

### **Feature Engineering (2시간)**

- [ ]  프로젝트 목표 달성에 도움이 될 새로운 피처(변수) 2개 이상 생성
- [ ]  생성한 피처의 분포 확인 및 유효성 검증
- [ ]  전처리가 완료된 데이터를 `_processed.csv`로 중간 저장하여 다음 날 바로 활용

### **EDA 시작**

- [ ]  주요 변수들의 기본 분포 확인 (`histplot`, `countplot`)
- [ ]  타겟 변수와 주요 피처들 간의 기초적인 관계 탐색


</details>

<details>
  <summary>Day 3 – 3일차 (6/30): 심화 EDA와 데이터 해석 / 모델링</summary>

### **목표**: 시각화를 통해 가설을 검증하고, 데이터 속 숨겨진 패턴과 인사이트를 발견하기

### **보충 학습 가이드**

### **상황별 최적 시각화 선택 (Seaborn, Matplotlib)**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

```

**그래프 선택 가이드:**

- `countplot`: 카테고리별 데이터 개수 확인 (예: 요일별 주문량)
- `histplot`/`kdeplot`: 숫자 데이터의 분포 확인 (예: 상품 가격 분포)
- `boxplot`: 카테고리별 숫자 데이터 분포 비교 (예: 카테고리별 앱 평점 비교)
- `scatterplot`: 두 숫자 데이터 간의 관계 확인 (예: 상품 설명 글자 수와 가격의 관계)
- `heatmap`: 여러 숫자 데이터 간의 상관관계 한눈에 보기 (`df.corr()`)

### **인사이트 도출: "So What?" 질문법**

단순한 결과에서 의미 있는 인사이트로 발전시키세요!

```
단순한 결과: "금요일에 주문량이 가장 많다."

의미 있는 인사이트: "금요일 주문량이 가장 많다. 이는 주말을 앞둔 소비 심리가
반영된 결과로 추정된다. 따라서 목요일 저녁이나 금요일 오전에 타겟 마케팅을
진행하면 효과적일 것이다."

```

### **머신러닝 모델링 기초**

**1단계: 문제 정의**

- **회귀(Regression)**: 연속된 숫자 예측 (예: Mercari 가격 예측)
- **분류(Classification)**: 카테고리 예측 (예: Olist 리뷰 점수가 5점인가 아닌가)

**2단계: 기본 파이프라인**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# 1. X(Feature), y(Target) 분리
X = df[['feature1', 'feature2', 'feature3']]  # 예측에 사용할 변수들
y = df['target']  # 예측하려는 목표 변수

# 2. 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 불러오기 및 학습
model = RandomForestRegressor(random_state=42)  # 회귀의 경우
# model = RandomForestClassifier(random_state=42)  # 분류의 경우
model.fit(X_train, y_train)

# 4. 예측
predictions = model.predict(X_test)

# 5. 평가
if problem_type == 'regression':
    mse = mean_squared_error(y_test, predictions)
    print(f'MSE: {mse}')
else:  # classification
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, predictions))

```

### **3일차 체크리스트**

### **상관관계 및 패턴 분석 (2시간)**

- [ ]  상관관계 분석(heatmap)을 통해 변수들 간의 전반적인 관계 파악

```python
# 상관관계 히트맵
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('변수 간 상관관계')
plt.show()

```

- [ ]  높은 상관관계(|r| > 0.7)를 보이는 변수 쌍 식별 및 해석

### **가설 검증 시각화 (3시간)**

- [ ]  세웠던 가설을 검증하기 위한 핵심 시각화 3개 이상 생성
- [ ]  각 시각화마다 명확한 제목과 해석 포함
- [ ]  시각화와 분석을 통해 발견한 의미 있는 인사이트 3가지 이상 정리

### **모델링 (선택사항, 2시간)**

- [ ]  **(모델링 진행 시)** 풀고자 하는 문제를 회귀/분류로 명확히 정의
- [ ]  **(모델링 진행 시)** Baseline 모델을 하나 이상 만들고 성능 평가 완료
- [ ]  피처 중요도 분석을 통해 어떤 변수가 예측에 가장 중요한지 파악

### **발표 준비 시작**

- [ ]  분석 결과와 인사이트를 발표 자료 초안에 정리 시작
- [ ]  가장 임팩트 있는 그래프 3-4개 선별

<img width="667" height="1115" alt="image" src="https://github.com/user-attachments/assets/9b5ab9ad-8c8c-4521-b67e-b58bcfea284a" />

### 컬럼명 정리 (merged_orders)

| **customer_id** | **region** | **order_id** | **order_status** | **payment_type** | **payment_installments** | **payment_value** |
| --- | --- | --- | --- | --- | --- | --- |
| 고객 id | 지역 | 주문 id | 주문상태  | 결제 유형  | 할부 | 지출금액 |
| **review_score** | **date** | **timestamp** | **product_category_name** | **product_category_name_english** | **main_category** |  |
| 리뷰점수 | 주문일 | 주문 시간  | 카테고리(포르투갈어) | 카테고리(영어) | 카테고리(분류) |  |

### 컬럼명 정리 (merged_customer)

| **customer_id** | **order_id** | **payment_value** | **order_spent** |
| --- | --- | --- | --- |
| 고객 id | 주문 횟수  | 지출 금액 | 고객 별 구매 당 평균 지출 금액 |

</details>

<details>
  <summary>Day 4 – 4일차 (7/1): 발표 준비 및 회고</summary>

### **목표**: 우리의 분석 여정과 결과를 설득력 있는 이야기로 만들어 청중에게 전달하기

### **발표 자료는 스토리텔링입니다**

**📖 완벽한 15분 발표 구성:**

1. **인트로 (2분)** - 문제 제기
    
    > "저희는 OOO 데이터에서 OOO라는 궁금증이 생겼습니다."
    > 
2. **데이터 소개 (2분)**
    
    > "저희가 사용한 데이터는 이런 특징을 가지고 있습니다."
    > 
3. **분석 과정 (6분)** - 여정
    
    > "이 궁금증을 풀기 위해 저희는 이런 전처리, EDA, 모델링을 했습니다."
    (가장 흥미로운 그래프 2~3개만!)
    > 
4. **핵심 결과 (3분)** - 발견
    
    > "그 결과, OOO라는 놀라운 사실을 발견했습니다."
    > 
5. **결론 및 제언 (1.5분)** - 인사이트
    
    > "따라서 저희는 OOO를 제안합니다. 이를 통해 OOO 효과를 기대할 수 있습니다."
    > 
6. **한계 및 향후 과제 (0.5분)**
    
    > "저희 분석은 OOO 한계가 있으며, 앞으로 OOO를 더 해보면 좋겠습니다."
    > 

### **팀 발표의 기술**

**역할 분담 예시:**

- **발표자 1**: 문제 정의 + 데이터 소개 - 지수님 OR 영서님
- **발표자 2**: EDA 결과 + 주요 인사이트 - 재우님
- **발표자 3**: 모델링 결과 + 결론 - 민영님
- **전체**: Q&A 대응

**발표 성공 팁:**

- 15분 시간을 **반드시** 지키도록 리허설 진행
- 각 슬라이드마다 핵심 메시지 1개씩만
- 복잡한 코드보다는 **결과**에 집중
- "우리가 발견한 것이 왜 중요한지" 강조

### **회고(Retrospective)의 중요성**

프로젝트의 성공/실패보다 더 중요한 것은 **'무엇을 배웠는가'** 입니다.

**KPT 방식 추천:**

- **Keep (잘한 점)**: 우리 팀이 계속 이어갔으면 하는 것
    - 소통이 원활하게 진행되었고 모든 팀원들이 책임감 있게 자신의 역할을 마무리했다.
    - 모든 팀원들이 열정적으로 소통하고 각자의 역할을 수행해 주었다.
    - 각자 맡은 바를 성실하게 수행하는 모습에 책임감을 얻었다.
    - 학습한 내용에 그치지 않고 새로운 기법들을 찾아보며 적용했다.
    
- **Problem (아쉬운 점)**: 이번 프로젝트에서 겪은 어려움
    - 각자의 역량과 시간 제한을 고려하지 못 하고 많은 소주제를 한번에 분석하려고 하니 시간에 쫓기며 프로젝트를 진행했다. 완성도 측면에서는 아쉬운 점이 있다.
    - 분석 기법 설명에만 집중한 나머지 우리의 스토리를 제대로 전달하지 못 했다. + 설득력이 부족했다.
    - 시간이 없다는 생각에 다른 파트의 분석 부분을 개별적으로 시도해보지 못한 부분
    - 각자가 맡은 부분에만 너무 집중했던 것과 발표 준비 시간이 짧았었던 부분

- **Try (다음에 시도할 것)**: 다음 프로젝트에서 새롭게 시도해볼 점
    - 이상치 추출에 다양한 방식을 써보지 못 한게 아쉬워서 다음엔 다양한 방식으로 시도해 보고싶다
    - 다른 주제로 새로운 분석을 시도해 보고 싶다.
    - 여러 소주제를 정해서 분석하려고 하기 보다는 하나의 주제를 정해 다양한 측면에서 디테일하게 분석해볼 것 (양보다는 질로!)
    - ppt 내용 분량을 조절해 발표 시간 엄수를 확실히 할 것

### **4일차 체크리스트**

### **발표 자료 완성 (4시간)**

- [x]  발표 자료(PPT, 구글 슬라이드 등) 초안 완성
- [x]  발표 스토리라인에 맞춰 슬라이드 순서 및 내용 수정
- [x]  각 슬라이드에 명확한 제목과 핵심 메시지 포함
- [x]  시각화 자료의 폰트, 색상, 크기 등 일관성 확보

### **발표 연습 (1.5시간)**

- [x]  팀원들과 함께 발표 리허설 진행 및 시간 체크
- [x]  발표자별 역할 분담 및 연결 부분 연습
- [ ]  Q&A 시간에 나올만한 예상 질문과 답변 준비
    - "왜 이 데이터셋을 선택했나요?"
    - "이 분석의 한계점은 무엇인가요?"
    - "실제 비즈니스에 어떻게 적용할 수 있나요?"

### **최종 제출 (0.5시간)**

- [x]  최종 발표 자료 제출
- [x]  분석에 사용한 주요 코드 파일 정리 및 공유

### **회고 및 마무리 (30분)**

- [x]  **(발표 후)** 팀 회고를 통해 프로젝트 경험 정리 및 공유
- [x]  개인별 학습 성과 및 아쉬운 점 솔직하게 공유
- [ ]  향후 데이터 분석 학습 계획 논의
</details>

---

## 진행한 EDA 예시 (서술용)

* 가격의 분포가 극단적으로 치우쳐 있어 로그 변환 필요
* 브랜드 정보가 가격 예측에 큰 영향을 미침
* 텍스트 설명의 길이가 가격과 일정 수준의 양의 상관관계 존재
* 카테고리 정보가 매우 다단계 구조 → Grouping 필요

---

## 모델링

* 기본 모델: Linear Regression, RandomForestRegressor, LightGBM
* 성능 평가 지표: RMSE
* 텍스트 설명을 TF-IDF 기반으로 벡터화 후 Tabular feature와 결합
* Hyperparameter tuning (GridSearch / Optuna 등 시도 가능)

---

## 결과

* Feature engineering 및 텍스트 반영 시 성능 향상 확인
* 카테고리/브랜드가 주요 변수로 작용
* 텍스트 정보 활용의 중요성 검증

---

## 발표 자료

* 분석 과정 및 인사이트 중심 구조로 구성
* 모델 평가 결과 및 한계점 공유
* 향후 개선 방향 제시

---

## Repository 구조 예시

```plaintext
├── data/                  # 데이터셋
├── notebooks/             # 분석 및 모델링 과정 ipynb 파일
├── src/                   # 전처리 / 모델 코드 분리 가능
├── images/                # 그래프, 시각화 이미지
├── README.md              # 프로젝트 설명 문서
└── presentation.pdf       # 발표 자료
```

---

## 회고

* 짧은 시간 동안 데이터 분석 전체 사이클을 경험할 수 있었음
* 역할 분담이 가능해 빠르게 실전 프로젝트 경험 확보
* 텍스트 데이터 다루는 능력 향상
* 비즈니스 관점에서 인사이트 도출의 중요성 깨달음

---

## 📮 Contact

필요한 내용이나 프로젝트 상세 설명 요청은 언제든지 문의해주세요!
