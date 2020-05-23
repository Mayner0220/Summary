# Document Similarity

Source: https://wikidocs.net/24602, https://wikidocs.net/24603, https://wikidocs.net/24654

---

### 0. 문서 유사도 (Document Similarity)

문서의 유사도를 구하는 일은 NLP의 주요 주제 중 하나이다.
사람들이 인식하는 문서의 유사도는 주로 문서들 간에 동일한 단어 또는 비슷한 단어가 얼마나 공통적으로 많이 사용됬는지에 의존한다.
기계도 마찬가지다.
기계가 계산하는 문서의 유사도의 성능은 각 문서의 단어들을 어떤 방법으로 수치화하여 표현했는지(DTM, Word2Vec 등), 문서 간의 단어들의 차이를 어떤 방법(유클리드 거리, 코사인 유사도 등)으로 계산했는지에 달렸다.

---

### 1.1. 코사인 유사도(Cosine Similarity)

코사인 유사도는 두 벡터 간의 코사인 각도를 이용하여 구할 수 있는 두 벡터의 유사도를 의미한다.
두 벡터의 방향이 완전히 동일한 경우에는 1의 값을 가지며, 90°의 각을 이루면 0, 180°로 반대의 방향을 가지면 -1의 값을 갖게 된다.
즉, 결국 코사인 유사도는 -1 이상 1 이하의 값을 가지며 값이 1에 가까울수록 유사도가 높다고 판단할 수 있다.
이를 직관적으로 이해하면 두 벡터가 가르키는 방향이 얼마나 유사한가를 의미한다.

![img](https://wikidocs.net/images/page/24603/%EC%BD%94%EC%82%AC%EC%9D%B8%EC%9C%A0%EC%82%AC%EB%8F%84.PNG)

두 벡터 A, B에 대해서 코사인 유사도는 식으로 표현하면 다음과 같다.
$$
similarity=cos(Θ)=\frac{A⋅B}{||A||\ ||B||}=\frac{\sum_{i=1}^{n}{A_{i}×B_{i}}}{\sqrt{\sum_{i=1}^{n}(A_{i})^2}×\sqrt{\sum_{i=1}^{n}(B_{i})^2}}
$$
문서 단어 행렬이나 TF-IDF 행렬을 통해서 문서의 유사도를 구하는 경우에는 문서 단어 행렬이나 TF-IDF 행렬이 각각의 특징 벡터 A, B가 된다.
그렇다면 문서 단어 행렬에 대해서 코사인 유사도를 구해보는 간단한 예제를 진행해보자.

- 문서1: 저는 사과 좋아요
- 문서2: 저는 바나나 좋아요
- 문서3: 저는 바나나 좋아요 저는 바나나 좋아요

위의 세 문서에 대해서 문서 단어 행렬을 만들면 이와 같다.

| -     | 바나나 | 사과 | 저는 | 좋아요 |
| ----- | :----: | :--: | :--: | :----: |
| 문서1 |   0    |  1   |  1   |   1    |
| 문서2 |   1    |  0   |  1   |   1    |
| 문서3 |   2    |  0   |  2   |   2    |

파이썬에서는 코사인 유사도를 구하는 방법은 여러가지가 있는데, 이번에는 Numpy를 이용해서 계산해보자.

```python
from numpy import dot
from numpy.linalg import norm
import numpy as np
def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))
```

코사인 유사도를 계산하는 함수를 만들었다.

```python
doc1=np.array([0,1,1,1])
doc2=np.array([1,0,1,1])
doc3=np.array([2,0,2,2])
```

예를 들었던 문서1, 문서2, 문서3에 대해서 각각 BoW를 만들었다.
이제 각 문서에 대한 코사인 유사도를 계산해보자.

```python
print(cos_sim(doc1, doc2)) #문서1과 문서2의 코사인 유사도
print(cos_sim(doc1, doc3)) #문서1과 문서3의 코사인 유사도
print(cos_sim(doc2, doc3)) #문서2과 문서3의 코사인 유사도
```

```python
0.67
0.67
1.00
```

눈여겨볼만한 점은 문서1과 문서2의 코사인 유사도와 문서1과 문서3의 코사인 유사도가 같다는 점과 문서2와 문서3의 코사인 유사도가 1이 나온다는 점이다.
앞서 1은 두 벡터의 방향이 완전히 동일한 경우에 1이 나오며, 코사인 유사도 관점에서는 유사도의 값이 최대임을 의미한다고 한 적이 있다.

문서3은 문서2에서 단지 모든 단어의 빈도수가 1씩 증가 했을 뿐 입니다.
다시 말해 한 문서 내의 모든 단어의 빈도수가 증가하는 경우에는 기존의 문서와 코사인 유사도의 값이 1이라는 것이다.
이것이 시사하는 점은 무엇일까.
코사인 유사도를 사용하지 않는다고 가정했을 때, 문서 A에 대해서 모든 문서와의 유사도를 구한다고 가정해보자.
다른 문서들과 문서 B나 거의 동일한 패턴을 가지는 문서임에도 문서 B가 단순히 다른 문서들보다 유사도가 더 높게 나온다면 이는 우리가 원하는 결과가 아니다.
코사인 유사도는 문서의 길이가 다른 상황에서 비교적 공정한 비교를 할 수 있도록 도와준다.

이는 코사인 유사도는 유사도를 구할 때, 벡터의 크기가 아니라 벡터의 방향(패턴)에 초점을 두기 때문이다.
코사인 유사도를 구하는 또 다른 방법인 내적과 가지는 차이이다.

---

### 1.2. 유사도를 이용한 추천 시스템 구현하기

캐글에서 사용되었던 영화 데이터셋을 가지고 영화 추천 시스템을 만들어보자.
TF-IDF와 코사인 유사도만으로 영화의 줄거리에 기반해서 영화를 추천하는 추천 시스템을 만들 수 있다.

다운로드 링크 : https://www.kaggle.com/rounakbanik/the-movies-dataset

원본 파일은 위 링크에서 movies_metadata.csv 파일을 다운로드 받으면 된다.
해당 데이터는 총 24개의 열을 가진 45,466개의 샘플로 구성된 영화 정보 데이터이다.

```
import pandas as pd
data = pd.read_csv('현재 movies_metadata.csv의 파일 경로', low_memory=False)
# 예를 들어 윈도우 바탕화면에 해당 파일을 위치시킨 저자의 경우
# pd.read_csv(r'C:\Users\USER\Desktop\movies_metadata.csv', low_memory=False)
data.head(2)
```

우선 다운로드 받은 훈련 데이터에서 2개의 샘플만 출력하여, 데이터가 어떤 형식을 갖고있는지 확인한다.
csv 파일을 읽기 위해서 파이썬 라이브러리인 pandas를 이용할 것이다.

|  -   | ...  | original_title |                      overview                      | ...  |   title   | video | vote_average | vote_count |
| :--: | :--: | :------------: | :------------------------------------------------: | :--: | :-------: | :---: | :----------: | :--------: |
|  0   | ...  |   Toy Story    | Led by Woody, Andy's toys live happily in his ...  | ...  | Toy Story | False |     7.7      |   5415.0   |
|  1   | ...  |    Jumanji     | When siblings Judy and Peter discover an encha ... | ...  |  Jumanji  | False |     6.9      |   2413.0   |

훈련 데이터는 총 24개의 열을 갖고 있으나, 편의상 일부를 생략했다.
여기서 코사인 유사도에 사용할 데이터는 영화 제목에 해당하는 title 열과 줄거리에 해당하는 overview 열이다.
좋아하는 영화를 입력하면, 해당 영화의 줄거리와 줄거리가 유사한 영화를 찾아서 추천하는 시스템을 만들 것이다.

```
data=data.head(20000)
```

만약 훈련 데이터의 양을 줄이고 학습을 진행하고자 한다면, 이와 같이 데이터를 줄여서 재저장할 수 있다.
20,000개의 샘플만 가지고 학습을 진행해보자.
tf-idf를 할 때 데이터에 Null 값이 들어있으면 에러가 발생한다.
tf-idf의 대상이 되는 data의 overview 열에 Null 값이 있는지 확인한다.

```
data['overview'].isnull().sum()
```

```
135
```

135개의 샘플에서 Null 값이 있다고 한다.
pandas를 이용하면 Null 값을 처리하는 도구인 fillna()를 사용할 수 있다.
괄호 안에 Nyll 대신 넣고자하는 값을 넣으면 되는데, 이 경우에는 빈 값(empty value)으로 대체하여 Null 값을 제거한다.

```
data['overview'] = data['overview'].fillna('')
# overview에서 Null 값을 가진 경우에는 값 제거
```

Null 값을 제거했다.
이제 .innull.sum()를 수행하면 0의 값이 나온다.
이제 tf-idf를 수행하자.

```
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])
# overview에 대해서 tf-idf 수행

print(tfidf_matrix.shape)
```

```
(20000, 47487)
```

overview 열에 대해서 tf-idf를 수행했다.
20,000개의 영화를 표현하기위해 총 47,487개의 단어가 사용되었음을 보여준다.
이제 코사인 유사도를 사용하면 문서의 유사도를 구할 수 있다.

```
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data['title']).drop_duplicates()

print(indices.head())
```

영화의 타이틀과 인덱스를 가진 테이블을 만든다.
이 중 5개만 출력해보자.

```
title
Toy Story                      0
Jumanji                        1
Grumpier Old Men               2
Waiting to Exhale              3
Father of the Bride Part II    4
dtype: int64
```

이 테이블의 용도는 영화의 타이틀을 입력하면 인덱스를 리턴하기 위함이다.

```
idx = indices['Father of the Bride Part II']

print(idx)
```

```
4
```

이제 선택한 영화에 대해서 코사인 유사도를 이용하여, 가장 overview가 유사한 10개의 영화를 찾아내는 함수를 만든다.

```
def get_recommendations(title, cosine_sim=cosine_sim):
    # 선택한 영화의 타이틀로부터 해당되는 인덱스를 받아옵니다. 이제 선택한 영화를 가지고 연산할 수 있습니다.
    idx = indices[title]

    # 모든 영화에 대해서 해당 영화와의 유사도를 구합니다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 영화를 받아옵니다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 영화의 인덱스를 받아옵니다.
    movie_indices = [i[0] for i in sim_scores]

    # 가장 유사한 10개의 영화의 제목을 리턴합니다.
    return data['title'].iloc[movie_indices]
```

영화 다크 나이트 라이즈와 overview가 유사한 영화들을 찾아보자.

```
get_recommendations('The Dark Knight Rises')
```

```
12481                            The Dark Knight
150                               Batman Forever
1328                              Batman Returns
15511                 Batman: Under the Red Hood
585                                       Batman
9230          Batman Beyond: Return of the Joker
18035                           Batman: Year One
19792    Batman: The Dark Knight Returns, Part 1
3095                Batman: Mask of the Phantasm
10122                              Batman Begins
Name: title, dtype: object
```

가장 유사한 영화가 출력되는데, 영화 다크나이트가 첫번째고, 그 외에도 전부 배트맨 영화를 찾아낸 것을 확인할 수 있다.

---

### 2. 여러가지 유사도 기법

문서의 유사도를 구하기 위한 방법으로는 코사인 유사도 외에도 여러가지 방법들이 있다.

---

#### 2.1 유클리드 거리(Euclidean distance)

유클리드 거리(euclidean distance)는 문서의 유사도를 구할 때 자카드 유사도나 코사인 유사도만큼, 유용한 방법은 아니다.
하지만 여러 가지 방법을 이해하고, 시도해 보는 것 자체만으로 다른 개념들을 이해할 때 도움이 되므로 의미가 있다.

다차원 공간에서 두개의 점 p와 q가 각각 p=(p1, p2, p3, ...., pn)과 q=(q1, q2, q3, ..., qn)의 좌표를 가질 때 두 점 사이의 거리를 계산하는 유클리드 거리 공식은 다음과 같다.
$$
\sqrt{(q_{1}-p_{1})^{2}+(q_{2}-p_{2})^{2}+\ ...\ +(q_{n}-p_{n})^{2}}=\sqrt{\sum_{i=1}^{n}(q_{i}-p_{i})^{2}}
$$
다차원 공간이라고 가정하면, 처음 보는 입장에서는 식이 너무 복잡해 보인다.
좀 더 쉽게 이해하기 위해서 2차원 공간이라고 가정하고 두 점 사이의 거리를 좌표 평면 상에서 시각화해보자.

![img](https://wikidocs.net/images/page/24654/2%EC%B0%A8%EC%9B%90_%ED%8F%89%EB%A9%B4.png)

2차원 좌표 평면 상에서 두 점 p와 q사이의 직선 거리를 구하는 문제이다.
위의 경우에는 직각 삼각형으로 표현이 가능하므로, 중학교 수학 과정인 피타고라스의 정리를 통해 p와 q 사이의 거리를 계산할 수 있다.
즉, 2차원 좌표 평면에서 두 점 사이의 유클리드 거리 공식은 피타고라스의 정리를 통해 두 점 사이의 거리를 구하는 것과 동일하다.

다시 원점으로 돌아가서 여러 문서에 대해서 유사도를 구하고자 유클리드 거리 공식을 사용한다는 것은, 앞서 본 2차원을 단어의 총 개수만큼의 차원으로 확장하는 것과 같다.
예를 들어 아래와 같은 DTM이 있다고 하자.

|   -   | 바나나 | 사과 | 저는 | 좋아요 |
| :---: | :----: | :--: | :--: | :----: |
| 문서1 |   2    |  3   |  0   |   1    |
| 문서2 |   1    |  2   |  3   |   1    |
| 문서3 |   2    |  1   |  2   |   2    |

단어의 개수가 4개이므로, 이는 4차원 공간에 문서1, 문서2, 문서3을 배치하는 것과 같다.
이때 다음과 같은 문서 Q에 대해서 문서1, 문서2, 문서3 중 가장 유사한 문서를 찾아내고자 한다.

|   -    | 바나나 | 사과 | 저는 | 좋아요 |
| :----: | :----: | :--: | :--: | :----: |
| 문서 Q |   1    |  1   |  0   |   1    |

이때 유클리드 거리를 통해 유사도를 구하려고 한다면, 문서 Q 또한 다른 문서들처럼 4차원 공간에 배치시켰다는 관점에서 4차원 공간에서의 각각의 문서들과 유클리드 거리를 구하면 된다.
이를 파이썬 코드로 구현해보자.

```python
import numpy as np
def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

doc1 = np.array((2,3,0,1))
doc2 = np.array((1,2,3,1))
doc3 = np.array((2,1,2,2))
docQ = np.array((1,1,0,1))

print(dist(doc1,docQ))
print(dist(doc2,docQ))
print(dist(doc3,docQ))
```

```python
2.23606797749979
3.1622776601683795
2.449489742783178
```

유클리드 거리의 값이 가장 작다는 것은, 문서 간의 거리가 가장 가깝다는 것을 의미한다.
즉, 문서1이 문서 Q와 가장 유사하다고 볼 수 있다.

---

### 2.2 자카드 유사도(Jaccard similarity)

A와 B 두 개의 집합이 있다고 하자.
이때 두 개의 집합에서 공통으로 가지고 있는 원소들의 집합을 말한다.
즉, 합집합에서 교집합의 비율을 구한다면 두 집합 A와 B의 유사도를 구할 수 있다는 것이 자카드 유사도의 아이디어이다.
자카드 유사도는 0과 1사이의 값을 가지며, 만약 두 집합이 동일하다면 1의 값을 가지고, 두 집합의 공통 원소가 없다면 0의 값을 가진다.
$$
J(A,B)=\frac{|A∩B|}{|A∪B|}=\frac{|A∩B|}{|A|+|B|-|A∩B|}
$$
두 개의 비교할 문서를 각각 doc1, doc2라고 했을 때 doc1과 doc2의 문서의 유사도를 구하기 위한 자카드 유사도는 이와 같다.
$$
J(doc_{1},doc_{2})=\frac{doc_{1}∩doc_{2}}{doc_{1}∪doc_{2}}
$$
즉, 두 문서 doc1, doc2 사이의 자카드 유사도 J(doc1, doc2)는 두 집합의 교집합 크기를 나눈 값으로 정의된다.
간단한 예를 통해서 이해해보자.

```python
# 다음과 같은 두 개의 문서가 있습니다.
# 두 문서 모두에서 등장한 단어는 apple과 banana 2개.
doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"

# 토큰화를 수행합니다.
tokenized_doc1 = doc1.split()
tokenized_doc2 = doc2.split()

# 토큰화 결과 출력
print(tokenized_doc1)
print(tokenized_doc2)
```

```python
['apple', 'banana', 'everyone', 'like', 'likey', 'watch', 'card', 'holder']
['apple', 'banana', 'coupon', 'passport', 'love', 'you']
```

이때 문서1과 문서2의 합집합을 구해보자.

```python
union = set(tokenized_doc1).union(set(tokenized_doc2))

print(union)
```

```python
{'card', 'holder', 'passport', 'banana', 'apple', 'love', 'you', 'likey', 'coupon', 'like', 'watch', 'everyone'}
```

문서1과 문서2의 합집합의 단어의 총 개수는 12개인 것을 확인할 수 있다.
그렇다면, 문서1과 문서2의 교집합을 구해보자.
즉, 문서1과 문서2에서 둘 다 등장한 단어를 구하게 된다.

```python
intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))

print(intersection)
```

```python
{'banana', 'apple'}
```

문서1과 문서2에서 둘 다 등장한 단어는 banana와 apple 총 2개이다.
이제 교집합의 수를 합집합의 수로 나누면 자카드 유사도가 계산된다.

```python
print(len(intersection)/len(union)) # 2를 12로 나눔.
```

```python
0.16666666666666666
```

위의 값은 자카드 유사도이자, 두 문서의 총 단어 집합에서 두 문서에서 공통적으로 등장한 단어의 비율이기도 한다.