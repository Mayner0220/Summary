# Count based word Representation (1)

source: https://wikidocs.net/24557, https://wikidocs.net/31767, https://wikidocs.net/22650, https://wikidocs.net/24559, https://wikidocs.net/31698

---

### 0. 카운트 기반의 단어 표현

NLP에서 텍스트를 표현하는 방법으로는 여러가지 방법이 있다.
우리가 앞서 배운 n-gram 또한 텍스트를 표현하는 방법 중 하나입니다.
하지만 ML등의 알고리즘이 적용된 본격적인 NLP를 위해서는 문자를 숫자로 수치화할 필요가 있다.
그런 측면에서 앞으로 문자를 숫자로 수치화하는 방법에 대해서 배우게 된다.

---

### 1.다양한 단어의 표현 방법

카운트 기반의 단어 표현 방법은 Bag of Words에서 배우게 된다.
지금은 카운트 기반의 단어 표현 방법 외에도 다양한 단어의 표현 방법에는 어떤 것이 있으며, 앞으로 어떤 순서로 단어 표현 방법을 학습하게 될 것인지에 대해서 먼저 설명한다.

----

### 1.1 단어의 표현 방법

단어의 표현 방법은 크게 국소 표현(Local Representation) 방법과 분산 표현(Distributed Representation) 방법으로 나뉜다.
국소 표현 방법은 해당 단어 그 자체만 보고, 특정값을 맵핑하여 단어를 표현하는 방법이며, 분산 표현 방법 그 단어를 표현하고자 주변을 참고하여 단어를 표현하는 방법이다.

예를 들어 puppy(강아지), cute(귀여운), lovely(사랑스러운)라는 단어가 있을 때 각 단어에 1번, 2번, 3번 등과 같은 숫자를 맵핑(mapping)하여 부여한다면 이는 국소 표현 방법에 해당된다.
반면, 분산 표현 방법의 예를 하나 들어보면 해당 단어를 표현하기 위해 주변 단어를 참고한다.
puppy(강아지)라는 단어 근처에는 주로 cute(귀여운), lovely(사랑스러운)이라는 단어가 자주 등장하므로, puppy라는 단어는 cute, lovely한 느낌이다로 단어를 정의한다.
이렇게 되면 이 두 방법의 차이는 국소 표현 방법은 단어의 의미, 뉘앙스를 표현할 수 없지만, 분산 표현 방법은 단어의 뉘앙스를 표현할 수 있게 된다.

또한 비슷한 의미로 국소 표현 방법(Local Representation)을 이산 표현(Discrete Representation)이라고도 하며, 분산 표현(Distributed Representation)을 연속 표현(Continuous Representation)이라고도 한다.

다른 의견으로는, 구글의 연구원 토마스 미코로브는 2016년에 한 발표에서 LSA나 LDA와 가은 방법들은 단어의 의미를 표현할 수 있다는 점에서 연속 표현이지만, 엄밀히 말해서 다른 접근의 방법론을 사용하고 있는 Word2vec과 같은 분산 표현은 아닌 것으로 분류하여 연속 표현을 포괄하고 있는 더 큰 개념으로 설명하기도 했다.

---

### 2. 단어 표현의 카테고리화

아래와 같은 기준으로 단어 표현을 카테고리화하여 작성됬다.

![img](https://wikidocs.net/images/page/31767/wordrepresentation.PNG)

Bags of Words는 국소 표현(Local Representation)에 속하며, 단어의 빈도수를 카운트하여 단어를 수치화하는 단어 표현 방법이다.
이번에는 BoW와 그의 확장인 DTM(또는 TDM)에 대해서 학습하고, 이러한 빈도수 기반 단어 표현에 단어의 중요도레 따른 가중치를 줄 수 있는 TF-IDF에 대해서 학습합니다.

단어의 뉘앙스를 반영하는 연속 표현의 일종인 LSA를 토픽 모델링이라는 주제로 학습합니다.

연속 표현에 속하면서, 예측(prediction)을 기반으로 단어의 뉘앙스를 표현하는 Word2vec과 그의 확장인 패스트텍스트(FastText)를 학습하고, 예측과 카운트라는 두 가지 방법이 모두 사용된 글로브(GloVe)에 대해서 학습한다.

---

### 2. Bag of Words(Bow)

단어의 등장 순서를 고려하지 않는 빈도수 기반의 단어 표현 방법인 Bag of Words에 대해서 알아보자.

---

### 2.1 Bag of Words?

Bag of Words란 단어들의 순서는 전혀 고려하지 않고, 단어들의 출현 빈도(frequency)에만 집중하는 텍스트 데이터의 수치화 표현 방법이다.
Bag of Words를 직역하면 단어들의 가방이라는 의미이다.
단어들이 들어있는 가방을 상상해보자.
갖고있는 어떤 텍스트 문서에 있는 단어들을 가방에다가 전부 넣는다고 생각해보자.
그러고나서 이 가방을 흔들어 단어들을 섞는다.
만약 해당 문서 내에서 특정 단어가 N번 등장 했다면, 이 가방에는 그 특정 단어가 N개 있게 된다.
또한 가방을 흔들어서 단어를 섞었기 때문에 더 이상 단어의 순서는 중요하지 않다.

BoW를 만드는 과정을 이렇게 두 가지 과정으로 생각해보자.

1. 우선, 각 단어에 고유한 정수 인덱스를 부여한다.
2. 각 인덱스의 위치에 단어 토큰의 등장 횟수를 기록한 벡터를 만든다.

한국어 예제를 통해서 BoW에 대해서 이해해보자.

- 예문1: 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.

위의 예문1에 애서 BoW를 만들어보자.
아래의 코드는 입력된 문서에 대해서 단어 집합을 만들어 인덱스를 할당하고, BoW를 만드는 코드이다.

```python
rom konlpy.tag import Okt
import re  
okt=Okt()  

token=re.sub("(\.)","","정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")  
# 정규 표현식을 통해 온점을 제거하는 정제 작업입니다.  
token=okt.morphs(token)  
# OKT 형태소 분석기를 통해 토큰화 작업을 수행한 뒤에, token에다가 넣습니다.  

word2index={}  
bow=[]  
for voca in token:  
         if voca not in word2index.keys():  
             word2index[voca]=len(word2index)  
# token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.   
             bow.insert(len(word2index)-1,1)
# BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 개수는 최소 1개 이상이기 때문입니다.  
         else:
            index=word2index.get(voca)
# 재등장하는 단어의 인덱스를 받아옵니다.
            bow[index]=bow[index]+1
# 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줍니다. (단어의 개수를 세는 것입니다.)  
print(word2index)  
```

```python
('정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9)  
```

```python
bow  
```

```python
[1, 2, 1, 1, 2, 1, 1, 1, 1, 1]  
```

예문1에 각 단어에 대해서 인덱스를 부여한 결과는 첫번째 출력 결과이다.
문서1의 BoW는 두번째 출력 결과이다.
두번째 출력 결과를 보면, 물가상승률의 인덱스는 4이며, 예문1에서 물가상승률은 2번 언급되었기 때문에 인덱스 4에 해당하는 값이 2임을 알 수 있다.

---

### 2.2 Bag of Words의 다른 예제들

BoW에 있어서 중요한 것은 단어의 등장 빈도이다.
단어의 순서, 즉, 인덱스의  순서는 전혀 상관이 없다.
예문1에 대한 인덱스 할당을 임의로 바꾸고, 그에 따른 BoW를 만든다고 해보자.

```python
# ('발표': 0, '가': 1, '정부': 2, '하는': 3, '소비자': 4, '과': 5, '물가상승률': 6, '느끼는': 7, '은': 8, '다르다': 9)  
[1, 2, 1, 1, 1, 1, 2, 1, 1, 1]
```

위의 BoW는 단지 단어들의 인덱스만 바뀌었을 뿐이며, 개념적으로는 여전히 앞서 만든 BoW와 동일한 BoW로 취급할 수 있다.

- 예문2: 소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.

만약, 위의 코드에 예문2로 입력하여 인덱스 할당과 BoW를 만드는 것을 진행한다면 아래와 같은 결과가 나온다.

```python
('소비자': 0, '는': 1, '주로': 2, '소비': 3, '하는': 4, '상품': 5, '을': 6, '기준': 7, '으로': 8, '물가상승률': 9, '느낀다': 10)  
[1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]  
```

예문1과 예문2를 합쳐서(이를 예문3이라고 하자.) BoW를 만들 수 있다.

- 예문3: 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다. 소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.

위의 코드에 예문3을 입력으로 하여 인덱스 할당과 BoW를 만든다면 아래화 같은 결과가 나온다.

```python
('정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9, '는': 10, '주로': 11, '소비': 12, '상품': 13, '을': 14, '기준': 15, '으로': 16, '느낀다': 17)  
[1, 2, 1, 2, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]  
```

예문3의 단어 집합은 예문1과 예문2의 단어들을 모두 포함하고 있는 것들을 볼 수 있다.
BoW는 종종 여러 문서의 단어 집합을 합친 뒤에, 해당 단어 집합에 대한 각 문서의 BoW를 구하기도 한다.
가령, 예문3에 대한 단어 집합을 기준으로 예문1, 예문2의 BoW를 만든다고 한다면 결과는 아래와 같다.

```python
문서3 단어 집합에 대한 문서1 BoW : [1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  
문서3 단어 집합에 대한 문서2 BoW : [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, 1, 1, 1]  
```

예문3 단어 집합에서 물가상승률이라는 단어는 인덱스가 4에 해당된다.
물가상승률이라는 단어는 예문2에서는 2회 등장하며, 예문에서는 1회 등장하였기 때문에 두 BoW의 인덱스 4의 값은 각각 2와 1이 되는 것을 볼 수 있다.

BoW는 각 단어가 등장한 횟수를 수치화하는 텍스트 표현 방법이기에, 주로 어떤 단어가 얼마나 등장했는지를 기준으로 문서가 어떤 성격의 문서인지를 판단하는 작업에 쓰인다.
즉, 분류 문제나 여러 문서 간의 유사도를 구하는 문제에 주로 쓰인다.
가령, '달리기', '체력', '근력'과 같은  단어가 자주 등장하면 해당 문서를 체육 관련 문서로 분류할 수 있을 것이며, '미분', '방정식', '부등식'과 같은 단어가 자주 등장한다면 수학 관련 문서로 분류할 수 있다.

---

### 2.3 CountVecorizer 클래스로 BoW 만들기

사이킷 런에서는 단어의 빈도를 Count하여 Vector로 만드는 CountVectorizer 클래스를 지원한다.
이를 이용하면 영어에 대해서는 손쉽게 BoW를 만들 수 있다.
CountVectorizer로 간단하고 빠르게 BoW를 만드는 실습을 해보자.

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
```

```python
[[1 1 2 1 2 1]]
{'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}
```

예제 문장에서 you와 love는 두 번씩 언급되었으므로 각각 인덱스 2와 인덱스 4에서 2의 값을 가지며, 그 외의 값에서는 1의 값을 가지는 것을 볼 수 있다.
또한 알파벳 I는 BoW를 만드는 과정에서 사라졌는데, 이는 CountVectorizer가 기본적으로 길이가 2 이상인 문장에 대해서만 토큰을 인식했기 때문이다.
정제 부분에서 나왔듯이, 영어에서는 길이가 짧은 문자를 제거하는 것 또한 전처리 작업으로 고려되기도 한다.

주의할 것은 CountVectorizer는 단지 띄어쓰기만을 기준으로 단어를 자르는 낮은 수준의 토큰화를 진행히고 BoW를 만든다는 점이다. 이는 영어의 경우 띄어쓰기만으로 토큰화가 수행되기 때문에 문제가 없지만 한국어에는 CountVectorizer를 적용하면, 조사 등의 이유로 제대로 BoW가 만들어지지 않음을 의미한다.

예를 들어, 앞서 BoW를 만드는데 사용했던  '정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.' 라는 문장을 CountVectorizer를 사용하여 BoW로 만들 경우, CountVectorizer는 '물가상승률'이라는 단어를 인식하지 못한다.
CountVectorizer는 띄어쓰기를 기준으로 분리한 뒤에 '물가상승률과'와 '물가상승률은' 으로 조사를 포함해서 하나의 단어로 판단하기 때문에 서로 다른 두 단어로 인식한다.
그리고 '물가상승률과'와 '물가상승률은'이 각자 다른 인덱스에서 1이라는 빈도의 값을 갖게 된다.

---

### 2.4 불용어를 제거한 BoW 만들기

앞서 불용어는 NLP에서 별로 의미를 갖지 않는 단어들이라고 한 적이 있다.
BoW를 사용한다는 것은 그 문서에서 각 단어가 얼마나 자주 등장했는지를 보겠다는 것이다.
그리고 각 단어에 대한 빈도수를 수치화 하겠다는 것은 결국 텍스트 내에서 어떤 단어들이 중요한지를 보고싶다는 의미를 함축하고 있다.
그렇다면 BoW를 만들 때 불용어를 제거하는 일은 NLP의 정확도를 높이기 위해서 선택할 수 있는 전처리 기법이다.

영어의 BoW를 만들기 위해 사용되는 CountVectorizer는 불용어를 지정하면, 불용어는 제외하고 BoW를 만들 수 있도록 불용어 제거 기능을 지원한다.

1. 사용자가 직접 정의한 불용어 사용

   ```python
   from sklearn.feature_extraction.text import CountVectorizer
   
   text=["Family is not an important thing. It's everything."]
   vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])
   print(vect.fit_transform(text).toarray()) 
   print(vect.vocabulary_)
   ```

   ```python
   [[1 1 1 1 1]]
   {'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}
   ```

2. CountVectorizer에서 제공하는 자체 불용어 사용

   ```python
   from sklearn.feature_extraction.text import CountVectorizer
   
   text=["Family is not an important thing. It's everything."]
   vect = CountVectorizer(stop_words="english")
   print(vect.fit_transform(text).toarray())
   print(vect.vocabulary_)
   ```

   ```python
   [[1 1 1]]
   {'family': 0, 'important': 1, 'thing': 2}
   ```

3. NLTK애서 지원하는 불용어 사용

   ```python
   from sklearn.feature_extraction.text import CountVectorizer
   from nltk.corpus import stopwords
   
   text=["Family is not an important thing. It's everything."]
   sw = stopwords.words("english")
   vect = CountVectorizer(stop_words =sw)
   print(vect.fit_transform(text).toarray()) 
   print(vect.vocabulary_)
   ```

   ```python
   [[1 1 1 1]]
   {'family': 1, 'important': 2, 'thing': 3, 'everything': 0}
   ```

   

