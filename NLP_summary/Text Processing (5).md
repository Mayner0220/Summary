# Text Processing (5)

Source: https://wikidocs.net/21694, https://wikidocs.net/21698, https://wikidocs.net/21693, https://wikidocs.net/21707, https://wikidocs.net/22530, https://wikidocs.net/21703, https://wikidocs.net/31766, https://wikidocs.net/22647, https://wikidocs.net/22592, https://wikidocs.net/33274
NLP에 있어서 Text Processing은 매우 중요한 작업이다.

---

### 8. 단어 분리하기 (Byte Pair Encoding, BPE)

ML을 이용한 NLP의 최종 목표는 기계가 사람 이상의 성능을 내는 것을 기대하는 것이다.
그런데 기계에게 아무리 많은 단어를 학습시켜도, 세상의 모든 단어를 알려줄 수는 없다.
그리고 더 많은 단어를 알려주려고 하면 그 만큼 계산 비용이 늘어난다는 부담이 있다.

기계가 훈련 단계에서 학습한 단어들의 집합을 단어 집합이라고 한다.
그리고 테스트 단계에서 기계가 미처 배우지 못한 모르는 단어가 등장하면 그 단어를 단어 집합에 없는 단어란 의미에서OOV 또는 UNK(Unknown Token)라고 표현한다.
기계가 문제를 풀 때, 모르는 단어가 등장하면 주어진 문제를 푸는 것이 훨씬 어려워진다.
이와 같이 모르는 단어로 인해 문제를 제대로 풀지 못하는 상황을 OOV 문제라고 한다.

단어 분리(Subword segmenation) 작업은 하나의 단어는 의미있는 여러 내부 단어들(subwords)의 조합으로 구성된 경우가 많기에, 하나의 단어를 여러 내부 단어로 분리해서 단어를 이해해보겠다는 의도를 가진 경우가 많기 때문에, 하나의 단어를 여러 내부 단어로 분리해서 단어를 이해해보겠다는 의도를 가진 전처리 작업이다.
실제로, 언어의 특성에 따라 영어권 언어나 한국어는 단어 분리를 시도했을 때 어느정도 의미있는 단위로 나누는 것이 가능하다.

단어 분리는 기계가 아직 배운 적이 없는 단어에 대해서 어느 정도 대처할 수 있도록 하며, 기계 번역 등에서 주요 전처리로 사용되고 있다.
지금부터 OOV 문제를 완화하는 대표적인 단어 분리 토크나이저인 BPE(Byte Pair Encoding) 알고리즘과 실무에서 사용할 수 있는 단어 분리 토크나이저 구현체인 센텐스피스(Sentencepiece)에 대해 알아보자.

---

### 8.1 BPE

BPE 알고리즘은 1994년에 제안된 데이터 압축 알고리즘이다.

```python
aaabdaaabac
```

BPE는 기본적으로 연속적으로 가장 많이 등장한 글자의 쌍을 찾아서 하나의 글자로 병합하는 방식을 수행한다.
태생이 압축 알고리즘인 만큼, 여기서는 글자 대신 바이트(byte)라는 표현을 사용하도록 하자.
예를 들어 위의 문자열 중 가장 자주 등장하고 있는 바이트의 쌍(byte pair)은 'aa'이다.
이 'aa'라는 바이트의 쌍을 하나의 바이트인 'Z'로 치환해보자.

```python
ZabdZabac
Z=aa
```

이제 위 문자열 중에서 가장 많이 등장하고 있는 바이트의 쌍은 'ab'이다.
이제 이 'ab'를 'Y'로 치환해봅시다.

```python
ZYdZYac
Y=ab
Z=aa
```

이제 가장 많이 등장하고 있는 바이트의 쌍은 'ZY'이다.
이를 'X'로 치환해보자.

```python
XdXac
X=ZY
Y=ab
Z=aa
```

이제 더 이상 병합할 바이트의 쌍은 없으므로 BPE는 위의 결과를 최종 결과로 하여 종료된다.

---

### 8.2 NLP에서의 BPE

*논문 주소: 논문 : https://arxiv.org/pdf/1508.07909.pdf

NLP에서의 BPE는 단어 분리 알고리즘이다.
기존에 있던 단어를 분리한다는 의미이다.
BPE를 요약하자면, 글자 단위에서 점차적으로 단어 집합을 만들어 내는 Bottom up 방식의 접근을 사용한다.
우선 훈련 데이터에 있는 단어들을 모든 글자 또는 유니코드 단위로 단어 집합을 만들고, 가장 많이 등장하는 유니그램을 하나의 유니그램으로 통합합니다.

BPE을 NLP에 사용한다고 제안한 논문(Sennrich et al. (2016))에서 이미 BPE의 코드를 공개하였기 때문에, 바로 파이썬 실습이 가능하다.
코드 실습을 진행하기 전에 육안으로 확인할 수 있는 간단한 예를 들어보자.

1. 기존의 접근
   어떤 훈련 데이터로부터 각 단어들의 빈도수를 카운트했다고 가정해보자.
   그리고 각 단어와 각 단어의 빈도수가 기록되어져 있는 해당 결과는 임의로 딕셔너리란 이름을 붙였다.

   ```python
   # dictionary
   # 훈련 데이터에 있는 단어와 등장 빈도수
   low : 5, lower : 2, newest : 6, widest : 3
   ```

   이 훈련 데이터에는 'low'란 단어가 등장하였고, 'lower'란 단어는 2회 등장하였으며, 'newest'란 단어는 6회, 'widest'란 단어는 3회 등장했다는 의미이다.
   그렇다면 딕셔너리로부터 이 훈련 데이터의 단어 집합을 얻는 것은 간단하다.

   ```python
   # vocabulary
   low, lower, newest, widest
   ```

   단어 집합은 중복을 배제한 단어들의 집합을 의미하므로 기존에 배운 단어 집합의 정의라면, 이 훈련 데이터의 단어 집합에는 'low', 'lower', 'newest', 'widest'라는 4개의 단어가 존재한다.
   그리고 이 경우 테스트 과정에서 'lowest'란 단어가 등장한다면 기계는 이 단어를 학습한 적이 없으므로 해당 단어에 대해서 제대로 대응하지 못하는 OOV 문제가 발생한다.
   그렇다면 BPE를 적용한다면 어떻게 될까?

2. BPE 알고리즘을 사용한 경우
   이제 위의 딕셔너리에 BPE를 적용해보자.
   우선 딕셔너리의 모든 단어들을 글자 단위로 분리한다.
   이 경우 딕셔너리는 아래와 같다.
   이제부터 딕셔너리는 자신 또한 업데이트되며 앞으로 단어 집합을 업데이트하기 위해 지속적으로 참고되는 참고 자료의 역할 한다.

   ```python
   # dictionary
   l o w : 5,  l o w e r : 2,  n e w e s t : 6,  w i d e s t : 3
   ```

   딕셔너리를 참고로 한 초기 단어 집합은 아래와 같다.
   간단히 말해 초기 구성은 글자 단위로 분리된 상태이다.

   ```python
   # vocabulary
   l, o, w, e, r, n, w, s, t, i, d
   ```

   BPE의 특징은 알고리즘의 동작을 몇 회 반복(iteration)할 것인지를 사용자가 정한다는 점이다.
   여기서는 총 10회를 수행한다고 가정하자.
   다시 말해 가장 빈도수가 높은 유니그램의 쌍을 하나의 유니그램으로 통합하는 과정을 총 10회 반복한다.
   위의 딕셔너리에 따르면 빈도수가 현재 가장 높은 유니그램의 쌍을 하나의 유니그램으로 통합하는 과정을 총 10회 반복한다.
   위의 딕셔너리에 따르면 빈도수가 현재 가장 높은 유니그램의 쌍은 (e, s)이다.

   1회 - 딕셔너리를 참고로 하였을 때, 빈도수가 9로 가장 높은 (e, s)의 쌍을 es로 통합한다.

   ```python
   # dictionary update!
   l o w : 5,
   l o w e r : 2,
   n e w es t : 6,
   w i d es t : 3
   ```

   ```python
   # vocabulary update!
   l, o, w, e, r, n, w, s, t, i, d, es
   ```

   2회 - 빈도수가 9로 가장 높은 (es, t)의 쌍을 est로 통합한다.

   ```python
   # dictionary update!
   l o w : 5,
   l o w e r : 2,
   n e w est : 6,
   w i d est : 3
   ```

   ```python
   # vocabulary update!
   l, o, w, e, r, n, w, s, t, i, d, es, est
   ```

   3회 - 빈도수가 7로 가장 높은 (l, o)의 쌍을 lo로 통합한다.

   ```python
   # dictionary update!
   lo w : 5,
   lo w e r : 2,
   n e w est : 6,
   w i d est : 3
   ```

   ```python
   # vocabulary update!
   l, o, w, e, r, n, w, s, t, i, d, es, est, lo
   ```

   이와 같은 방식으로 총 10회 반복하였을 때, 얻은 딕셔너리와 간어 집합은 아래와 같다.

   ```python
   # dictionary update!
   low : 5,
   low e r : 2,
   newest : 6,
   widest : 3
   ```

   ```python
   # vocabulary update!
   l, o, w, e, r, n, w, s, t, i, d, es, est, lo, low, ne, new, newest, wi, wid, widest
   ```

   이 경우 테스트 과정에서 'lowest'란 단어가 등장한다면, 기존에는 OOV에 해당되는 단어가 되었겠지만 BPE 알고리즘을 사용한 위의 단어 집합에서는 더 이상 'lowest'는 OOV가 아니다.
   기계는 우선 'lowest'를 전부 글자 단위로 분할한다.
   즉, 'l, o, w, e, s, t'가 된다.
   그리고 기계는 위의 단어 집합을 참고로 하여, 'low'와 'est'를 찾아낸다.
   즉, 'lowest'를 기계는 'low'와 'est' 두 단어로 인코딩한다.
   그리고 이 두 단어는 둘 다 집합에 있는 단어이므로 OOV가 아니다.

3. 코드 실습하기
   논문에서 공개한 코드를 통해 실습해보자.
   우선 필요한 툴들을 import하자.

   ```python
   import re, collections
   ```

   BPE을 몇 회 수행할 것인지를 정한다.
   여기서는 10회로 정했다.

   ```python
   num_merges = 10
   ```

   BPE에 사용할 단어가 low, lower, newest, widest일 때, BPE의 입력으로 사용하는 실제 단어 집합은 아래와 같다.
   </w>는 단어의 맨 끝에 붙이는 특수 문자이며, 각 단어는 글자 단위로 분리한다.

   ```python
   vocab = {'l o w </w>' : 5,
            'l o w e r </w>' : 2,
            'n e w e s t </w>':6,
            'w i d e s t </w>':3
            }
   ```

   BPE의 코드는 아래와 같다.
   알고리즘은 위에서 설명했던 것과 동일하게 가장 빈도수가 높은 유니그램의 쌍을 하나의 유니그램으로 통합하는 과정으로 `num_merges`회 반복한다.

   ```python
   def get_stats(vocab):
       pairs = collections.defaultdict(int)
       for word, freq in vocab.items():
           symbols = word.split()
           for i in range(len(symbols)-1):
               pairs[symbols[i],symbols[i+1]] += freq
       return pairs
   
   def merge_vocab(pair, v_in):
       v_out = {}
       bigram = re.escape(' '.join(pair))
       p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
       for word in v_in:
           w_out = p.sub(''.join(pair), word)
           v_out[w_out] = v_in[word]
       return v_out
   
   for i in range(num_merges):
       pairs = get_stats(vocab)
       best = max(pairs, key=pairs.get)
       vocab = merge_vocab(best, vocab)
       print(best)
   ```

   이를 실행하면 출력 결과는 아래와 같으며, 이는 글자들의 과정을 보여주고 있다.

   ```python
   ('e', 's')
   ('es', 't')
   ('est', '</w>')
   ('l', 'o')
   ('lo', 'w')
   ('n', 'e')
   ('ne', 'w')
   ('new', 'est</w>')
   ('low', '</w>')
   ('w', 'i')
   ```

   e와 s의 쌍은 초기 단어 집합에서 총 9회 등장했다.
   그렇기 때문에 es로 통합된다.
   그 다음으로는 es와 t의 쌍을, 그 다음으로는 est와  </w>의 쌍을 통합시킨다.
   빈도수가 가장 높은 순서대로 통합하는 이 과정을 총 `num_merges`회 반복한 것이다.

---

### 8.3 WPM(Wordpiece Model)

*WPM의 아이디어를 제시한 논문 : https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/37842.pdf
*구글이 위 WPM을 변형하여 번역기에 사용했다는 논문 : https://arxiv.org/pdf/1609.08144.pdf

기존의 BPE 외에도 WPM아너 Unigram Language Model Tokenizer와 같은 단어 분리 토크나이저들이 존재한다.
여기서는 WPM에 대해서만 간략히 언급한다.
구글은 WPM을 일종의 BPE의 변형으로 소개한다.
WPM은 BPE와 달리 빈도수가 아니라 우도(likehood)를 통해서 단어를 분리한다.

- WPM을 수행하기 이전의 문장: Jet makers feud over seat width with big orders at stake
  
- WPM을 수행한 결과(wordpieces): _J et _makers _fe ud _over _seat _width _with _big _orders _at _stake**

Jet는 J와 et로 나누어졌으며, feud는 fe와 ud로 나누어진 것을 볼 수 있다.
WPM은 모든 단어의 맨 앞에 '_'를 붙이고, 단어는 내부단어로 통계에 기반하여 띄어쓰기로 분리한다.
여기서 언더바는 문장 복원을 위한 장치이다.
예컨대, WPM의 결과로 나온 문장을 보면, Jet → _J ey와 같이 기존에 없던 띄어쓰기가 추가되어 내부 단어들을 구분하는 구분자 역할을 하고 있다.
그렇다면 기존에 있던 띄어쓰기와 구분자 역할의 띄어쓰기는 어떻게 구별할까?
이 역할을 수행하는 것이 단어들 앞에 붙은 언더바이다.
WPM이 수행된 결과로부터 다시 수행 전의 결과로 돌리는 방법은 현재 있는 모든 띄어쓰기를 전부 제거하고, 언더바를 띄어쓰기로 바꾸면 된다.

---

### 8.4 센텐스피스(Sentencepiece)

*논문 : https://arxiv.org/pdf/1808.06226.pdf
*센텐스피스 깃허브 : https://github.com/google/sentencepiece

결론적으로 실무에서 단어 분리를 위해서 어떤 구현체를 사용해야 하냐고 묻는다면, 구글의 센텐스피스를 사용한다.
구글은 BPE 알고리즘과 Unigram Language Model Tokenizer를 구현한 센텐스피스를 깃허브에 공개했다.
기존의 BPE 알고리즘 논문 저자 또한 BPE 코드를 깃허브 공개하기는 했지만, 이를 실무에 적용하기에는 속도가 매우 느리기에 권장하지는 않는다.

센텐스피스의 이점은 또 있다.
단어 분리 알고리즘을 사용하기 위해서, 데이터에 단어 토큰화를 먼저 진행한 상태여야 한다면 이 단어 분리 알고리즘을 모든 언어에사용하는 것은 쉽지 않다.
영어와 달리 한국어와 같은 언어는 단어 토큰화부터가 쉽지 않기 때문이다.
그런데, 이런 사전 토큰화 작업(pretokenization) 없이 전처리를 하지 않은 데이터에 바로 단어 분리 토크나이저를 사용할 수 있다면, 이 토크나이저는 그 어떤 언어에도 적용할 수 있는 토크나이저가 될 것이다.
센텐스피스는 이 이점을 살려서 구현되었다.
센텐스피스는 사전 토큰화 작업 없이 단어 분리 토큰화를 수행하므로 언어 종속되지 않는다.

---

### 9. 데이터의 분리(Splitting Data)

ML/DL 모델에서 데이터를 훈련시키기 위해서는 데이터를 적절히 분리하는 작업이 필요하다.

---

### 9.1 지도 학습(Survised Learning)

지도 학습의 훈련 데이터는 문제지를 연상케 한다.
지도 학습의 훈련 데이터는 정답이 무엇인지 맞춰 하는 '문제'에 해당되는 데이터와 레이블이라고 부르는 '정답'이 적혀있는 데이터로 구성되어 있다.
쉽게 비유하자면, 기계는 정답이 적혀져 있는 문제지를 문제와 정답을 함께 보면서 열심히 공부하고, 향후에는 정답이 없는 문제에 대해서도 잘 예측해야 한다.

예를 들어 스팸 메일 분류기를 만들기 위한 데이터 같은 경우에는 메일의 내용과 해당 메일이 정상 메일인지, 스팸 메일인지 적혀있는 레이블로 구성되어져 있다.
예를 들어 아래와 같은 형식의 데이터가 약 20,000개 있다고 가정해보자.
이 데티어는 두 개의 열로 구성되는데, 바로 메일의 본문에 해당되는 첫번째 열과 해당 메일이 정상 메일인지 스팸 메일인지 적혀있는 정답에 해당되는 두번째 열이다.
그리고 이러한 데이터 배열이 총 20,000개의 행을 가진다.

| 텍스트(메일의 내용)              | 레이블(스팸 여부) |
| -------------------------------- | ----------------- |
| 당신에게 드리는 마지막 혜택! ... | 스팸 메일         |
| 내일 뵐 수 있을지 확인 부탁...   | 정상 메일         |
| ...                              | ...               |
| (광고) 멋있어질 수 있는...       | 스팸 메일         |

 이해를 쉽게 하기위해서 우리는 기계를 지도하는 선생님의 입장이 되어보자.
기계를 가르치기 위해서 데이터를 총 4개로 나눈다.
우선 메일의 내용이 담긴 첫번째 열을 x에 저장한다.
그리고 메일이 스팸인지 정상인지 정답이 적혀있는 두번째 열을 y에 저장한다.
이제 문제지에 해당되는 20,000개의 x와 정답지에 해당되는 20,000개의 y가 생겼다.

그리고 이제 이 X와 y에 대해서 일부 데이터를 또 다시 분리한다.
이는 문제지를 다 공부하고 나서 실력을 평가하기 위해 테스트용으로 일부로 일부 문제와 정답지를 빼놓는 것이다.
여기서는 2,000개를 분리한다고 가정하자.
이 때, 분리 시에는 여전히 X와 y의 맵핑 관계를 유지해야 한다.
어떤 x(문제)에 대한 어떤 y(정답)인지 바로 찾을 수 있어야 한다.
이렇게 되면 학습용에 해당되는 18,000개의 X, y의 쌍과 테스트용에 해당되는 2,000개의 X, y의 쌍이 생긴다.

- 훈련 데이터
  X_train: 문제지 데이터(학습용 데이터)
  Y_train: 문제지에 대한 정답 데이터(학습용 데이터의 레이블)
- 테스트 데이터
  X_test: 시험지 데이터(테스트용 데이터)
  Y_test: 시험지에 대한 정답 데이터(테스트용 데이터의 레이블)

기계는 이제부터 X_train과 Y_train에 대해서 학습을 한다.
기계는 현 상태에서는 정답지인 Y_train을 볼 수 있기에 18,000개의 문제지 X_train을 보면서 어떤 메일 내용일 때 정상 메일인지 스팸 메일인지를 열심히 규칙을 도출해나가면서 정리해나간다.
그리고 학습을 다 한 기계에게 Y_test는 보여주지 않고, X_test에 대해서 정답을 예측하게 한다.
그리고 기계가 예측한 답과 실제 정답인 Y_test를 비교하면서 기계가 정답을 얼마나 맞췄는지를 평가한다.
이 수치가 기계의 정확도(Accuracy)가 된다.

---

### 9.2 x와 y 분리하기

1. zip 함수를 이용하여 분리하기
   zip() 함수는 동일한 개루를 가지는 시퀸스 자료형에서 각 순서에 등장하는 원소들끼리 묶어주는 역할을 한다.
   리스트의 리스트 구성에서 zip 함수는 X와 y를 분리하는데 유용하다.
   우선 zip 함수가 어떤 역할을 하는지 확인해보도록 하자.

   ```python
   X,y = zip(['a', 1], ['b', 2], ['c', 3])
   
   print(X)
   print(y)
   ```

   ```python
   ('a', 'b', 'c')
   (1, 2, 3)
   ```

   각 데이터에서 첫번째로 등장한 원소들끼리 묶이고, 두번째로 등장한 원소들끼리 묶인 것을 볼 수 있다.

   ```python
   sequences=[['a', 1], ['b', 2], ['c', 3]] # 리스트의 리스트 또는 행렬 또는 뒤에서 배울 개념인 2D 텐서.
   X,y = zip(*sequences) # *를 추가
   
   print(X)
   print(y)
   ```

   ```python
   ('a', 'b', 'c')
   (1, 2, 3)
   ```

   각 데이터에서 첫번째로 등장한 원소들끼리 묶이고, 두번째로 등장한 원소들끼리 묶인 것을 볼 수 있다.
   이를 각각 X데이터와 y데이터로 사용할 수 있다.

2. 데이터프레임을 이용하여 분리하기

   ```python
   import pandas as pd
   
   values = [['당신에게 드리는 마지막 혜택!', 1],
   ['내일 뵐 수 있을지 확인 부탁드...', 0],
   ['도연씨. 잘 지내시죠? 오랜만입...', 0],
   ['(광고) AI로 주가를 예측할 수 있다!', 1]]
   columns = ['메일 본문', '스팸 메일 유무']
   
   df = pd.DataFrame(values, columns=columns)
   df
   ```

   ![img](https://wikidocs.net/images/page/33274/%EB%A9%94%EC%9D%BC.PNG)

   데이터프레임은 열의 이름으로 각 열에 접근이 가능하므로, 이를 이용하면 손쉽게 x 데이터와 y 데이터를 분리할 수 있다.

   ```python
   X=df['메일 본문']
   y=df['스팸 메일 유무']
   ```

   우선 x 데이터를 출력해보자.

   ```python
   print(X)
   ```

   ```python
   0          당신에게 드리는 마지막 혜택!
   1      내일 뵐 수 있을지 확인 부탁드...
   2      도연씨. 잘 지내시죠? 오랜만입...
   3    (광고) AI로 주가를 예측할 수 있다!
   Name: 메일 본문, dtype: object
   ```

   정상적으로 스팸 메일 유무이라는 이름을 가졌던 두번째 열에 대해서만 저장이 된 것을 확인할 수 있다. 

   ```python
   0    1
   1    0
   2    0
   3    1
   Name: 스팸 메일 유무, dtype: int64
   ```

3. Numpy를 이용하여 분리하기

   ```python
   import numpy as np
   ar = np.arange(0,16).reshape((4,4))
   
   print(ar)
   ```

   ```python
   [[ 0  1  2  3]
    [ 4  5  6  7]
    [ 8  9 10 11]
    [12 13 14 15]]
   ```

   ```python
   X=ar[:, :3]
   
   print(X)
   ```

   ```python
   [[ 0  1  2]
    [ 4  5  6]
    [ 8  9 10]
    [12 13 14]]
   ```

   ```python
   y=ar[:,3]
   
   print(y)
   ```

   ```python
   [ 3  7 11 15]
   ```

---

### 9.3 테스트 데이터 분리하기

위에서 X와 y를 분리하는 작업에 대해서 배웠다.
이번에는 이미 X와 y가 분리된 데이터에서 테스트 데이터를 분리하는 과정에 대해서 알아보자.

1. 사이킷 런을 이용하여 분리하기
   여기서 훈련 데이터와 테스트 데이터를 유용하게 나눌 수 있는 하나의 방법을 하나 소개해보겠다.
   사이킷 런은 학습용 데이터와 테스트용 데이터를 분리하게 해주는 train_test_split를 지원한다.

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)
   ```

   - X: 독립 변수 데이터 (배열이나 데이터프레임)
   - y: 종속 변수 데이터 (레이블 데이터)

   

   - test_size: 테스트용 데이터 개수를 지정한다.
     1보다 작은 실수를 기재할 경우, 비율을 나타낸다.
   - train_size: 학습용 데이터의 개수를 지칭한다.
     1보다 작은 실수를 기재할 경우, 비율을 나타낸다.
   - random_state: 난수 시드


   예를 들어보자.

   ```python
   import numpy as np
   from sklearn.model_selection import train_test_split
   X, y = np.arange(10).reshape((5, 2)), range(5)
   # 실습을 위해 임의로 X와 y가 이미 분리 된 데이터를 생성
   print(X)
   print(list(y)) #레이블 데이터
   ```

   ```python
   [[0 1]
    [2 3]
    [4 5]
    [6 7]
    [8 9]]
   [0, 1, 2, 3, 4]
   ```

   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
   ```

   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
   #3분의 1만 test 데이터로 지정.
   #random_state 지정으로 인해 순서가 섞인 채로 훈련 데이터와 테스트 데이터가 나눠진다.
   ```

   ```python
   print(X_train)
   print(X_test)
   ```

   ```python
   [[2 3]
    [4 5]
    [6 7]]
   [[8 9]
    [0 1]]
   ```

   ```python
   print(y_train)
   print(y_test)
   ```

   ```python
   [1, 2, 3]
   [4, 0]
   ```

2. 수동으로 분리하기
   데이터를 분리하는 방법 중 하나는 수동으로 분리하는 것이다.
   우선 임의로 X 데이터와 y 데이터를 만들어보자.

   ```python
   import numpy as np
   X, y = np.arange(0,24).reshape((12,2)), range(12)
   # 실습을 위해 임의로 X와 y가 이미 분리 된 데이터를 생성
   ```

   ```python
   print(X)
   ```

   ```python
   [[ 0  1]
    [ 2  3]
    [ 4  5]
    [ 6  7]
    [ 8  9]
    [10 11]
    [12 13]
    [14 15]
    [16 17]
    [18 19]
    [20 21]
    [22 23]]
   ```

   ```python
   print(list(y))
   ```

   ```python
   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
   ```

   이제 훈련 데이터의 개수와 테스트 데이터의 개수를 정해보자.
   n_of_train은 훈련 데이터의 개수를 의미하며, n_of_test는 테스트 데이터의 개수를 의미한다.

   ```python
   n_of_train = int(len(X) * 0.8) # 데이터의 전체 길이의 80%에 해당하는 길이값을 구한다.
   n_of_test = int(len(X) - n_of_train) # 전체 길이에서 80%에 해당하는 길이를 뺀다.
   
   print(n_of_train)
   print(n_of_test)
   ```

   ```python
   9
   3
   ```

   주의할 점은 아직 훈련 데이터와 테스트 데이터를 나눈 것이 아니라, 이 두 개의 개수를 몇 개로 할지 정하기만 한 상태이다.

   또한 여기서 n_of_train을 len(X)*0.8로 구했듯이 n_of_test 또한 len(X)*0.2로 계산하면 되지 않을까하고 생각할 수 있지만, 그렇게 할 경우에는 데이터에 누락이 발생한다.
   예를 들어, 전체 데이터의 개수가 4,518이라고 가정했을 때, 4,518의 80%의 값은 3,614.4로 소수점을 내리면 3,614가 된다.
   또한 4,518의 20%의 값은 903.6으로 소수점을 내리면 903이 된다.
   그리고 3,614+903=4517이므로 데이터 1개가 누락된 것을 알 수 있다.
   그러므로 어느 한 쪽을 먼저 계산하고 그 값만큼 제외하는 방식으로 계산해야 한다.

   ```python
   X_test = X[n_of_train:] #전체 데이터 중에서 20%만큼 뒤의 데이터 저장
   y_test = y[n_of_train:] #전체 데이터 중에서 20%만큼 뒤의 데이터 저장
   X_train = X[:n_of_train] #전체 데이터 중에서 80%만큼 앞의 데이터 저장
   y_train = y[:n_of_train] #전체 데이터 중에서 80%만큼 앞의 데이터 저장1. 
   ```

   실제로 데이터를 나눌 때도 n_of_train와 같이 하나의 변수만 사용하면 데이터의 누락을 방지할 수 있다.
   앞에서 구한 데이터의 개수만큼 훈련 데이터와 테스트 데이터를 분할합니다.

   ```python
   print(X_test)
   print(list(y_test))
   ```

   제대로 분할이 되었는지 책의 지면의 한계상 테스트 데이터만 출력해보자.

   ```python
   [[18 19]
    [20 21]
    [22 23]]
   [9, 10, 11]
   ```

   각각 길이가 3인 것을 보아, 제대로 분할된 것을 알 수 있다.



