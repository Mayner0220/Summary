# Recurrent Neural Network (2)

source: https://wikidocs.net/48558, https://wikidocs.net/22886, https://wikidocs.net/22888, https://wikidocs.net/22889, https://wikidocs.net/46496, https://wikidocs.net/45101, https://wikidocs.net/48649

---

### 5. RNN을 이용한 텍스트 생성(Text Generation using RNN)

다 대 일(many-to-one) 구조의 RNN을 사용하여 문맥을 반영해서 텍스트를 생성하는 모델을 구현해보자.

---

### 5.1 RNN을 이용하여 텍스트 생성하기

예를 들어서 '경마자에 있는 말이 뛰고 있다'와 '그의 말이 법이다'와 '가는 말이 고와야 오는 말이 곱다'라는 세 가지 문장이 있다고 해보자.
모델이 문맥을 학습할 수 있도록 전체 문장의 앞의 단어들을 전부 고려하여 학습하도록 데이터를 재구성한다면 아래와 같이 총 11개의 샘플이 구성된다.

| sample |             X              |   y    |
| :----: | :------------------------: | :----: |
|   1    |          경마장에          |  있는  |
|   2    |       경마장에 있는        |  말이  |
|   3    |     경마장에 있는 말이     |  뛰고  |
|   4    |  경마장에 있는 말이 뛰고   |  있다  |
|   5    |            그의            |  말이  |
|   6    |         그의 말이          | 법이다 |
|   7    |            가는            |  말이  |
|   8    |         가는 말이          | 고와야 |
|   9    |      가는 말이 고와야      |  오는  |
|   10   |   가는 말이 고와야 오는    |  말이  |
|   11   | 가는 말이 고와야 오는 말이 |  곱다  |

1. 데이터에 대한 이해와 전처리

   ```python
   from tensorflow.keras.preprocessing.text import Tokenizer
   from tensorflow.keras.preprocessing.sequence import pad_sequences
   import numpy as np
   from tensorflow.keras.utils import to_categorical
   ```

   우선 예제로 언급한 3개의 한국어 문장을 저장한다.

   ```python
   text="""경마장에 있는 말이 뛰고 있다\n
   그의 말이 법이다\n
   가는 말이 고와야 오는 말이 곱다\n"""
   ```

   단어 집합을 생성하고 크기를 확인해보자.

   ```python
   t = Tokenizer()
   t.fit_on_texts([text])
   vocab_size = len(t.word_index) + 1
   # 케라스 토크나이저의 정수 인코딩은 인덱스가 1부터 시작하지만,
   # 케라스 원-핫 인코딩에서 배열의 인덱스가 0부터 시작하기 때문에
   # 배열의 크기를 실제 단어 집합의 크기보다 +1로 생성해야하므로 미리 +1 선언 
   print('단어 집합의 크기 : %d' % vocab_size)
   ```

   ```python
   단어 집합의 크기 : 12
   ```

   각 단어와 단어에 부여된 정수 인덱스를 출력해보자.

   ```python
   print(t.word_index)
   ```

   ```python
   {'말이': 1, '경마장에': 2, '있는': 3, '뛰고': 4, '있다': 5, '그의': 6, '법이다': 7, '가는': 8, '고와야': 9, '오는': 10, '곱다': 11}
   ```

   이제 훈련 데이터를 만들어보자.

   ```python
   sequences = list()
   for line in text.split('\n'): # Wn을 기준으로 문장 토큰화
       encoded = t.texts_to_sequences([line])[0]
       for i in range(1, len(encoded)):
           sequence = encoded[:i+1]
           sequences.append(sequence)
   
   print('학습에 사용할 샘플의 개수: %d' % len(sequences))
   ```

   샘플의  개수는 총 11개가 나온다.
   전체 셈플을 출력해보자.

   ```python
   print(sequences)
   ```

   ```python
   [[2, 3], [2, 3, 1], [2, 3, 1, 4], [2, 3, 1, 4, 5], [6, 1], [6, 1, 7], [8, 1], [8, 1, 9], [8, 1, 9, 10], [8, 1, 9, 10, 1], [8, 1, 9, 10, 1, 11]]
   ```

   위의 데이터는 아직 레이블로 사용될 단어를 분리하지 않은 훈련 데이터이다.
   [2, 3]은 [경마장에, 있는]에 해당되며 [2, 3, 1]은 [경마장에, 있는, 말이]에 해당된다.
   전체 훈련 데이터에 대해서 맨 우측에 있는 단어에 대해서만 레이블로 분리해야 한다.

   우선 전체 샘플에 대해서 길이를 일치시켜 준다.
   가장 긴 샘플의 길이를 기준으로 한다.
   현재 육안으로 봤을 때, 길이가 가장 긴 샘플은 [8, 1,  9, 10, 1, 11]이고 길이는 6이다.
   이를 코드로는 다음과 같이 구할 수 있다.

   ```python
   max_len=max(len(l) for l in sequences) # 모든 샘플에서 길이가 가장 긴 샘플의 길이 출력
   print('샘플의 최대 길이 : {}'.format(max_len))
   ```

   ```python
   샘플의 최대 길이 : 6
   ```

   전체 훈련 데이터에서 가장 긴 샘플의 길이가 6임을 확인했다.
   이제 전체 샘플의 길이를 6으로 패딩한다.

   ```python
   sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
   ```

   pad_sequences()는 모든 샘플에 댛서 0을 사용하여 길이를 맞춰준다.
   maxlen의 값으로 6을 주면 모든 샘플의 길이를 6으로 맟춰주며, padding의 인자로 'pre'를 주면 길이가 6보다 짧은 샘플의 앞에 0으로 채운다.
   전체 훈련 데이터를 출력해보자.

   ```python
   print(sequences)
   ```

   ```python
   [[ 0  0  0  0  2  3]
    [ 0  0  0  2  3  1]
    [ 0  0  2  3  1  4]
    [ 0  2  3  1  4  5]
    [ 0  0  0  0  6  1]
    [ 0  0  0  6  1  7]
    [ 0  0  0  0  8  1]
    [ 0  0  0  8  1  9]
    [ 0  0  8  1  9 10]
    [ 0  8  1  9 10  1]
    [ 8  1  9 10  1 11]]
   ```

   길이가 6보다 짧은 모든 샘플에 대해서 앞에 0을 채워서 모든 샘플의 길이를 6으로 바꿨다.
   이제 각 샘플의 마지막 단어를 레이블로 분리한다.
   레이블의 분리는 Numpy를 이용해서 가능하다.

   ```python
   sequences = np.array(sequences)
   X = sequences[:,:-1]
   y = sequences[:,-1]
   # 리스트의 마지막 값을 제외하고 저장한 것은 X
   # 리스트의 마지막 값만 저장한 것은 y. 이는 레이블에 해당됨.
   ```

   분리된 X와 y에 대해서 출력해보면 다음과 같다.

   ```python
   print(X)
   ```

   ```python
   [[ 0  0  0  0  2]
    [ 0  0  0  2  3]
    [ 0  0  2  3  1]
    [ 0  2  3  1  4]
    [ 0  0  0  0  6]
    [ 0  0  0  6  1]
    [ 0  0  0  0  8]
    [ 0  0  0  8  1]
    [ 0  0  8  1  9]
    [ 0  8  1  9 10]
    [ 8  1  9 10  1]]
   ```

   ```python
   print(y) # 모든 샘플에 대한 레이블 출력
   ```

   ```python
   [ 3  1  4  5  1  7  1  9 10  1 11]
   ```

   레이블이 분리되었다.
   이제 RNN 모델에 훈련 데이터를 훈련 시키기 전에 레이블에 대해서 원-핫 인코딩을 수행한다.

   ```python
   y = to_categorical(y, num_classes=vocab_size)
   ```

   원-핫 인코딩이 수행되었는지 출력한다.

   ```python
   print(y)
   ```

   ```python
   [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.] # 3에 대한 원-핫 벡터
    [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] # 1에 대한 원-핫 벡터
    [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.] # 4에 대한 원-핫 벡터
    [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.] # 5에 대한 원-핫 벡터
    [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] # 1에 대한 원-핫 벡터
    [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] # 7에 대한 원-핫 벡터
    [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] # 1에 대한 원-핫 벡터
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] # 9에 대한 원-핫 벡터
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.] # 10에 대한 원-핫 벡터
    [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] # 1에 대한 원-핫 벡터
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]] # 11에 대한 원-핫 벡터
   ```

   정상적으로 원-핫 인코딩이 수행된 것을 볼 수 있다.

2. 모델 설계하기
   이제 RNN 모델에 데이터를 훈련시킨다.

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
   ```

   ```python
   model = Sequential()
   model.add(Embedding(vocab_size, 10, input_length=max_len-1)) # 레이블을 분리하였으므로 이제 X의 길이는 5
   model.add(SimpleRNN(32))
   model.add(Dense(vocab_size, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.fit(X, y, epochs=200, verbose=2)
   ```

   각 단어의 임베딩 벡터는 10차원을 가지고, 32의 은닉 상태 크기를 가지는 바닐라 RNN을 사용한다.

   ```python
   Epoch 1/200
    - 1s - loss: 2.4945 - acc: 0.0909
   ... 중략 ...
   Epoch 200/200
    - 0s - loss: 0.1299 - acc: 1.0000
   ```

   모델이 정확하게 예측하고 있는지 문장을 생성하는 함수를 만들어서 출력해보자.

   ```python
   def sentence_generation(model, t, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
       init_word = current_word # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
       sentence = ''
       for _ in range(n): # n번 반복
           encoded = t.texts_to_sequences([current_word])[0] # 현재 단어에 대한 정수 인코딩
           encoded = pad_sequences([encoded], maxlen=5, padding='pre') # 데이터에 대한 패딩
           result = model.predict_classes(encoded, verbose=0)
       # 입력한 X(현재 단어)에 대해서 Y를 예측하고 Y(예측한 단어)를 result에 저장.
           for word, index in t.word_index.items(): 
               if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                   break # 해당 단어가 예측 단어이므로 break
           current_word = current_word + ' '  + word # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
           sentence = sentence + ' ' + word # 예측 단어를 문장에 저장
       # for문이므로 이 행동을 다시 반복
       sentence = init_word + sentence
       return sentence
   ```

   이제 입력된 단어로부터 다음 단어를 예측해서 문장을 생성하는 함수를 만들어보자.

   ```python
   print(sentence_generation(model, t, '경마장에', 4))
   # '경마장에' 라는 단어 뒤에는 총 4개의 단어가 있으므로 4번 예측
   ```

   ```python
   경마장에 있는 말이 뛰고 있다
   ```

   ```python
   print(sentence_generation(model, t, '그의', 2)) # 2번 예측
   ```

   ```python
   그의 말이 법이다
   ```

   ```python
   print(sentence_generation(model, t, '가는', 5)) # 5번 예측
   ```

   ```python
   가는 말이 고와야 오는 말이 곱다
   ```

   이제 앞의 문맥을 기준으로 '말이' 라는 단어 다음에 나올 단어를 기존의 훈련 데이터와 일치하게 예측함을 보여준다.
   이 모델은 충분한 훈련 데이터를 갖고 있지 못하므로 위에서 문장의 길이에 맞게 적절하게 예측해야하는 횟수 4, 2, 5를 각각 인자값으로 주었다.
   이 이상의 숫자를 주면 기계는 '있다', '법이다', '곱다' 다음에 나오는 단어가 무엇인지 배운 적이 없으므로 임의 예측을 한다.
   이번에는 더 많은 훈련 데이터를 가지고 실습해보자.

---

### 5.3 LSTM을 이용하여 텍스트 생성하기

이번에는 LSTM을 통해 보다 많은 데이터로 텍스트를 생성해보자.
본질적으로 앞에서 한 것과 동일한 내용이다.

1. 데이터에 대한 이해와 전처리
   사용항 데이터는 뉴욕 타임즈 기사의 제목이다.
   아래의 링크에서 ArticlesApril2018.csv 데이터를 다운로드한다.

   - 파일 다운로드 링크: https://www.kaggle.com/aashita/nyt-comments

   ```python
   import pandas as pd
   from string import punctuation
   from tensorflow.keras.preprocessing.text import Tokenizer
   from tensorflow.keras.preprocessing.sequence import pad_sequences
   import numpy as np
   from tensorflow.keras.utils import to_categorical
   ```

   다운로드한 훈련 데이터를 데이터프레임에 저장한다.

   ```python
   df=pd.read_csv('ArticlesApril2018.csv 파일의 경로')
   df.head()
   ```

   열의 개수가 굉장히 많기에 한 눈에 보기 어렵다.
   어떤 열이 있고, 열이 총 몇 개가 있는지 출력해보자.

   ```python
   print('열의 개수: ',len(df.columns))
   print(df.columns)
   ```

   ```python
   열의 개수:  15
   Index(['articleID', 'articleWordCount', 'byline', 'documentType', 'headline',
          'keywords', 'multimedia', 'newDesk', 'printPage', 'pubDate',
          'sectionName', 'snippet', 'source', 'typeOfMaterial', 'webURL'], dtype='object')
   ```

   총 15개의 열이 존재한다.
   여기서 사용할 열은 제목에 해당되는 headline 열이다.
   Null 값이 있는지 확인해보자.

   ```python
   df['headline'].isnull().values.any()
   ```

   ```python
   False
   ```

   Null 값은 별도로 없는 것으로 보인다.
   headline 열에서 모든 신문 기사의 제목을 뽑아서 하나의 리스트로 저장해보자.

   ```python
   headline = [] # 리스트 선언
   headline.extend(list(df.headline.values)) # 헤드라인의 값들을 리스트로 저장
   headline[:5] # 상위 5개만 출력
   ```

   headline이라는 리스트에 모든 신문 기사의 제목을 저장했다.
   저장한 리스트에서 상위 5개만 출력해보자.

   ```python
   ['Former N.F.L. Cheerleaders’ Settlement Offer: $1 and a Meeting With Goodell',
    'E.P.A. to Unveil a New Rule. Its Effect: Less Science in Policymaking.',
    'The New Noma, Explained',
    'Unknown',
    'Unknown']
   ```

   그런데 4번째, 5번째 샘프에 Unknown 값이 들어가 있다.
   headline 전체에 걸쳐서 Unknown 값을 가진 샘플이 있을 것으로 추정된다.
   비록 Null 값은 아니지만 지금 하고자 하는 실습에 되지 않는 노이즈 데이터이므로 제거해줄 필요가 있다.
   제거하기 전에 현재 샘플의 개수를 확인해보고 제거 전, 후의 샘플의 개수를 비교해보자.

   ```python
   print('총 샘플의 개수 : {}'.format(len(headline)) # 현재 샘플의 개수
   총 샘플의 개수 : 1324
   ```

   ```python
   총 샘플의 개수 : 1324
   ```

   노이즈 데이터를 제거하기 전 데이터의 개수는 1,324이다.
   즉, 신문 기사의 제목이 총 1,324개이다.

   ```python
   headline = [n for n in headline if n != "Unknown"] # Unknown 값을 가진 샘플 제거
   print('노이즈값 제거 후 샘플의 개수 : {}'.format(len(headline)) # 제거 후 샘플의 개수
   ```

   ```python
   노이즈값 제거 후 샘플의 개수 : 1214
   ```

   샘플의 수가 1,324에서 1,214로 110개의 샘플이 제거되었는데, 기존에 출력했던 5개의 샘플을 출력해보자.

   ```python
   headline[:5]
   ```

   ```python
   ['Former N.F.L. Cheerleaders’ Settlement Offer: $1 and a Meeting With Goodell',
    'E.P.A. to Unveil a New Rule. Its Effect: Less Science in Policymaking.',
    'The New Noma, Explained',
    'How a Bag of Texas Dirt  Became a Times Tradition',
    'Is School a Place for Self-Expression?']
   ```

   기존에 4번째, 5번째 샘플에서는 Unknown 값이 있었는데 현재 제거가 된 것을 확인하였다.
   이제 데이터 전처리를 수행한다.
   여기서 선택한 전처리는 구두점 제거와 단어의 소문자화이다.
   전처리를 수행하고, 다시 샘플 5개를 출력한다.

   ```python
   def repreprocessing(s):
       s=s.encode("utf8").decode("ascii",'ignore')
       return ''.join(c for c in s if c not in punctuation).lower() # 구두점 제거와 동시에 소문자화
   
   text = [repreprocessing(x) for x in headline]
   text[:5]
   ```

   ```python
   ['former nfl cheerleaders settlement offer 1 and a meeting with goodell',
    'epa to unveil a new rule its effect less science in policymaking',
    'the new noma explained',
    'how a bag of texas dirt  became a times tradition',
    'is school a place for selfexpression']
   ```

   기존의 출력과 비교하면 모든 단어들이 소문자화되었으며 N.F.L이나 Cheerleaders’ 등과 같이 기존에 구두점이 붙어있던 단어들에서 구두점이 제거되었다.
   이제 단어 집합(vocabulary)을 만들고 크기를 확인한다.

   ```python
   t = Tokenizer()
   t.fit_on_texts(text)
   vocab_size = len(t.word_index) + 1
   print('단어 집합의 크기 : %d' % vocab_size)
   ```

   ```python
   단어 집합의 크기 : 3494
   ```

   총 3,494개의 단어가 존재한다.
   이제 정수 인코딩과 동시에 하나의 문장을 여러 줄로 분해하여 훈련 데이터를 구성한다.

   ```python
   sequences = list()
   
   for line in text: # 1,214 개의 샘플에 대해서 샘플을 1개씩 가져온다.
       encoded = t.texts_to_sequences([line])[0] # 각 샘플에 대한 정수 인코딩
       for i in range(1, len(encoded)):
           sequence = encoded[:i+1]
           sequences.append(sequence)
   
   sequences[:11] # 11개의 샘플 출력
   ```

   ```python
   [[99, 269], # former nfl
    [99, 269, 371], # former nfl cheerleaders
    [99, 269, 371, 1115], # former nfl cheerleaders settlement
    [99, 269, 371, 1115, 582], # former nfl cheerleaders settlement offer
    [99, 269, 371, 1115, 582, 52], # 'former nfl cheerleaders settlement offer 1
    [99, 269, 371, 1115, 582, 52, 7], # former nfl cheerleaders settlement offer 1 and
    [99, 269, 371, 1115, 582, 52, 7, 2], # ... 이하 생략 ...
    [99, 269, 371, 1115, 582, 52, 7, 2, 372],
    [99, 269, 371, 1115, 582, 52, 7, 2, 372, 10],
    [99, 269, 371, 1115, 582, 52, 7, 2, 372, 10, 1116], # 모든 단어가 사용된 완전한 첫번째 문장
    # 바로 위의 줄 : former nfl cheerleaders settlement offer 1 and a meeting with goodell
    [100, 3]] # epa to에 해당되며 두번째 문장이 시작됨.
   ```

   이해를 돕기 위해 출력 결과에 주석을 추가했다.
   왜 하나의 문장을 저렇게 나눌까.
   예를 들어 '경마장에 있는 말이 뛰고 있다'라는 문장 하나가 있을 때, 최종적으로 원하는 훈련 데이터의 형태는 다음과 같다.
   하나의 단어를 예측하기 위해 이전에 등장한 단어들을 모두 참고하는 것이다.

   | samples |            X            |  y   |
   | :-----: | :---------------------: | :--: |
   |    1    |        경마장에         | 있는 |
   |    2    |      경마장에 있는      | 말이 |
   |    3    |   경마장에 있는 말이    | 뛰고 |
   |    4    | 경마장에 있는 말이 뛰고 | 있다 |

   위의 sequences는 모든 문장을 각 단어가 각 시점(time step)마다 하나씩 추가적으로 등장하는 형태로 만들기는 했지만, 아직 예측할 단어에 해당되는 레이블을 분리하는 작업까지는 수행하지 않은 상태이다.
   어떤 정수가 어떤 단어를 의미하는지 알아보기 위해 인덱슬로부터 단어를 찾는 index_to_word를 만든다.

   ```python
   index_to_word={}
   for key, value in t.word_index.items(): # 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
       index_to_word[value] = key
   
   print('빈도수 상위 582번 단어 : {}'.format(index_to_word[582]))
   ```

   ```python
   빈도수 상위 582번 단어 : offer
   ```

   582이라는 인덱스를 가진 단어는 본래 offer이라는 단어였다.
   원한다면 다른 숫자로도 시도해보자.
   이제 y 데이터를 분리하기 전에 전체 샘플의 길이를 동일하게 만드는 패딩 작업을 수행한다.
   패딩 작업을 수행하기 전에 가장 긴 샘플의 길이를 확인한다.

   ```python
   max_len=max(len(l) for l in sequences)
   print('샘플의 최대 길이 : {}'.format(max_len))
   ```

   ```python
   샘플의 최대 길이 : 24
   ```

   가장 긴 샘플의 길이인 24로 모든 샘플의 길이를 패딩하자.

   ```python
   sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
   print(sequences[:3])
   ```

   ```python
   [[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0    0    0   99  269]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0    0   99  269  371]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   99  269  371 1115]
   ```

   padding='pre'를 설정하여 샘플의 길이가 24보다 짧은 경우에 앞에 0으로 패딩되었다.
   이제 맨 우측 단어만 레이블로 분리한다.

   ```python
   sequences = np.array(sequences)
   X = sequences[:,:-1]
   y = sequences[:,-1]
   ```

   ```python
   print(X[:3])
   ```

   ```python
   [[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0    0    0   99]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0    0   99  269]
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   99  269  371]
   ```

   훈련 데이터 X에서 3개의 샘플만 출력해보았는데, 맨 우측에 있던 정수값 269, 371, 1115가 사라진 것을 볼 수 있다.
   뿐만 아니라, 각 샘플의 길이가 24에서 23으로 줄었다.

   ```python
   print(y[:3]) # 레이블
   ```

   ```python
   [ 269  371 1115]
   ```

   훈련 데이터 y 중 3개의 샘플만 출력해보았는데, 기존 훈련 데이터에서 맨 우측에 있던 정수들이 별도로 저장되었다.

   ```python
   y = to_categorical(y, num_classes=vocab_size)
   ```

   레이블 데이터 y에 대해서 원-핫 인코딩을 수행하였다.
   이제 모델을 설계하자.

2. 모델 설계하기

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Embedding, Dense, LSTM
   ```

   ```python
   model = Sequential()
   model.add(Embedding(vocab_size, 10, input_length=max_len-1))
   # y데이터를 분리하였으므로 이제 X데이터의 길이는 기존 데이터의 길이 - 1
   model.add(LSTM(128))
   model.add(Dense(vocab_size, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.fit(X, y, epochs=200, verbose=2)
   ```

   각 단어의 임베딩 벡터는 10차원을 가지고, 128의 은닉 상태 크기를 가지는 LSTM을 사용한다.
   문장을 생성하는 함수 sentence_generation을 만들어서 문장을 생성해보자.

   ```python
   def sentence_generation(model, t, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
       init_word = current_word # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
       sentence = ''
       for _ in range(n): # n번 반복
           encoded = t.texts_to_sequences([current_word])[0] # 현재 단어에 대한 정수 인코딩
           encoded = pad_sequences([encoded], maxlen=23, padding='pre') # 데이터에 대한 패딩
           result = model.predict_classes(encoded, verbose=0)
       # 입력한 X(현재 단어)에 대해서 y를 예측하고 y(예측한 단어)를 result에 저장.
           for word, index in t.word_index.items(): 
               if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                   break # 해당 단어가 예측 단어이므로 break
           current_word = current_word + ' '  + word # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
           sentence = sentence + ' ' + word # 예측 단어를 문장에 저장
       # for문이므로 이 행동을 다시 반복
       sentence = init_word + sentence
       return sentence
   ```

   ```python
   print(sentence_generation(model, t, 'i', 10))
   # 임의의 단어 'i'에 대해서 10개의 단어를 추가 생성
   ```

   ```python
   i disapprove of school vouchers can i still apply for them
   ```

   ```python
   print(sentence_generation(model, t, 'how', 10))
   # 임의의 단어 'how'에 대해서 10개의 단어를 추가 생성
   ```

   ```python
   how to make facebook more accountable will so your neighbor chasing
   ```

---

### 6. 글자 단위 RNN(Char RNN)

이전 시점의 예측 글자를 다음 시점의 입력으로 사용하는 글자 단위 RNN 언어 모델을 구현해보자.
앞서 배운 단어 단위 RNN 언어 모델과 다른 점은 단어 단위가 아니라 글자 단위를 입/출력으로 사용하므로 임베딩층(embedding layer)을 여기서는 사용하지 않는다.
여기서는 언어 모델의 훈련 과정과 테스트 과정의 차이를 이해하는데 초점을 둔다.

- 다운로드 링크:  http://www.gutenberg.org/files/11/11-0.txt

고전 소설들은 저작권에 보호받지 않으므로, 무료로 쉽게 다운로드 받을 수 있는 좋은 훈련 데이터이다.
위의 링크에서 '이상한 나라의 앨리스(Alice's Adventures in Wonderland)'라는 소설을 다운로드한다.
우선, 파일을 불러오고 간단한 전처리를 수행한다.

1. 데이터에 대한 이해와 전처리

   ```python
   import numpy as np
   import urllib.request
   from tensorflow.keras.utils import to_categorical
   ```

   ```python
   urllib.request.urlretrieve("http://www.gutenberg.org/files/11/11-0.txt", filename="11-0.txt")
   f = open('11-0.txt', 'rb')
   lines=[]
   for line in f: # 데이터를 한 줄씩 읽는다.
       line=line.strip() # strip()을 통해 \r, \n을 제거한다.
       line=line.lower() # 소문자화.
       line=line.decode('ascii', 'ignore') # \xe2\x80\x99 등과 같은 바이트 열 제거
       if len(line) > 0:
           lines.append(line)
   f.close()
   ```

   간단한 전처리가 수행된 결과가 lines란 이름의 리스트에 저장되었다.
   리스트에서 5개의 원소만 출력해보자.

   ```python
   lines[:5]
   ```

   ```python
   ['project gutenbergs alices adventures in wonderland, by lewis carroll',
    'this ebook is for the use of anyone anywhere at no cost and with',
    'almost no restrictions whatsoever.  you may copy it, give it away or',
    're-use it under the terms of the project gutenberg license included',
    'with this ebook or online at www.gutenberg.org']
   ```

   각 원소는 문자열로 구성되어져 있는데, 특별히 의미있게 문장 토큰화가 된 상태는 아니다.
   이를 하나의 문자열로 통합하자.

   ```python
   text = ' '.join(lines)
   print('문자열의 길이 또는 총 글자의 개수: %d' % len(text))
   ```

   ```python
   문자열의 길이 또는 총 글자의 개수: 158783
   ```

   하나의 문자열로 통합되었고, 문자열의 길이는 약 15만 8천이다.
   일부를 출력해보자.

   ```python
   print(text[:200])
   ```

   ```python
   project gutenbergs alices adventures in wonderland, by lewis carroll this ebook is for the use of anyone anywhere at no cost and with almost no restrictions whatsoever.  you may copy it, give it away 
   ```

   이 문자열은 어떤 글자로도 구성되어져 있을까.
   이제 이 문자열로부터 글자 집합을 만들어보자.
   기존에는 중복을 제거한 단어들의 모음인 단어 집합(vocabulary)을 만들었으나, 이번에 만들 집합은 단어 집합이 이나라 글자 집합이다.

   ```python
   char_vocab = sorted(list(set(text)))
   vocab_size=len(char_vocab)
   print ('글자 집합의 크기 : {}'.format(vocab_size))
   ```

   ```python
   글자 집합의 크기 : 55
   ```

   영어가 훈련 데이터일 때 대부분의 경우에서 글자 집합의 크기가 단어 집합을 사용했을 경우보다 집합의 크기가 현저히 작다는 특징이 있다.
   아무리 훈련 코퍼스에 수십만 개 이상의 많은 영어 단어가 존재한다고 하더라도, 영어 단어를 표현하기 위해서 글자 집합에 포함되는 글자는 26개의 알파벳뿐이기 때문이다.
   만약 훈련 데이터의 알파벳이 대, 소문자가 구분된 상태라고 하더라도 모든 영어 단어는 총 52개의 알파벳으로 표현이 가능하다.

   어떤 방대한 양의 텍스트라도 집합의 크기를 적게 가져갈 수 있다는 것은 구현과 테스트를 굉장히 쉽게 할 수 있다는 이점을 가지므로, RNN의 동작 메커니즘 이해를 위한 토이 프로젝트로 굉장히 많이 사용된다.
   글자 집합에 인덱스를 부여하고 전부 출력해보자.

   ```python
   char_to_index = dict((c, i) for i, c in enumerate(char_vocab)) # 글자에 고유한 정수 인덱스 부여
   print(char_to_index)
   ```

   ```python
   {' ': 0, '!': 1, '#': 2, '$': 3, '%': 4, '(': 5, ')': 6, '*': 7, ',': 8, '-': 9, '.': 10, '/': 11, '0': 12, '1': 13, '2': 14, '3': 15, '4': 16, '5': 17, '6': 18, '7': 19, '8': 20, '9': 21, ':': 22, ';': 23, '?': 24, '@': 25, '[': 26, ']': 27, '_': 28, 'a': 29, 'b': 30, 'c': 31, 'd': 32, 'e': 33, 'f': 34, 'g': 35, 'h': 36, 'i': 37, 'j': 38, 'k': 39, 'l': 40, 'm': 41, 'n': 42, 'o': 43, 'p': 44, 'q': 45, 'r': 46, 's': 47, 't': 48, 'u': 49, 'v': 50, 'w': 51, 'x': 52, 'y': 53, 'z': 54}
   ```

   인덱스 0부터 28까지는 공백을 포함한 각종 구두점, 특수문자가 존재하고, 인덱스 29부터 54까지는 a부터 z까지 총 26개의 알파벳 소문자가 글자 집합에 포함되어져 있다.
   이제 반대로 인덱스로부터 글자를 리턴하는 index_to_char index_to_char을 만든다.

   ```python
   index_to_char={}
   for key, value in char_to_index.items():
       index_to_char[value] = key
   ```

   훈련 데이터를 구성해보자.
   훈련 데이터 구성을 위한 간소화 된 예를 들어보자.
   훈련 데이터에 apple이라는 시퀀스가 있고, 입력 시퀀스의 길이. 즉, 샘플의 길이를 4라고 한다면 입력 시퀀스와 예측해야 하는 출력 시퀀스는 다음과 같이 구성된다.

   ```python
   # Example) 샘플의 길이가 4라면 4개의 입력 글자 시퀀스로 부터 4개의 출력 글자 시퀀스를 예측. 즉, RNN의 time step은 4번
   appl -> pple
   # appl은 train_X(입력 시퀀스), pple는 train_y(예측해야하는 시퀀스)에 저장한다.
   ```

   이제 15만 8천의 길이를 가진 text 문자열로부터 다수의 문장 샘플들로 분리해보자.
   분리하는 방법은 문장 샘플의 길이를 정하고, 해당 길이만큼 문자열 전체를 전부 등분라는 것이다.

   ```python
   seq_length = 60 # 문장의 길이를 60으로 한다.
   n_samples = int(np.floor((len(text) - 1) / seq_length)) # 문자열을 60등분한다. 그러면 즉, 총 샘플의 개수
   print ('문장 샘플의 수 : {}'.format(n_samples))
   ```

   ```python
   문장 샘플의 수 : 2646
   ```

   만약 문장의 길이를 60으로 한다면 15만 8천을 60으로 나눈 수가 샘플의 수가 된다.
   여기서는 총 샘플의 수가 2,646개이다.
   
   ```python
   train_X = []
   train_y = []
   
   for i in range(n_samples): # 2,646번 수행
       X_sample = text[i * seq_length: (i + 1) * seq_length]
       # 0:60 -> 60:120 -> 120:180로 loop를 돌면서 문장 샘플을 1개씩 가져온다.
       X_encoded = [char_to_index[c] for c in X_sample] # 하나의 문장 샘플에 대해서 정수 인코딩
       train_X.append(X_encoded)
   
       y_sample = text[i * seq_length + 1: (i + 1) * seq_length + 1] # 오른쪽으로 1칸 쉬프트한다.
       y_encoded = [char_to_index[c] for c in y_sample]
       train_y.append(y_encoded)
   ```
   
   train_X와 train_y의 첫번째 샘플과 두번째 샘플을 출력하여 데이터의 구성을 확인해보자.
   
   ```python
   print(train_X[0])
   ```
   
   ```python
   [44, 46, 43, 38, 33, 31, 48, 0, 35, 49, 48, 33, 42, 30, 33, 46, 35, 47, 0, 29, 40, 37, 31, 33, 47, 0, 29, 32, 50, 33, 42, 48, 49, 46, 33, 47, 0, 37, 42, 0, 51, 43, 42, 32, 33, 46, 40, 29, 42, 32, 8, 0, 30, 53, 0, 40, 33, 51, 37, 47]
   ```
   
   ```python
   print(train_y[0])
   ```
   
   ```python
   [46, 43, 38, 33, 31, 48, 0, 35, 49, 48, 33, 42, 30, 33, 46, 35, 47, 0, 29, 40, 37, 31, 33, 47, 0, 29, 32, 50, 33, 42, 48, 49, 46, 33, 47, 0, 37, 42, 0, 51, 43, 42, 32, 33, 46, 40, 29, 42, 32, 8, 0, 30, 53, 0, 40, 33, 51, 37, 47, 0]
   ```
   
   train_y[0]은 train_X[0]에서 오른쪽으로 한 칸 쉬프트 된 문장임을 알 수 있다.
   
   ```python
   print(train_X[1])
   ```
   
   ```python
   [0, 31, 29, 46, 46, 43, 40, 40, 0, 48, 36, 37, 47, 0, 33, 30, 43, 43, 39, 0, 37, 47, 0, 34, 43, 46, 0, 48, 36, 33, 0, 49, 47, 33, 0, 43, 34, 0, 29, 42, 53, 43, 42, 33, 0, 29, 42, 53, 51, 36, 33, 46, 33, 0, 29, 48, 0, 42, 43, 0]
   ```
   
   ```python
   print(train_y[1])
   ```
   
   ```python
   [31, 29, 46, 46, 43, 40, 40, 0, 48, 36, 37, 47, 0, 33, 30, 43, 43, 39, 0, 37, 47, 0, 34, 43, 46, 0, 48, 36, 33, 0, 49, 47, 33, 0, 43, 34, 0, 29, 42, 53, 43, 42, 33, 0, 29, 42, 53, 51, 36, 33, 46, 33, 0, 29, 48, 0, 42, 43, 0, 31]
   ```
   
   마찬가지로 train_y[1]은 train_X[1]에서 오른쪽으로 한 칸 쉬프트 된 문장임을 알 수 있다.
   이제 train_X와 train_y에 대해서 원-핫 인코딩을 수행한다.
   글자 단위 RNN에서는 입력 시퀀스에 대해서 워드 임베딩을 하지 않는다.
   다시 말해 임베딩층(embedding layer)을 사용하지 않을 것이므로, 입력 시퀀스인 train_X에 대해서도 원-핫 인코딩을 한다.
   
   ```python
   train_X = to_categorical(train_X)
   train_y = to_categorical(train_y)
   ```
   
   ```python
   print('train_X의 크기(shape) : {}'.format(train_X.shape)) # 원-핫 인코딩
   print('train_y의 크기(shape) : {}'.format(train_y.shape)) # 원-핫 인코딩
   ```
   
   ```python
   train_X의 크기(shape) : (2646, 60, 55)
   train_y의 크기(shape) : (2646, 60, 55)
   ```
   
   train_X와 train_y의 크기는 2,646 * 60 * 55이다.
   
   ![img](https://wikidocs.net/images/page/22886/rnn_image6between7.PNG)
   
   이는 샘플의 수(No. of samples)가 2,646개, 입력 시퀀스의 길이(input_length)가 60, 각 벡터의 차원(input_dim)이 55임을 의미한다.
   원-핫 벡터의 차원은 글자 집합의 크기인 55이어야 하므로 원-핫 인코딩이 수행되었음을 알 수 있다.
   
2. 모델 설계하기

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
   ```

   ```python
   model = Sequential()
   model.add(LSTM(256, input_shape=(None, train_X.shape[2]), return_sequences=True))
   model.add(LSTM(256, return_sequences=True))
   model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.fit(train_X, train_y, epochs=80, verbose=2)
   ```

   ```python
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.fit(train_X, train_y, epochs=80, verbose=2)
   ```

   ```python
   Epoch 1/80
    - 17s - loss: 3.0753 - acc: 0.1831
   ... 중략 ...
   Epoch 80/80
    - 18s - loss: 0.1855 - acc: 0.9535
   ```

   ```python
   def sentence_generation(model, length):
       ix = [np.random.randint(vocab_size)] # 글자에 대한 랜덤 인덱스 생성
       y_char = [index_to_char[ix[-1]]] # 랜덤 익덱스로부터 글자 생성
       print(ix[-1],'번 글자',y_char[-1],'로 예측을 시작!')
       X = np.zeros((1, length, vocab_size)) 
       # (1, length, 55) 크기의 X 생성. 즉, LSTM의 입력 시퀀스 생성
   
       for i in range(length):
           X[0][i][ix[-1]] = 1 
           # X[0][i][예측한 글자의 인덱스] = 1, 즉, 예측 글자를 다음 입력 시퀀스에 추가
           print(index_to_char[ix[-1]], end="")
           ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
           y_char.append(index_to_char[ix[-1]])
       return ('').join(y_char)
   ```

   ```python
   sentence_generation(model, 100)
   ```

   ```python
   49 번 글자 u 로 예측을 시작!
   ury-men would have done just as well. the twelve jurors were to say in that dide. he went on in a di'
   ```

---

### 6.2 글자 단위 RNN(Char RNN)으로 텍스트 생성하기

이번에는 다 대 일(many-to-one) 구조의 RNN을 글자 단위로 학습시키고, 텍스트 생성을 해보자.

1. 데이터에 대한 이해와 전처리

   ```python
   import numpy as np
   from tensorflow.utils import to_categorical
   ```

   다음과 같이 임의로 만든 노래 가사가 있다.

   ```python
   text='''
   I get on with life as a programmer,
   I like to contemplate beer.
   But when I start to daydream,
   My mind turns straight to wine.
   
   Do I love wine more than beer?
   
   I like to use words about beer.
   But when I stop my talking,
   My mind turns straight to wine.
   
   I hate bugs and errors.
   But I just think back to wine,
   And I'm happy once again.
   
   I like to hang out with programming and deep learning.
   But when left alone,
   My mind turns straight to wine.
   '''
   ```

   우선 위의 텍스트에 존재하는 단락 구분을 없애고 하나의 문자열로 재저장하자.

   ```python
   tokens = text.split() # '\n 제거'
   text = ' '.join(tokens)
   print(text)
   ```

   ```python
   I get on with life as a programmer, I like to contemplate beer. But when I start to daydream, My mind turns straight to wine. Do I love wine more than beer? I like to use words about beer. But when I stop my talking, My mind turns straight to wine. I hate bugs and errors. But I just think back to wine, And I'm happy once again. I like to hang out with programming and deep learning. But when left alone, My mind turns straight to wine.
   ```

   단락 구분이 없어지고 하나의 문자열로 재저장된 것을 확인할 수 있다.
   이제 이로부터 글자 집합을 만들어보자.
   기존에는 중복을 제거한 단어들의 모음인 단어 집합(vocabulary)을 만들었으나, 이번에 만들 집합은 단어 집합이 아니라 글자 집합이다.

   ```python
   char_vocab = sorted(list(set(text))) # 중복을 제거한 글자 집합 생성
   print(char_vocab)
   ```

   ```python
   [' ', "'", ',', '.', '?', 'A', 'B', 'D', 'I', 'M', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y']
   ```

   기존의 단어 단위의 집합이 아니라 알파벳 또는 구두점 등의 단위의 집합인 글자 집합이 생성되었다.

   ```python
   vocab_size=len(char_vocab)
   print ('글자 집합의 크기 : {}'.format(vocab_size))
   ```

   ```python
   글자 집합의 크기 : 33
   ```

   글자 집합의 크기는 33이다.

   ```python
   char_to_index = dict((c, i) for i, c in enumerate(char_vocab)) # 글자에 고유한 정수 인덱스 부여
   print(char_to_index)
   ```

   ```python
   {' ': 0, "'": 1, ',': 2, '.': 3, '?': 4, 'A': 5, 'B': 6, 'D': 7, 'I': 8, 'M': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'r': 26, 's': 27, 't': 28, 'u': 29, 'v': 30, 'w': 31, 'y': 32}
   ```

   이번 실습의 글자 집합의 경우 훈련 데이터에 등장한 알파벳의 대, 소문자를 구분하고 구두점과 공백을 포함하였다.
   이제 훈련에 사용할 문장 샘플들을 만들어보자.
   여기서는 RNN을 이용한 생성한 텍스트 챕터와 유샇하게 데이터를 구성한다.
   다만, 단위가 글자 단위라는 점이 다르다.
   예를 들어 훈련 데이터에 student라는 단어가 있고, 입력 시퀀스의 길이를 5라고 한다면 입력 시퀀스와 예측해야하는 글자는 다음과 같이 구성된다.

   ```python
   # Example) 5개의 입력 글자 시퀀스로부터 다음 글자 시퀀스를 예측. 즉, RNN의 time step은 5번
   stude -> n 
   tuden -> t
   ```

   여기서는 입력 시퀀스의 길이. 즉, 모든 샘플들의 길이가 10이 되도록 데이터를 구성해보자.
   예측 대상이 되는 글자도 필요하므로 우선 길이가 11이 되도록 데이터를 구성한다.

   ```python
   length = 11
   sequences = []
   for i in range(length, len(text)):
       seq = text[i-length:i] # 길이 11의 문자열을 지속적으로 만든다.
       sequences.append(seq)
   print('총 훈련 샘플의 수: %d' % len(sequences))
   ```

   ```python
   총 훈련 샘플의 수: 426
   ```

   총 샘플의 수는 426개로 완성되었다.
   이 중 10개만 출력해보자.

   ```python
   sequences[:10]
   ```

   ```python
   ['I get on wi',
    ' get on wit',
    'get on with',
    'et on with ',
    't on with l',
    ' on with li',
    'on with lif',
    'n with life',
    ' with life ',
    'with life a']
   ```

   첫번째 문장이었던 'I get on with life as a programmer,' 가 10개의 샘플로 분리된 것을 확인할 수 있다.
   다른 문장들에 대해서도 sequences에 모두 저장되어져 있다.
   원한다면, sequences[30:45] 등과 같이 인덱스 범위를 변경하여 출력해보자.
   이제 앞서 만든 char_to_index를 사용하여 전체 데이터에 대해서 정수 인코딩을 수행한다.

   ```python
   X = []
   for line in sequences: # 전체 데이터에서 문장 샘플을 1개씩 꺼낸다.
       temp_X = [char_to_index[char] for char in line] 
       # 문장 샘플에서 각 글자에 대해서 정수 인코딩을 수행.
       
       X.append(temp_X)
   ```

   정수 인코딩 된 결과가 X에 저장되었다.
   5개만 출력해보자.

   ```python
   for line in X[:5]:
       print(line)
   ```

   ```python
   [8, 0, 16, 14, 28, 0, 24, 23, 0, 31, 18]
   [0, 16, 14, 28, 0, 24, 23, 0, 31, 18, 28]
   [16, 14, 28, 0, 24, 23, 0, 31, 18, 28, 17]
   [14, 28, 0, 24, 23, 0, 31, 18, 28, 17, 0]
   [28, 0, 24, 23, 0, 31, 18, 28, 17, 0, 21]
   ```

   정상적으로 정수 인코딩이 수행되었다.
   이제 예측 대상인 글자를 분리시켜주는 작업을 한다.
   모든 샘플 문장에 대해서 맨 마지막 글자를 분리시켜준다.

   ```python
   sequences = np.array(X)
   X = sequences[:,:-1]
   y = sequences[:,-1] # 맨 마지막 위치의 글자를 분리
   ```

   정상적으로 분리가 되었는지 X와 y 둘 다 5개씩 출력해보자.

   ```python
   for line in X[:5]:
       print(line)
   ```

   ```python
   [ 8  0 16 14 28  0 24 23  0 31]
   [ 0 16 14 28  0 24 23  0 31 18]
   [16 14 28  0 24 23  0 31 18 28]
   [14 28  0 24 23  0 31 18 28 17]
   [28  0 24 23  0 31 18 28 17  0]
   ```

   ```python
   print(y[:5])
   ```

   ```python
   [18 28 17  0 21]
   ```

   앞서 출력한 5개의 샘플에서 각각 맨 뒤의 글자였던 18, 28, 17, 0, 21이 별도로 분리되어 y에 저장되었다.
   이제 X와 y에 대해서 원-핫 인코딩을 수행해보자.

   ```python
   sequences = [to_categorical(x, num_classes=vocab_size) for x in X] # X에 대한 원-핫 인코딩
   X = np.array(sequences)
   y = to_categorical(y, num_classes=vocab_size) # y에 대한 원-핫 인코딩
   ```

   원-핫 인코딩이 수행되었는지 확인하기 위해 수행한 후의 X의 크기(shape)를 보자.

   ```python
   print(X.shape)
   ```

   ```python
   (426, 10, 33)
   ```

   현재 X의 크기는 426 * 10 * 33이다.

   ![img](https://wikidocs.net/images/page/22886/rnn_image6between7.PNG)

   이는 샘플의 수(No. of samples)가 426개, 입력 시퀀스의 길이(input_length)가 10, 각 벡터의 차원(input_dim)이 33임을 의미한다.
   원-핫 벡터의 차원은 글자 집합의 크기인 33이어야 하므로 X에 대해서 원-핫 인코딩이 수행되었음을 알 수 있다.

2. 모델 설계하기

   ```python
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense, LSTM
   from tensorflow.keras.preprocessing.sequence import pad_sequences
   ```

   ```python
   model = Sequential()
   model.add(LSTM(80, input_shape=(X.shape[1], X.shape[2]))) # X.shape[1]은 25, X.shape[2]는 33
   model.add(Dense(vocab_size, activation='softmax'))
   ```

   LSTM을 사용하고, 은닉 상태의 크기는 80, 그리고 출력층에 단어 집합의 크기만큼의 뉴런을 배치하여 모델을 설계한다.

   ```python
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.fit(X, y, epochs=100, verbose=2)
   ```

   출력층의 활성화 함수로는 소프트맥스 함수, 손실 함수로는 크로스 엔트로피 함수를 사용하여 총 100번의 에포크를 수행한다.

   ```python
   Epoch 1/100
    - 1s - loss: 3.4793 - acc: 0.0900
   ... 중략 ...
   Epoch 100/100
    - 0s - loss: 0.2806 - acc: 0.9830
   ```

   문장을 생성하는 함수 sentence_generation을 만들어서 생성해본다.

   ```python
   def sentence_generation(model, char_to_index, seq_length, seed_text, n):
   # 모델, 인덱스 정보, 문장 길이, 초기 시퀀스, 반복 횟수
       init_text = seed_text # 문장 생성에 사용할 초기 시퀀스
       sentence = ''
   
       for _ in range(n): # n번 반복
           encoded = [char_to_index[char] for char in seed_text] # 현재 시퀀스에 대한 정수 인코딩
           encoded = pad_sequences([encoded], maxlen=seq_length, padding='pre') # 데이터에 대한 패딩
           encoded = to_categorical(encoded, num_classes=len(char_to_index))
           result = model.predict_classes(encoded, verbose=0)
           # 입력한 X(현재 시퀀스)에 대해서 y를 예측하고 y(예측한 글자)를 result에 저장.
           for char, index in char_to_index.items(): # 만약 예측한 글자와 인덱스와 동일한 글자가 있다면
               if index == result: # 해당 글자가 예측 글자이므로 break
                   break
           seed_text=seed_text + char # 현재 시퀀스 + 예측 글자를 현재 시퀀스로 변경
           sentence=sentence + char # 예측 글자를 문장에 저장
           # for문이므로 이 작업을 다시 반복
   
       sentence = init_text + sentence
       return sentence
   ```

   ```python
   print(sentence_generation(model, char_to_index, 10, 'I get on w', 80))
   ```

   ```python
   I get on with life as a programmer, I like to hang out with programming and deep learning.
   ```

   두 개의 문장이 출력되었는데 훈련 데이터에서는 연속적으로 나온 적이 없는 두 문장임에도 모델이 임의로 잘 생성해낸 것 같다.

