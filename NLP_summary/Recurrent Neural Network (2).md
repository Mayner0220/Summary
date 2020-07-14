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

   

