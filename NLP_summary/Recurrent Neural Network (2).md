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

   

