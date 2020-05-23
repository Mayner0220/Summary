# Text Processing (4)

Source: https://wikidocs.net/21694, https://wikidocs.net/21698, https://wikidocs.net/21693, https://wikidocs.net/21707, https://wikidocs.net/22530, https://wikidocs.net/21703, https://wikidocs.net/31766, https://wikidocs.net/22647, https://wikidocs.net/22592, https://wikidocs.net/33274
NLP에 있어서 Text Processing은 매우 중요한 작업이다.

---

### 6. 정수 인코딩(Integer Encoding)

컴퓨터는 텍스트보다는 숫자를 더 잘 처리할 수 있다.
이를 위해 NLP에서는 텍스트를 숫자로 바꾸는 여러가지 기법들이 존재한다.
그리고 그러한 기법들을 본격적으로 적용시키기 위한 첫 단계로 각 단어를 고유한 정수에 맵핑(mapping)시키는 전처리 작업이 필요할 때가 있다.

예를 들어 갖고 있는 텍스트에 단어가 5000개가 있다면, 5000개의 단어들 각각에 1번부터 5000번까지 단어와 맵핑되는 고유한 정수, 다른 표현으로는 인덱스를 부여한다.
가령 book은 150번, dog는 17번, love는 192번, books 212번과 같이 숫자가 부여된다.
인덱스를 부여하는 방법은 여러가지가 있을 수 있는데, 랜덤으로 부여하기도 하지만, 보통은 전처리 또는 빈도수가 높은 단어들만 사용하기 위해 단어에 대한 빈도수를 기준으로 정렬한 뒤에 부여한다.

---

### 6.1 정수 인코딩

어떤 과정으로 단어에 정수 인덱스를 부여하는지에 대해서 정리해보자.

단어에 정수를 부여하는 방법 중 하나로, 단어를 빈도수 순으로 정렬한 단어 집합(vocabulary)을 만들고, 빈도수가 높은 순서대로 차례로 낮은 숫자부터 정수를 부여하는 방법이 있다.
이해를 위해 빈도수가 적당히 분포되도록 의도적으로 만든 텍스트 데이터를 가지고 실습해보자.

1. dictionary 사용하기

   ```python
   from nltk.tokenize import sent_tokenize
   from nltk.tokenize import word_tokenize
   from nltk.corpus import stopwords
   
   text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
   ```

   우선 여러 문장이 함께 있는 텍스트 데이터로부터 문장 토큰화를 수행해보자.

   ```python
   # 문장 토큰화
   text = sent_tokenize(text)
   print(text)
   ```

   ```python
   ['A barber is a person.', 'a barber is good person.', 'a barber is huge person.', 'he Knew A Secret!', 'The Secret He Kept is huge secret.', 'Huge secret.', 'His barber kept his word.', 'a barber kept his word.', 'His barber kept his secret.', 'But keeping and keeping such a huge secret to himself was driving the barber crazy.', 'the barber went up a huge mountain.']
   ```

   기존의 텍스트 데이터가 문장 단위로 토큰화 된 것을 확인 할 수 있다.
   이제 정제 작업을 병행하며, 단어 토큰화를 수행한다.

   ```python
   # 정제와 단어 토큰화
   vocab = {} # 파이썬의 dictionary 자료형
   sentences = []
   stop_words = set(stopwords.words('english'))
   
   for i in text:
       sentence = word_tokenize(i) # 단어 토큰화를 수행합니다.
       result = []
   
       for word in sentence: 
           word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄입니다.
           if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거합니다.
               if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거합니다.
                   result.append(word)
                   if word not in vocab:
                       vocab[word] = 0 
                   vocab[word] += 1
       sentences.append(result) 
   print(sentences)
   ```

   ```python
   [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
   ```

   텍스트를 숫자로 바꾸는 단계라는 것은 본격적으로 NLP 작업에 들어간다는 의미이므로, 단어가 텍스트일 때만 할 수 있는 최대한의 전처리를 끝내놓아야 한다.
   위의 코드를 보면, 동일한 단어가 대문자로 표기되었다는 이유로 서로 다른 단어로 카운트되는 일이 없도록 모든 단어를 소문자로 바꾸었다.
   그리고 NLP에서 크게 의미를 갖지 못하는 불용어와 길이가 짧은 단어를 제거하는 방법을 사용했다.

   현재 `vocab`에는 중복을 제거한 단어와 각 단어에 대한 빈도수가 기록되어져 있다.
   `vocab`을 출력해보자.

   ```python
   print(vocab)
   ```

   ```python
   {'barber': 8, 'person': 3, 'good': 1, 'huge': 5, 'knew': 1, 'secret': 6, 'kept': 4, 'word': 2, 'keeping': 2, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1}
   ```

   단어를 키(key)로, 단어에 대한 빈도수가 값(value)으로 저장되어져 있다.
   `vocab`에 단어를 입력하면 빈도수를 리턴한다.

   ```python
   print(vocab["barber"]) # 'barber'라는 단어의 빈도수 출력
   ```

   ```python
   8
   ```

   이제 높은 빈도수를 가진 단어일수록 낮은 인덱스를 부여한다.

   ```python
   word_to_index = {}
   i=0
   for (word, frequency) in vocab_sorted :
       if frequency > 1 : # 정제(Cleaning) 챕터에서 언급했듯이 빈도수가 적은 단어는 제외한다.
           i=i+1
           word_to_index[word] = i
   print(word_to_index)
   ```

   ```python
   {'barber': 1, 'person': 2, 'huge': 3, 'secret': 4, 'kept': 5, 'word': 6, 'keeping': 7}
   ```

   1의 인덱스를 가진 단어가 가장 빈도수가 높은 단어가 된다.
   그리고 이러한 작업을 수행하는 동시에 각 단어의 빈도수를 알 경우에만 할 수 있는 전처리인 빈도수가 적은 단어를 제외시키는 작업을 한다.
   등장 빈도가 낮은 단어는 NLP에서 의미를 가지지 않을 가능성이 높기 때문이다.
   여기서 빈도수가 1인 단어들을 전부 제외시켰다.

   NLP를 하다보면, 텍스트 데이터에 있는 단어를 모두 사용하기 보다는 빈도수가 가장 높은 n개의 단어만 사용하고 싶은 경우가 많다.
   위 단어들은 빈도수가 높은 순으로 낮은 정수가 부여되어져 있으므로 빈도수 상위 n개의 단어만 사용하고 싶다면 vocab에서 정수값이 1부터 n까지인 단어들만 사용하면 사용된다.
   여기서 상위 5개 단어만 사용한다고 가정해보자.

   ```python
   vocab_size = 5
   words_frequency = [w for w,c in word_to_index.items() if c >= vocab_size + 1] # 인덱스가 5 초과인 단어 제거
   for w in words_frequency:
       del word_to_index[w] # 해당 단어에 대한 인덱스 정보를 삭제
   ```

   ```python
   {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
   ```

   이제 `word_to_index`에는 빈도수가 높은 상위 5개의 단어만 저장되었다.
   이제 `word_to_index`를 사용하여 단어 토큰화가 된 상태로 저장된 sentences에 있는 각 단어를 정수로 바꾸는 작업을 해보자.

   예를 들어 `sentences`에서 첫번째 문장은 `['barber', 'person']`이었는데, 이 문장에 대해서는 `[1, 5]`로 인코딩한다.
   그런데 두번째 문장인 `['barber', 'good', 'person']`에는 더 이상 `word_to_index`에 존재하지 않은 단어인 'good'이라는 단어가 있다.

   이처럼 훈련 데이터 또는 테스트 데이터에 대해서 단어 집합에 존재하지 않는 단어들을 Out-Of-Vocabulary(단어 집합에 없는 단어)의 약자로 OOV라고 한다.
   `word_to_index`에 OOV란 단어를 새롭게 추가하고, 단어 집합에 없는 단어들은 OOV의 인덱스로 인코딩해보자.

   ```python
   word_to_index['OOV'] = len(word_to_index) + 1
   ```

   이제 `word_to_index`를 사용하여 `sentences`의 모든 단어들을 맵핑되는 정수로 인코딩해보자.

   ```python
   encoded = []
   for s in sentences:
       temp = []
       for w in s:
           try:
               temp.append(word_to_index[w])
           except KeyError:
               temp.append(word_to_index['OOV'])
       encoded.append(temp)
   print(encoded)
   ```

   ```python
   [[1, 5], [1, 6, 5], [1, 3, 5], [6, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [6, 6, 3, 2, 6, 1, 6], [1, 6, 3, 6]]
   ```

   지금까지 파이썬의 dictionary 자료형으로 정수 인코딩을 진행해보았다.
   그런데 이보다는 좀 더 쉽게 하기 위해서 `Counter`, `FreqDist`, `enumerate` 또는 케라스 토크나이저를 사용하는 것을 권장한다.

2. Counter 사용하기

   ```python
   from collections import Counter
   ```

   ```python
   print(sentences)
   ```

   ```python
   [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
   ```

   현재 `sentences`는 단어 토큰화가 된 결과가 저장되어져 있다.
   단어 집합(vocabulary)을 만들기 위해서 `sentences`에서 문장의 경계인 [, ]를 제거하고 하나의 리스트로 만들어보자.

   ```python
   words = sum(sentences, [])
   # 위 작업은 words = np.hstack(sentences)로도 수행 가능.
   print(words)
   ```

   ```python
   ['barber', 'person', 'barber', 'good', 'person', 'barber', 'huge', 'person', 'knew', 'secret', 'secret', 'kept', 'huge', 'secret', 'huge', 'secret', 'barber', 'kept', 'word', 'barber', 'kept', 'word', 'barber', 'kept', 'secret', 'keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy', 'barber', 'went', 'huge', 'mountain']
   ```

   이를 파이썬의 `Counter()`의 입력으로 사용하면 중복을 제거하고 단어의 빈도수를 기록한다.

   ```python
   vocab = Counter(words) # 파이썬의 Counter 모듈을 이용하면 단어의 모든 빈도를 쉽게 계산할 수 있습니다.
   print(vocab)
   ```

   ```python
   Counter({'barber': 8, 'secret': 6, 'huge': 5, 'kept': 4, 'person': 3, 'word': 2, 'keeping': 2, 'good': 1, 'knew': 1, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1})
   ```

   단어를 키(key)로, 단어에 대한 빈도수가 값(value)으로 저장되어져 있다.
   `vocab`에 단어를 입력하면 빈도수를 리턴한다.

   ```python
   print(vocab["barber"]) # 'barber'라는 단어의 빈도수 출력
   ```

   ```python
   8
   ```

   barber란 단어가 총 8번 등장했다.
   `most_common()`은 상위 빈도수를 가진 주어진 수의 단어만을 리턴한다.
   이를 사용하여 등장 빈도수가 높은 단어들을 원하는 개수만큼만 얻을 수 있다.
   등장 빈도수 상위 5개의 단어만 집합으로 저장해보자.

   ```python
   vocab_size = 5
   vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
   vocab
   ```

   ```python
   {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
   ```

3. NLTK의 FreqDist 사용하기
   NLTK에서는 빈도수 계산 도구인 `FreqDist()` 지원한다.
   위에서 사용한 `Counter()`랑 같은 방법으로 사용할 수 있다.

   ```python
   from nltk import FreqDist
   import numpy as np
   ```

   ```python
   # np.hstack으로 문장 구분을 제거하여 입력으로 사용 . ex) ['barber', 'person', 'barber', 'good' ... 중략 ...
   vocab = FreqDist(np.hstack(sentences))
   ```

   단어를 key로, 단어에 대한 빈도수가 value으로 저장되어져 있다.
   `vocab`에 단어를 입력하면 빈도수를 리턴한다.

   ```python
   print(vocab["barber"]) # 'barber'라는 단어의 빈도수 출력
   ```

   ```python
   8
   ```

   barber란 단어가 총 8번 등장하였다.
   `most_common()`은 상위 빈도수를 가진 주어진 수의 단어만을 리턴한다.
   이를 사용하여 등장 빈도수가 높은 단어들을 원하는 개수만큼만 얻을 수 있다.
   등장 빈도수 상위 5개의 단어만 집합으로 저장해보자.

   ```python
   vocab_size = 5
   vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
   vocab
   ```

   ```python
   [('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]
   ```

   앞서 `Counter()`를 사용했을 때와 결과가 같다.
   이전 실습들과 마찬가지로 높은 빈도수를 가진 단어일수록 낮은 인덱스를 부여한다.
   그런데 이번에는 `enumerate()`를 사용하여 좀 더 짧은 코드로 인덱스를 부여해보자.

   ```python
   word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
   print(word_to_index)
   ```

   ```python
   {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
   ```

   위와 같이 인덱스를 부여할 때는 enumerate()를 사용하는 것이 편리하다.

4. enumerate 이해하기
   `enumerate()`는 순서가 있는 자료형(list, set, tuple, dictionary, string)을 입력으로 받아 인덱스를 순차적으로 함께 리턴한다는 특징이 있다.
   간단한 예제를 통해서 `enumerate()`를 이해해보자.

   ```python
   test=['a', 'b', 'c', 'd', 'e']
   for index, value in enumerate(test): # 입력의 순서대로 0부터 인덱스를 부여함.
     print("value : {}, index: {}".format(value, index))
   ```

   ```python
   value : a, index: 0
   value : b, index: 1
   value : c, index: 2
   value : d, index: 3
   value : e, index: 4
   ```

   위의 출력 결과는 모든 토큰에 대해서 인덱스가 순차적으로 증가되며 부여된 것을 보여준다.

---

### 6.2  케라스의 텍스트 전처리

케라스는 기본적인 전처리를 위한 도구들을 제공한다.
때로는 정수 인코딩을 위해서 케라스의 전처리 도구인 토크나이저를 사용하기도 하는데, 사용 방법과 그 특징에 대해서 이해해보자.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
```

```python
sentences=[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
```

단어 토큰화까지 수행된 앞서 사용한 텍스트 데이터와 동일한 데이터를 사용한다.

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성한다.
```

`fit_on_texts`는 입력한 텍스트로부터 단어 빈도수가 높은 순으로 낮은 정수 인덱스를 부여하는데, 정확히 앞서 설명한 정수 인코딩 작업이 이루어진다고 보면된다.
각 단어에 인덱스가 어떻게 부여되었는지를 보려면, `word_index`를 사용한다.

```python
print(tokenizer.word_index)
```

```python
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}
```

각 단어의 빈도수가 높은 순서대로 인덱스가 부여된 것을 확인할 수 있다.
각 단어가 카운트를 수행하였을 때 몇 개였는지를 보고자 한다면 `word_counts`을 사용한다.

```python
print(tokenizer.word_counts)
```

```python
OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])
```

`texts_to_sequences()`는 입력으로 들어온 코퍼스에 대해서 각 단어를 이미 정해진 인덱스로 변환한다.

```python
print(tokenizer.texts_to_sequences(sentences))
```

```python
[[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
```

앞서 빈도수가 가장 높은 단어 n개만을 사용하기 위해서 `most_common()`을 사용했었다.
케라스 토크나이저에서는 `tokenizer=Tokenizer(num_words=숫자)`와 같은 방법으로 빈도수가 높은 강위 몇 개의 단어만 사용하겠다고 지정할 수 있다.
여기서는 1번 단어부터 5번 단어까지만 사용하자.
상위 5개 단어를 사용한다고 토크나이저를 재정의 해보자.

```python
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1) # 상위 5개 단어만 사용
tokenizer.fit_on_texts(sentences)
```

`num_words`에서 +1을 더해서 값을 넣어주는 이유는 `num_words`는 숫자를 0부터 카운트한다.
만약 5를 넣으면 0 ~ 4번 단어 보존을 의미하게되므로 뒤의 실습에서 1번 단어부터 4번 단어만 남게된다.
그렇기 때문에 1 ~ 5번 단어까지 사용하고 싶다면 `num_words`에 숫자 5를 넣어주는 것이 아니라 5+1인 값을 넣어줘야 한다.

실질적으로 숫자 0에 지정된 단어가 존재하지 않는데도 케라스 토크나이저가 숫자 0까지 단어 집합의 크기로 산정하는 이유는 NLP에서 패딩(padding)이라는 작업 때문이다.
케라스 토크나이저를 사용할 때는 숫자 0도 단어의 집합 크기로 고려해야 한다.

`word_index`를 확인해보자.

```python
print(tokenizer.word_index)
```

```python
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}
```

상위 5개의 단어만 사용하겠다고 선언하였는데 여전히 13개의 단어가 모두 출력된다.
`word_counts`를 확인해보자.

```python
print(tokenizer.word_counts)
```

```python
OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])
```

word_counts에서도 마찬가지로 13개의 단어가 모두 출력된다.
사실 실제 적용은 texts_to_sequences를 사용할 때 적용이 된다.

```python
print(tokenizer.texts_to_sequences(sentences))
```

```python
[[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]
```

코퍼스에 대해서 각 단어를 이미 정해진 인덱스로 변환하는데, 상위 5개의 단어만을 사용하겠다고 지정하였으므로 1번 단어부터 5번 단어까지만 보존되고 나머지 제거된 것을 볼 수 있다.
만약, `word_index`와 `word_counts`에서도 지정된 `num_words`만큼의 단어만 남기고 싶다면 아래의 코드도 방법이다.

```python
tokenizer = Tokenizer() # num_words를 여기서는 지정하지 않은 상태
tokenizer.fit_on_texts(sentences)
```

```python
vocab_size = 5
words_frequency = [w for w,c in tokenizer.word_index.items() if c >= vocab_size + 1] # 인덱스가 5 초과인 단어 제거
for w in words_frequency:
    del tokenizer.word_index[w] # 해당 단어에 대한 인덱스 정보를 삭제
    del tokenizer.word_counts[w] # 해당 단어에 대한 카운트 정보를 삭제
print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(sentences))
```

```python
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
OrderedDict([('barber', 8), ('person', 3), ('huge', 5), ('secret', 6), ('kept', 4)])
[[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]
```

케라스 토크나이저는 기본적으로 단어 집합에 없는 단어인 OOV에 대해서는 단어를 정수로 바꾸는 과정에서 아예 단어를 제거한다는 특징이 있다.
단어 집합에 없는 단어들을 OOV로 간주하여 보존하고 싶다면 `Tokenizer`의 인자 `oov_token`을 사용한다.

```python
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token = 'OOV')
# 빈도수 상위 5개 단어만 사용. 숫자 0과 OOV를 고려해서 단어 집합의 크기는 +2
tokenizer.fit_on_texts(sentences)
```

만약 oov_token을 사용하기로 했다면 케라스 토크나이저는 기본적으로 OOV의 인덱스를 1로 한다.

```
print('단어 OOV의 인덱스 : {}'.format(tokenizer.word_index['OOV']))
```

```
단어 OOV의 인덱스 : 1
```

이제 코퍼스에 대해서 정수 인코딩을 진행해보자.

```python
print(tokenizer.texts_to_sequences(sentences))
```

```python
[[2, 6], [2, 1, 6], [2, 4, 6], [1, 3], [3, 5, 4, 3], [4, 3], [2, 5, 1], [2, 5, 1], [2, 5, 3], [1, 1, 4, 3, 1, 2, 1], [2, 1, 4, 1]]
```

빈도수 상위 5개의 단어는 2 ~ 6까지의 인덱스를 가졌으며, 그 외 단어 집합에 없는 'good'과 같은 단어들은 전부 'OOV'의 인덱스인 1로 인코딩되었다.

---

### 7. One-hot encoding

컴퓨터 또는 기계는 문자보다는 숫자를 더 잘 처리할 수 있다.
이를 위해 NLP에서는 문자를 수사로 바꾸는 기법 여러가지가 있다.
one-hot encoding은 그 많은 기법 중에서 단어를 표현하는 가장 기본적인 표현 방법이며, ML/DL을 하기 위해서는 반드시 배워야 하는 표현 방법이다.

원-핫 인코딩에 대해서 배우기에 앞서 단어 집합에 대해서 정의해보도록 하자.
단어 집합은 앞으로 NLP에서 나오는 개념이기 때문에 여기서 이해하고 가야한다.
단어 집합은 서로 다른 단어들의 집합이다.
여기서 혼동이 없도록 서로 다른 단어라는 정의에 대해서 좀 더 주목할 필요가 있다.

단어 집합에서는 기본적으로 book과 books와 같이 단어의 변형 형태도 다른 단어로 간주한다.
단어 집합에 있는 단어들을 가지고, 문자를 숫자(더 구체적으로 벡터)로 바꾸는 원-핫 인코딩을 포함한 여러 방법에 대해서 배우게 된다.

원-핫 인코딩을 위해서 먼저 해야할 일은 단어 집합을 만드는 일이다.
텍스트의 모든 단어를 중복을 허용하지 않고 모아놓으면 이를 단어 집합이라고 한다.
그리고 이 단어 집합에 고유한 숫자를 부여하는 정수 인코딩을 진행한다.
텍스트에 단어가 총 5,000개가 존재한다면, 단어 집합의 크기는 5,000이다.
5,000개의 단어가 있는 이 단어 집합의 단어들마다 1번부터 5,000번까지 인덱스를 부여한다고 해보자.
가령, book은 150번, dog는 171번, love는 192번, books는 212번과 같이 부여할 수 있다.

---

### 7.1 One-hot encoding이란?

원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식이다.
이렇게 표현된 벡터를 원-핫 벡터(One-hot vector)라고 한다.

원-핫 인코딩을 두 가지 과정으로 정리해보자.

1. 각 단어에 고유한 인덱스를 부여한다. (정수 인코딩)
2. 표현하고 싶은 단어의 인덱스의 위치에 1을 부여하고, 다른 단어의 인덱스의 위치에는 0을 부여한다.

이해를 돕기 위해 한국어 문장을 예제로 원-핫 벡터를 만들어보자.

```python
from konlpy.tag import Okt  
okt=Okt()  
token=okt.morphs("나는 자연어 처리를 배운다")  

print(token)
```

```python
['나', '는', '자연어', '처리', '를', '배운다']
```

코엔엘파이의 Okt 형태소 분석기를 통해서 우선 문장에 대해서 토큰화를 수행했다.

```python
word2index={}
for voca in token:
     if voca not in word2index.keys():
       word2index[voca]=len(word2index)
       
print(word2index)
```

```python
{'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}  
```

각 토큰에 대해서 고유한 인덱스를 부여했다.
지금의 문장이 짧기 때문에 각 단어의 빈도수를 고려하지 않지만, 빈도수 순대로 단어를 정렬하여 고유한 인덱스를 부여하는 작업이 사용되기도 한다.

```python
def one_hot_encoding(word, word2index):
       one_hot_vector = [0]*(len(word2index))
       index=word2index[word]
       one_hot_vector[index]=1
       return one_hot_vector
```

토큰을 입력하면 해당 토큰에 대한 원-핫 벡터를 만들어내는 함수를 만들어낸다.

```python
one_hot_encoding("자연어",word2index)
```

```python
[0, 0, 1, 0, 0, 0]  
```

해당 함수에 '자연어'라는 토큰을 입력으로 넣어봤더니 [0, 0, 1, 0, 0, 0]라는 벡터가 나왔다.
자연어는 단어 지밥에서 인덱스가 2이므로, 자연어를 표현하는 원-핫 벡터는 인덱스 2의 값이 1이며, 나머지 값은 0인 벡터가 나온다.

---

### 7.2 케라스를 이용한 원-핫 인코딩

위에서는 원-핫 인코딩을 이해하기 위해 파이썬으로 직접 코드를 작성했지만, 케라스는 원-핫 인코딩을 수행하는 유용한 도구 `to_categorical()`를 지원한다.
이번에는 케라스만으로 정수 인코딩과 원-핫 인코딩을 순차적으로 진행해보도록 하자.

```python
text="나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"
```

위와 같은 문장이 있다고 했을 때, 정수 인코딩 챕터에서와 같이 케라스 토크나이저를 이용한 정수 인코딩은 다음과 같다.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text="나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

t = Tokenizer()
t.fit_on_texts([text])

print(t.word_index) # 각 단어에 대한 인코딩 결과 출력.
```

```python
{'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}
```

위와 같이 생성된 단어 집합에 있는 단어들만 구성된 텍스트가 있다면, `texts_to_sequences()`를 통해서 이를 정수 시퀀스로 변환 가능하다.
생성된 단어 집합 내의 일부 단어들로만 구성된 서브 텍스트인 `sub_text`를 만들어 확인해보자.

```python
sub_text="점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded=t.texts_to_sequences([sub_text])[0]

print(encoded)
```

```python
[2, 5, 1, 6, 3, 7]
```

지금까지 진행한 것은 이미 정수 인코딩 챕터에서 배운 내용이다.
이제 해당 결과를 가지고, 원-핫 인코딩을 진행해보자.
케라스는 정수 인코딩 된 결과로부터 원-핫 인코딩을 수행하는 `to_categorical()`를 지원한다.

```python
one_hot = to_categorical(encoded)

print(one_hot)
```

```python
[[0. 0. 1. 0. 0. 0. 0. 0.] #인덱스 2의 원-핫 벡터
 [0. 0. 0. 0. 0. 1. 0. 0.] #인덱스 5의 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0.] #인덱스 1의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 1. 0.] #인덱스 6의 원-핫 벡터
 [0. 0. 0. 1. 0. 0. 0. 0.] #인덱스 3의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 1.]] #인덱스 7의 원-핫 벡터
```

위의 결과는 "점심 먹으러 갈래 메뉴는 햄버거 최고야"라는 문장이 [2, 5, 1, 6, 3, 7]로 정수 인코딩이 되고나서, 각각의 인코딩 된 결과를 인덱스로 원-핫 인코딩이 수행된 모습을 보여준다.

---

### 7.3 원-핫 인코딩의 한계

이러한 표현 방식은 단어의 개수가 늘어날 수록, 벡터를 저장하기 위해 필요한 공간이 계속 늘어난다는 단점이 있다.
다른 말로는 벡터의 차원이 계속 늘어난다고도 표현한다.
원-핫 벡터는 단어  집합의 크기가 곧 벡터의 차원 수가 됩니다.
가령, 단어가 1,000개인 코퍼스를 가지고 원-핫 벡터를 만들면, 모든 단어 각각은 모두 1,000개의 차원을 가진 벡터가 된다.
다시 말해 모든 단어 각각은 하나의 값만 1을 가지고, 999개의 값은 0의 값을 가지는 벡터가 되는데 이는 저장 공간 측면에서 매우 비효율적인 표현 방법이다.

또한 원-핫 벡터는 단어의 유사도를 표현하지 못한다는 단점이 있다.
예를 들어서 늑대, 호랑이, 강아지, 고양이라는 4개의 단어에 대해서 원-핫 인코딩을 해서 각각, [1, 0 , 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]이라는 원-핫 벡터를 부여받았다고 하자.
이 때 원-핫 벡터로는 강아지와 늑대가 유사하고, 호랑이와 고양이가 유사하다는 것을 표현할 수가 없다.
좀 더 극단적으로는 강아지, 개, 냉장고라는 단어가 있을 때 강아지라는 단어가 개와 냉장고라는 단어 중 어떤 단어와 더 유사한지도 알 수 없다.

단어 간 유사성을 알 수 없다는 단점은 검색 시스템 등에서 심각한 문제이다.
가령, 여행을 가려고 웹 검색창에 '삿포로 숙소'라는 단어를 검색한다고 하자.
제대로 된 검색 시스템이라면, '삿포로 숙소'라는 검색어에 대해서 '삿포로 게스트 하우스', '삿포로 료칸', '삿포로 호텔'과 같은 유사 단어에 대한 결과도 함께 보여줄 수 있어야 한다.
하지만 단어간 유사성을 계산할 수 없다면, '게스트 하우스'와 '료칸'과 '호텔'이라는 연관 검색어를 보여줄 수 없다.

이러한 단점을 해결하기 위해 단어의 잠재 의미를 반영하여 다차원 공간에 벡터화 하는 기법으로 크게 두 가지가 있다.
첫째는 카운트 기반의 벡터화 방법인 LSA, HAL 등이 있으며, 둘째는 예측 기반으로 벡터화하는 NNLM, RNNLM, Word2Vec, FastText 등이 있다.
그리고 카운트 기반과 예측 기반 두 가지 방법을 모두 사용하는 방법으로 GloVe라는 방법이 존재한다.

