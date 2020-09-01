# Front-End & Back-End :thinking:

참고: [https://kamang-it.tistory.com/entry/Web-Applicateion%EC%84%9C%EB%B2%84-%EC%82%AC%EC%9D%B4%EB%93%9C%EC%99%80-%ED%81%B4%EB%9D%BC%EC%9D%B4%EC%96%B8%ED%8A%B8-%EC%82%AC%EC%9D%B4%EB%93%9C%EB%B0%B1%EC%95%A4%EB%93%9C%EC%99%80-%ED%94%84%EB%A1%A0%ED%8A%B8%EC%95%A4%EB%93%9C](https://kamang-it.tistory.com/entry/Web-Applicateion서버-사이드와-클라이언트-사이드백앤드와-프론트앤드)

---

### Front-End?

HTML, CSS, JS 등

### Back-End?

PHP, JSP, ASP, ASP.NET, Node.JS, Flask 등

---

### Front-End(클라이언트 사이드)

개발자의 입장으로서, 웹브라우저는 "인터프리터"이다.

![img](https://t1.daumcdn.net/cfile/tistory/2579F63A59736D2C13)

인터프리터는 코드를 읽어서 실행시켜주는 프로그램이다.
웹브라우저는 인터프리터로 HTML + CSS + js로 이루어진 코드를 읽어내어 화면에 보여주게 된다.

![img](https://t1.daumcdn.net/cfile/tistory/2471EF3C59736E3B2C)

좀 더 풀어서 말해본다면, 프론트엔드라고 불리는 클라이언트 사이드가 실행되는 곳이 웹브라우저이다.

즉, **내 컴퓨터**에 설치 되어 있는 웹브라우저에서 코드들이 실행되게 되는 것이다.
다시 말해 웹브라우저가 설치되어 있는 내 컴퓨터에서 코드들이 돌아간다는 것이다.

사용자의 컴퓨터에서 사용되기에 클라이언트 사이드(사용자 측)이라고 불리며, Front-End라고 불리는 이유 또한 **사용자에게 보여지는 말단**이기 때문이다.

웹 어플리케이션의 경우에는 서버에서 사용자에게 HTML + CSS + js로 만들어진 하나의 페이지를 사용자에게 전송해준다.
사용자의 컴퓨터에 깔려 있는 웹브라우저라는 인터프리터를 돌려서 페이지를 화면에 보여주는 것이다.

여기서 알 수 있는 내용은 "서버는 단지 사용자에게 HTML + CSS + js로 이루어진 페이지만 제공하면 되는 것이다."이다.

---

### Back-End(서버 사이드)

Front-End 개발이 사용자의 컴퓨터에서 실행될 HTML + CSS + js를 만드는 작업이라면, 그 반대인 Back-End는 어떨까.

단순히 말하자면, 서버 사이드의 역할은 사용자에게 제공될 데이터를 만들어주는 역할을 하는 것이다.

![img](https://t1.daumcdn.net/cfile/tistory/251F6B4C59736B9C25)

위 사진에 나와있는 기술 내용들은 HTML + CSS + js로 만든 파일을 생성할 수 있다.
그 파일을 만들고 사용자에게 제공하는 것이다.

이렇게 말하면 좀 헷갈릴 수 있다.
Front-End와 Back-End는 무엇이 다른건가.

Front-End는 사용자에게 보여줄 결과물이다.

좀 더 자세하게 이야기해보자.

만약 동적인 페이지를 만들 필요가 없다면 HTML + CSS + js를 그냥 launching하면 된다.
이 경우, Front-End 프로그래밍만으로도 페이지를 만들 수 있다.

그러나 경우에 따라서 코드가 달라질 수 있다.
예를 들어, HTML안에 a태그가 4개가 있다면 확장자가 HTML인 폴더의 경우 어떻게 해도 a태그는 4개일 이다.

![img](https://t1.daumcdn.net/cfile/tistory/223AD93F597374FD15)

이를 도식화하면 이런 모습이다.

 즉, HTML은 서버에서 바로 클라이언트 측으로 데이터를 제공한다.

그럼 다시 본론으로 돌아가, Back-End을 봐보자.

![img](https://t1.daumcdn.net/cfile/tistory/212ED24F597375F00D)

Back-End의 예시로 jsp를 든다면, jsp에서 페이지를 만든 도면은 Front-End에 보여줄 화면과 일치하지 않는다.
아직 서버에 코드 상태로 있을 때는 어떤 화면이 나올지 모른다.
그러나 사용자의 입력이 있다면 그 입력에 맟춰서 페이지를 변경시켜, 다시 Front-End에 데이터를 제공한다.

Front-End를 개발할 때는, 보여줄 화면을 어떻게 보여질지 만들어내는 것이다.
따라서 HTML과 CSS로 웹브라우저의 요소와 시각 정보를 제공하고, 기능적인 요소를 js로 만들면 되는 것이다.

Back-End는 Front-End에서 개발된 코드를 사용자에게 제공하고, 이 과정 중에 Back-End는 서버를 사용할 수 있으며 서버는 항상 켜져있으므로 DB와 연동하여 데이터를 다룰 수 있다.

---

### 오늘날의 Back-End

오늘날의 Back-End을 단순히 말하면, 단지 **데이터를 어떻게 가공하여 이를 어떻게 효율적으로 보여줄 수 있는가**가 주 목적이 되었다.

과거의 Front-End와 Back-End을 말하면서, 아직도 헷갈려 하는 이들도 존재하지만 그렇다고 모든 내용들이 틀렸다는 것은 아니다.

정리하자면, Back-End는 UI나 GUI로 구성된 화면의 통신이나 요청에 대하여 DB나 인터페이스 등을 통해 시스템 구성의 실체에 접근하는 것이라고 할 수 있다.
굳이 풀어서 말해보자면, 웹을 운영하는데에 필요한 데이터를 구성, 관리 등을 하는 것이 Back-End이다.