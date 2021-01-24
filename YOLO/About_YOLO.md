# About YOLO

## What is the "YOLO"? :thinking:

- YOLO는 'You Only Look Once'의 약어로 Joseph Redmon이 워싱턴 대학교에서 여러 친구들과 함께 2015년에 YOLOv1을 처음 논문과 함께 발표했습니다. 당시에는 Object Detection 분야에서는 대부분 Faster R-CNN(Region with Convolutional Neural Network)가 가장 좋은 성능을 내고 있었습니다.
- YOLO는 처음으로 One-shot-detection 방법을 고안하였습니다. Two-shot-detection으로 Object Detection을 구성하였는데 실시간성이 굉장히 부족하여, 5~7 FPS의 실시간성을 보여줬었습니다.

## The FAST Boy! YOLO! :runner:

### Theory of YOLO :books:

- 기존의 Object Detection은 Classification 문제를 2단계로 나눠 검출(Two-Shot-Detection)하여 정확도가 높았지만, 네트워크를 여러번 호출하기에 속도는 매우 느렸습니다.
- 하지만 YOLO는 One-Stage 검출기를 이용하여 조금은 정확도가 떨어지지만 엄청나게 빠른 검출기를 만들어 냈습니다. 즉, 하나의 이미지를 딱 한 번만 신경망을 통과시켜 빠른 속도의 검출기를 만들어낸 것입니다.

### Description of YOLO :speech_balloon:

