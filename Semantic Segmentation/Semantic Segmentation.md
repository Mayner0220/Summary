# Semantic Segmentation :scissors:

Reference: [1편: Semantic Segmentation 첫걸음!](https://medium.com/hyunjulie/1%ED%8E%B8-semantic-segmentation-%EC%B2%AB%EA%B1%B8%EC%9D%8C-4180367ec9cb)

## :thinking: What is the "semantic segmentation"?

- Semantic Segmentation은 Computer Vision(CV) 분야에서 가장 핵심적인 분야 중 하나입니다.
  단순히 이미지를 분류하는 것에 그치지 않고, 그 장면을 완벽하게 이해해야 높은 수준의 task입니다.

### :goal_net: Purpose of semantic segmentation

- Semantic Image Segmentation의 목적은 사진에 있는 모든 픽셀에 해당하는 class로 분류하는 것 입니다.
  이미지에 있는 모든 픽셀에 대한 예측을 하는 것이기 때문에 dense prediction이라고도 부릅니다.
- Semantic Image Segmentation은 같은 class의 instance를 구별하지 않습니다.

### :rescue_worker_helmet: ​Semantic Segmentation Task 

- Input: RGB Image 또는 GrayScale Image
- Output: 각 픽셀별로 어떤 class에 속하는지 나타내는 레이블을 나타낸 Segmentation Map
- One-Hot Encoding으로 각 class에 대해 출력 채널을 만들어서 segmentation map을 만듭니다.
  Class의 개수 만큼 만들어진 채널을 argmax를 통해서 출력물을 내놓습니다.

### :books: Various Semantic Segmentation Methods

- AlexNet, VGG 등 분류에 자주 쓰이는 Deep한 신경망들은 Semantic Segmentation에 적합하지 않습니다.
- 이런 모델은 parameter의 개수와 차원을 줄이는 layer를 가지고 있기에 자세한 위치 정보를 잃게 됩니다.
  또한, 보통 마지막에 쓰이는 Fully Connected Layer에 의해서 위치에 대한 정보를 잃게 됩니다.
- 만약 공간, 위치에 대한 정보를 잃지 않기 위해서 Pooling과 Fully Connected Layer를 없애고 stride 1이고 Padding도 일정한 Convolution을 진행할 수도 있을 것입니다.
- Input의 차원은 보존하겠지만, parameter의 개수가 많아져서 메모리 문제나 계산하는데 resource가 너무 많이 들어서 현실적으로 불가능할 것입니다.
- 이 문제의 중간점을 찾기 위해서 보통 Semantic Segmentation 모델들은 보통 Downsampling & Upsampling의 형태를 가지고 있습니다.
  - Downsampling: 주 목적은 차원을 줄여서 적은 메모리로 Deep Convolution을 할 수 있게 하는 것입니다.
    보통 stride를 2 이상으로 하는 Convolution을 사용하거나, pooling을 사용합니다.
    이 과정을 진행하면 어쩔 수 없이 feature의 정보를 잃게 됩니다.
    마지막에 Fully-Connected Layer를 넣지 않고, Fully Connected Network를 주로 사용합니다.
    FCN 모델에서 이러한 방법을 제시한 후 이후에 나온 대부분의 모델들에서 사용하는 방법입니다.
  - Upsampling: Downsampling을 통해서 받은 결과의 차원을 늘려서 Input과 같은 차원으로 만들어 주는 과정입니다.
    주로 Strided Transpose Convolution을 사용합니다.
- 논문들에서는 Downsampling하는 부분을 인코더, Upsampling을 하는 부분을 디코더라고 부릅니다.
  인코더를 통해서 입력 받은 이미지의 정보를 압축된 벡터에 표현하고, 디코더를 통해서 원하는 결과물의 크기로 만들어냅니다.
- 이러한 인코더-디코더 형태를 가진 유명한 모델들로는 FCN, SegNet, UNet등이 있습니다.
  Shortcut connection을 어떻게 하느냐에 따라서 몇 가지의 모델들이 있습니다. 

