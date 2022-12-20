욜랙트 안드로이드 앱

src 폴더 안에 분할압축된 모델 압축해제 후 안드로이드 스튜디오로 빌드

gradle 설정:

android sdk 32
android sdk tools 31.0.0

dependencies {
    implementation 'org.pytorch:pytorch_android_lite:1.12.2'
    implementation 'org.pytorch:pytorch_android_torchvision_lite:1.12.2'
    implementation 'org.pytorch:torchvision_ops:0.13.1'
}

앱 원본 : 파이토치 안드로이드 데모 앱
https://github.com/pytorch/android-demo-app

모델 :
Yolact
@article{bolya-arxiv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  journal   = {arXiv},
  year      = {2019},
}

코드 참조 :
Yolact to onnx
https://github.com/Ma-Dan/yolact.git

Yolact coreml
https://github.com/Ma-Dan/Yolact-CoreML
