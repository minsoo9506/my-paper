- [git wiki](https://github.com/minsoo9506/my-paper/wiki)에 리서치 관련 내용 간단 정리

- AE 류 모델 사용
- normal data만 이용하여 train
- train을 어느 정도 하고 reconstruction error를 기준으로 weight sampling with replacement
- 그러면 normal의 long-tail 부분을 덜 학습하게 되어서 abnormal을 더 잘 찾아줄 것이라는 아이디어