- [git wiki](https://github.com/minsoo9506/my-paper/wiki)에 리서치 관련 내용 간단 정리

- AE 류 모델 사용
- normal data만 이용하여 train
- train을 어느 정도 하고 reconstruction error를 기준으로 weight sampling with replacement
- 그러면 normal의 long-tail 부분을 덜 학습하게 되어서 abnormal을 더 잘 찾아줄 것이라는 아이디어

## 생각정리
- train, val, test set 들의 anomaly score 분석해서 insight 뽑을 수 있을듯
  - New 방법 사용시 train anomaly score가 많이 커지면 test set의 recall이 높아진다? 등등

## To to
- [ ] `trainer.py` 함수 doc
- [ ] `utils.py` 함수 doc
- [ ] 새해 전까지 데이터셋 5개정도 해보고 중간보고서 작성 -> 교수님께 메일 및 피드백