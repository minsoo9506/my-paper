- [git wiki](https://github.com/minsoo9506/my-paper/wiki)에 리서치 관련 내용 간단 정리

- AE 모델 사용
- normal data만 이용하여 train
- train을 어느 정도 하고 reconstruction error를 기준으로 weight sampling with replacement
- 그러면 normal의 long-tail 부분을 덜 학습하게 되어서 abnormal을 더 잘 찾아줄 것이라는 아이디어 (boosting, metacost 로부터 아이디어 얻음)

## To to
- [o] 새해 전까지 데이터셋 5개정도 해보고 중간보고서 작성 -> 교수님께 메일 및 피드백
- [o] 아래 Results 결과 자동저장 코드 작성
- [ ] `run_tabular.py` 수정
- [ ] New 방법이 잘되고 안되는 경우 비교해보기
  - [ ] 시뮬레이션용 데이터 만들고 실험
- [ ] weight를 계산하는 방법 더 고민
  - 아니면 모든 sample의 weight를 계산하지 말고 recon_error가 큰 애들만 weight로 resampling? 제외?
  - 모든 sample의 weight를 계산하는게 너무 heavy하다

## 실험
- `early_stop_round`, `initial_epochs`, `hidden_size`, `sampling_term`
  - early_stop 없이 100, 200, 300 epoch 진행
  - early_stop=20, epoch=300 진행
  - 나머지 요소들은 일단 default만 진행