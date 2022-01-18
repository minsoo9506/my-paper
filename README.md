- [git wiki](https://github.com/minsoo9506/my-paper/wiki)에 리서치 관련 내용 간단 정리

- AE 모델 사용
- normal data만 이용하여 train
- train을 어느 정도 하고 reconstruction error를 기준으로 weight sampling with replacement
- 그러면 normal의 long-tail 부분을 덜 학습하게 되어서 abnormal을 더 잘 찾아줄 것이라는 아이디어 (boosting, metacost 로부터 아이디어 얻음)

## To to
- [o] 새해 전까지 데이터셋 5개정도 해보고 중간보고서 작성 -> 교수님께 메일 및 피드백
- [o] 아래 Results 결과 자동저장 코드 작성
- [ ] New 방법이 잘되고 안되는 경우 비교해보기
  - [ ] 시뮬레이션용 데이터 만들고 실험
- [ ] 2차원 시뮬레이션용 데이터 만들어서 훈련과정 시각화, 결과 시각화 (Base보다 New방법의 score가 더 잘 나온 경우로 만들어서)
  - epoch지나면서 sample weight 변화? 히스토그램?
  - 노이즈데이터가 정말 안뽑히는지 train과정에서 나가리되는지 보여줘야함
  - 주피터에서 작은 모델 만들어서 진행
- [ ] weight를 계산하는 방법 더 고민 -> 아니면 next step으로 제안만?
  - 아니면 모든 sample의 weight를 계산하지 말고 recon_error가 큰 애들만 제외하고 recon_error 낮은 애들로 대체? (MetaCost 느낌)
  - 모든 sample의 weight를 계산하는게 너무 heavy하다

## 실험
- `early_stop_round`, `initial_epochs`, `hidden_size`, `sampling_term`
  - 완료
    - 크기가 작은 데이터셋 (time series의 경우)
    - early_stop 없이 100, 200, 300 epoch 진행
    - early_stop=20, epoch=300 진행
    - 나머지 요소들은 일단 default만 진행
  - To do
    - early_stop=20, epoch=1000 진행
    - `initial_epochs`는 전부 다 하지 말고 New의 결과가 잘나오는 몇 개만 더 해보자
      - 추가적인 성능 향상이 있는지 확인
      - initial epoch도 그러면 early_stop처럼 어느정도 수렴하면 자동으로 되도록...? how...? (next step?)