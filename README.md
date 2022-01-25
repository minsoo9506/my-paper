- [git wiki](https://github.com/minsoo9506/my-paper/wiki)에 리서치 관련 내용 간단 정리

- AE 모델 사용
- normal data만 이용하여 train
- train을 어느 정도 하고 reconstruction error를 기준으로 weight sampling with replacement
- 그러면 normal의 long-tail 부분을 덜 학습하게 되어서 abnormal을 더 잘 찾아줄 것이라는 아이디어 (boosting, metacost 로부터 아이디어 얻음)

## To to
- recon_error가 높은 상위 ~% 대신해서 낮은 상위 ~%의 데이터를 넣어서 학습시켜보기
  - weight sampling을 하는게 아니라 rule-based로 해보는 것 -> 이것도 성능은 잘 나오는 듯
- 근데 noise를 잘 걸러내느냐? 이건 또 의문....일단 noise의 recon-error가 평균적으로는 높게 나온다.
  - 그리고 noise의 정도가 심해지니까 epoch을 진행할 수록 recon-error 평균이 점점 높아지는 모습!

## 실험
- `early_stop_round`, `initial_epochs`, `hidden_size`, `sampling_term`
  - 완료
    - 크기가 작은 데이터셋 (time series의 경우)
    - early_stop 없이 100, 200, 300 epoch 진행
    - early_stop=30, epoch=1000 진행
    - 나머지 요소들은 일단 default만 진행
  - To do
    - epoch=1000으로 fix
    - 모든 데이터는 하기 어려우니까 tabular simulation dataset을 이용
      - 아래를 통해 원하는 바?
      - epoch 지남에 따라 noise들의 recon-error 변화 + 실제로 몇 개나 train에서 빠지는지 갯수
    - `early_stop_round`: 10, 50, 100
    - `initial_epochs`: 5, 20
    - `sampling_term`: 1, 4, 16
    - data의 크기(row): 5000, 50000, 500000
    - noise 심한 정도(noise 만들 때 가중평균): 0.1, 0.5, 0.9
    - noise의 비율: 0.01, 0.1
