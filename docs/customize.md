# カスタマイズ（JSONを入れるだけ）

## orbit_schedule.json（軌道）
`data/orbit_schedule.json` を置くと、太陽・地球・金星・IKAROSの位置がそのデータになります。

- day: 日（0,14,28… など）
- sun/earth/venus/ikaros: [x,y]

単位は AU でも km でもOK。ただし **全部同じ単位** にしてね。

## sensitivity_schedule.json（感度行列）
`data/sensitivity_schedule.json` を置くと、B-planeの動きがそのデータになります。

- turn: 0から
- C: 2×2 行列（β_in,β_out → B_T,B_R）
