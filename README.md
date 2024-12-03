## 설치 및 실행

```sh
apt-get install libgl1
mkdir uploads
python -m pip install -r requirements.txt
flask --app=application run --host=0.0.0.0 --port=8080
```

## API 호출

```sh
curl -X POST -F "file=@data_img/IMG_4844.JPG" -H "Content-Type: multipart/form-data" localhost:8080/image
```

예시 응답

```json
{
  "schedule": [
    {
      "day": "목",
      "fill_ratio": 0.9153473277949253,
      "time": "09:00",
      "x": 593,
      "y": 54
    },
    {
      "day": "목",
      "fill_ratio": 0.9580882766951776,
      "time": "10:00",
      "x": 593,
      "y": 186
    },
    {
      "day": "목",
      "fill_ratio": 0.8897539452199239,
      "time": "12:00",
      "x": 593,
      "y": 450
    },
    {
      "day": "목",
      "fill_ratio": 0.963462887035928,
      "time": "13:00",
      "x": 593,
      "y": 582
    },
    {
      "day": "목",
      "fill_ratio": 0.9580882766951776,
      "time": "14:00",
      "x": 593,
      "y": 714
    },
    {
      "day": "목",
      "fill_ratio": 0.7533412160951664,
      "time": "17:00",
      "x": 593,
      "y": 1110
    },
    {
      "day": "목",
      "fill_ratio": 0.9580882766951776,
      "time": "18:00",
      "x": 593,
      "y": 1242
    },
    {
      "day": "수",
      "fill_ratio": 0.90958881671555,
      "time": "11:00",
      "x": 416,
      "y": 318
    },
    {
      "day": "수",
      "fill_ratio": 0.9966063174705547,
      "time": "12:00",
      "x": 416,
      "y": 450
    },
    {
      "day": "수",
      "fill_ratio": 0.8046559481580442,
      "time": "15:00",
      "x": 416,
      "y": 846
    },
    {
      "day": "수",
      "fill_ratio": 0.9745960084560535,
      "time": "16:00",
      "x": 416,
      "y": 978
    },
    {
      "day": "월",
      "fill_ratio": 0.8164289041425449,
      "time": "15:00",
      "x": 62,
      "y": 846
    },
    {
      "day": "월",
      "fill_ratio": 0.9966063174705547,
      "time": "16:00",
      "x": 62,
      "y": 978
    },
    {
      "day": "월",
      "fill_ratio": 0.9966063174705547,
      "time": "20:00",
      "x": 62,
      "y": 1506
    },
    {
      "day": "토",
      "fill_ratio": 0.9138117248404252,
      "time": "20:00",
      "x": 947,
      "y": 1506
    },
    {
      "day": "화",
      "fill_ratio": 0.946827188362177,
      "time": "09:00",
      "x": 239,
      "y": 54
    },
    {
      "day": "화",
      "fill_ratio": 0.9911037402169295,
      "time": "10:00",
      "x": 239,
      "y": 186
    },
    {
      "day": "화",
      "fill_ratio": 0.9911037402169295,
      "time": "11:00",
      "x": 239,
      "y": 318
    },
    {
      "day": "화",
      "fill_ratio": 0.9911037402169295,
      "time": "12:00",
      "x": 239,
      "y": 450
    },
    {
      "day": "화",
      "fill_ratio": 0.8361358087252959,
      "time": "14:00",
      "x": 239,
      "y": 714
    },
    {
      "day": "화",
      "fill_ratio": 0.7710006500719173,
      "time": "17:00",
      "x": 239,
      "y": 1110
    },
    {
      "day": "화",
      "fill_ratio": 0.9911037402169295,
      "time": "18:00",
      "x": 239,
      "y": 1242
    }
  ]
}
```
