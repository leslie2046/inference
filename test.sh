#!/bin/sh
curl -X 'POST'   'http://192.168.1.33:9997/v1/rerank'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
 "model": "bge-reranker-large",
 "query": "A man is eating pasta.",
 "documents": [
     "A man is eating food.",
     "A man is eating a piece of bread.",
     "The girl is carrying a baby.",
     "A man is riding a horse.",
     "A woman is playing violin."]
	 }' -w "\n时间总计: %{time_total} 秒\n"
	 
curl -X 'POST'   'http://192.168.1.33:9997/v1/rerank'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
 "model": "jina-reranker-v2",
 "query": "A man is eating pasta.",
 "documents": [
     "A man is eating food.",
     "A man is eating a piece of bread.",
     "The girl is carrying a baby.",
     "A man is riding a horse.",
     "A woman is playing violin."]
	 }' -w "\n时间总计: %{time_total} 秒\n"  

curl -X 'POST'   'http://192.168.1.33:9997/v1/embeddings' \
-H 'accept: application/json'   -H 'Content-Type: application/json' \
-d '{
"model": "bge-m3",
"input": ["我是中国人"]
}'  -w "\n时间总计: %{time_total} 秒\n"

curl -X 'POST'   'http://192.168.1.33:9997/v1/embeddings' \
-H 'accept: application/json'   -H 'Content-Type: application/json' \
-d '{
"model": "jina-embeddings-v3",
"input": ["我是中国人"]
}'  -w "\n时间总计: %{time_total} 秒\n"

curl -X 'POST'   'http://192.168.1.33:9997/v1/audio/transcriptions' \
    -H 'accept: application/json' \
    -H "Content-Type: multipart/form-data" \
    -F file="@./bill_gates-TED.mp3" \
    -F model="SenseVoiceSmall" \
    -w "\n时间总计: %{time_total} 秒\n"
    
curl -X 'POST'   'http://192.168.1.33:9997/v1/audio/transcriptions' \
    -H 'accept: application/json' \
    -H "Content-Type: multipart/form-data" \
    -F file="@./chinese_test.wav" \
    -F use_itn="false" \
    -F model="SenseVoiceSmall" \
    -F language="zh" \
    -w "\n时间总计: %{time_total} 秒\n"
curl -X 'POST'   'http://192.168.1.33:9997/v1/audio/transcriptions' \
    -H 'accept: application/json' \
    -H "Content-Type: multipart/form-data" \
    -F file="@./bill_gates-TED.mp3" \
    -F model="SenseVoiceSmall-CPU" \
    -w "\n时间总计: %{time_total} 秒\n"
    
curl -X 'POST'   'http://192.168.1.33:9997/v1/audio/transcriptions' \
    -H 'accept: application/json' \
    -H "Content-Type: multipart/form-data" \
    -F file="@./chinese_test.wav" \
    -F use_itn="false" \
    -F model="SenseVoiceSmall-CPU" \
    -F language="zh" \
    -w "\n时间总计: %{time_total} 秒\n"
: <<EOF
curl -X 'POST'   'http://192.168.1.33:9997/v1/audio/transcriptions' \
    -H 'accept: application/json' \
    -H "Content-Type: multipart/form-data" \
    -F file="@./bill_gates-TED.mp3" \
    -F model="whisper-large-v3-turbo" \
    -w "\n时间总计: %{time_total} 秒\n"
curl -X 'POST'   'http://192.168.1.33:9997/v1/images/ocr' \
    -H 'accept: application/json' \
    -H "Content-Type: multipart/form-data" \
    -F "image=@./assets_train_sample.jpg;type=application/octet-stream" \
    -F model="GOT-OCR2_0" \
    -F ocr_type="format" \
    -w "\n时间总计: %{time_total} 秒\n"
EOF
