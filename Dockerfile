FROM lmsysorg/sglang-omni:dev

ARG HF_TOKEN
ARG MODEL_ID=bosonai/higgs-audio-v3-tts-4b

RUN pip install --no-cache-dir cryptography==41.0.7 runpod boto3 requests huggingface_hub

RUN HF_TOKEN=$HF_TOKEN huggingface-cli download $MODEL_ID --local-dir /models/higgs-v3

ENV MODEL_PATH=/models/higgs-v3

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
