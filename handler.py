import os
import subprocess
import time
import traceback

import requests
import runpod

SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/higgs-v3")
REF_DIR = os.environ.get("REF_DIR", "/tmp/higgs_refs")
READY_TIMEOUT = int(os.environ.get("SERVER_READY_TIMEOUT", "600"))
REQUEST_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", "600"))
SAMPLE_RATE = 24000

INIT_ERROR = None
_session = requests.Session()


def start_server():
    os.makedirs(REF_DIR, exist_ok=True)
    cmd = [
        "sgl-omni", "serve",
        "--model-path", MODEL_PATH,
        "--allowed-local-media-path", REF_DIR,
        "--port", str(SERVER_PORT),
    ]
    print(f"[init] Launching SGLang-Omni: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd)


def wait_for_ready(proc, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"sgl-omni exited early with code {proc.returncode}")
        for endpoint in ("/health", "/v1/models"):
            try:
                r = _session.get(SERVER_URL + endpoint, timeout=5)
                if r.status_code == 200:
                    print(f"[init] Server ready ({endpoint}).", flush=True)
                    return
            except requests.RequestException:
                pass
        time.sleep(3)
    raise RuntimeError(f"Timed out after {timeout}s waiting for sgl-omni to become ready")


try:
    _server_proc = start_server()
    wait_for_ready(_server_proc, READY_TIMEOUT)
except Exception:
    INIT_ERROR = traceback.format_exc()
    print(f"[init] FAILED:\n{INIT_ERROR}", flush=True)


def upload_to_r2(job_id, file_path, ext):
    import boto3

    s3 = boto3.client("s3",
        endpoint_url=os.environ["BUCKET_ENDPOINT_URL"],
        aws_access_key_id=os.environ["BUCKET_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["BUCKET_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["BUCKET_NAME"]
    key = f"outputs/{job_id}.{ext}"
    s3.upload_file(file_path, bucket, key)
    return s3.generate_presigned_url("get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=3600,
    )


def build_body(inp):
    text = inp.get("input") or inp.get("text")
    if not text:
        raise ValueError("input (text) is required")

    response_format = str(inp.get("response_format", "wav")).lower()
    body = {"input": text, "response_format": response_format, "stream": False}

    if inp.get("voice"):
        body["voice"] = inp["voice"]

    for key in ("temperature", "top_p"):
        if inp.get(key) is not None:
            body[key] = float(inp[key])
    for key in ("top_k", "max_new_tokens"):
        if inp.get(key) is not None:
            body[key] = int(inp[key])
    seed = inp.get("seed")
    if seed is not None and int(seed) != 0:
        body["seed"] = int(seed)

    return body, response_format


def handler(job):
    if INIT_ERROR:
        return {"error": f"Server failed to start:\n{INIT_ERROR}"}

    import base64
    import tempfile

    inp = job["input"]
    ref_path = None
    out_path = None

    try:
        body, response_format = build_body(inp)

        ref_b64 = inp.get("ref_audio_base64") or inp.get("reference_audio_base64")
        ref_text = inp.get("ref_text") or inp.get("reference_text")
        if ref_b64:
            os.makedirs(REF_DIR, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", dir=REF_DIR, delete=False)
            tmp.write(base64.b64decode(ref_b64))
            tmp.close()
            ref_path = tmp.name
            reference = {"audio_path": ref_path}
            if ref_text:
                reference["text"] = ref_text
            body["references"] = [reference]

        resp = _session.post(
            f"{SERVER_URL}/v1/audio/speech", json=body, timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return {"error": f"inference failed ({resp.status_code}): {resp.text[:1000]}"}
        if not resp.content:
            return {"error": "inference returned empty audio"}

        out_fd = tempfile.NamedTemporaryFile(suffix=f".{response_format}", delete=False)
        out_fd.write(resp.content)
        out_fd.close()
        out_path = out_fd.name

        audio_url = upload_to_r2(job["id"], out_path, response_format)
        return {
            "audio_url": audio_url,
            "format": response_format,
            "sample_rate": SAMPLE_RATE,
        }

    except ValueError as e:
        return {"error": str(e)}
    except Exception:
        return {"error": traceback.format_exc()}

    finally:
        for p in (ref_path, out_path):
            if p and os.path.exists(p):
                os.unlink(p)


runpod.serverless.start({"handler": handler})
