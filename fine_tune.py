#!/usr/bin/env python3
import os
import time
import logging
import requests

# ——— Configuration ———
TRAIN_FT      = "dataset/train_ft.jsonl"
VAL_FT        = "dataset/val_ft.jsonl"
API_KEY       = os.getenv("OPENAI_API_KEY")
MODEL         = "gpt-4o-2024-08-06"
POLL_INTERVAL = 30    # seconds
TIMEOUT       = 3600  # seconds

# ——— Logging setup ———
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

if not API_KEY:
    logging.error("OPENAI_API_KEY not set in environment.")
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

def upload_file(path: str) -> str:
    """Upload a file via REST API; return file_id."""
    logging.info(f"Uploading {path} …")
    with open(path, "rb") as f:
        files = {"file": f}
        data = {"purpose": "fine-tune"}
        resp = requests.post(
            "https://api.openai.com/v1/files",
            headers=HEADERS,
            files=files,
            data=data
        )
    try:
        resp.raise_for_status()
    except Exception:
        logging.error(f"Upload failed: {resp.status_code} {resp.text}")
        resp.raise_for_status()
    file_id = resp.json().get("id")
    logging.info(f"Uploaded {path} → file_id={file_id}")
    return file_id

def create_fine_tune(train_id: str, val_id: str) -> str:
    """Create a fine-tuning job; return job_id."""
    payload = {
        "training_file": train_id,
        "validation_file": val_id,
        "model": MODEL
    }
    # Correct endpoint
    url = "https://api.openai.com/v1/fine-tunes"
    logging.info(f"Creating fine-tune job at {url}")
    logging.debug(f"Payload: {payload}")

    resp = requests.post(
        url,
        headers={**HEADERS, "Content-Type": "application/json"},
        json=payload
    )

    # Debug information
    logging.debug(f"Request URL: {resp.request.url}")
    logging.debug(f"Request headers: {resp.request.headers}")
    logging.debug(f"Request body: {resp.request.body}")
    logging.debug(f"Response status: {resp.status_code}")
    logging.debug(f"Response body: {resp.text}")

    try:
        resp.raise_for_status()
    except Exception:
        logging.error(f"Fine-tune creation failed: {resp.status_code} {resp.text}")
        resp.raise_for_status()

    job_id = resp.json().get("id")
    logging.info(f"Fine-tune job created: job_id={job_id}")
    return job_id

def wait_for_completion(job_id: str) -> dict:
    """Poll job status until succeeded or failed; return final job dict."""
    logging.info(f"Polling status for job {job_id} …")
    start = time.time()
    url = f"https://api.openai.com/v1/fine-tunes/{job_id}"
    while True:
        resp = requests.get(url, headers=HEADERS)
        try:
            resp.raise_for_status()
        except Exception:
            logging.error(f"Status check failed: {resp.status_code} {resp.text}")
            resp.raise_for_status()
        job = resp.json()
        status = job.get("status")
        logging.info(f"Status: {status}")
        if status == "succeeded":
            return job
        if status == "failed":
            err = job.get("fine_tune_error", "No error message")
            logging.error(f"Job failed: {err}")
            raise RuntimeError(f"Fine-tune failed: {err}")
        if time.time() - start > TIMEOUT:
            logging.error("Timeout waiting for fine-tune completion.")
            raise TimeoutError("Fine-tune did not complete in time.")
        time.sleep(POLL_INTERVAL)

def save_model_name(name: str, out="fine_tuned_model.txt"):
    """Save the fine-tuned model name locally."""
    logging.info(f"Saving model name `{name}` to {out}")
    with open(out, "w") as f:
        f.write(name)
    logging.info("Model name saved successfully.")

def main():
    # 1) Upload JSONL files
    train_id = upload_file(TRAIN_FT)
    val_id   = upload_file(VAL_FT)

    # 2) Create the fine-tune job
    job_id = create_fine_tune(train_id, val_id)

    # 3) Wait for job to finish
    final_job = wait_for_completion(job_id)

    # 4) Save the resulting model name
    model_name = final_job.get("fine_tuned_model")
    if model_name:
        save_model_name(model_name)
        logging.info(f"Your fine-tuned model is: {model_name}")
    else:
        logging.error("No `fine_tuned_model` in the final response.")

if __name__ == "__main__":
    main()
