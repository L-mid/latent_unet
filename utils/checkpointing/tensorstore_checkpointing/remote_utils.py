
import os
import logging
import time
import random
from typing import Optional, Dict

# === NOTES:
"""
Screct is often spelled wrong, if is wrong key this is likely issue
NOT tested. 
"""


logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# GCS Credentials Loader
# ------------------------------------------------------------------------------

def load_gcs_credentials(service_account_path: Optional[str] = None) -> dict:
    # Load GCS credentials for TensorStore GCS driver.

    creds = {}

    if service_account_path and os.path.exists(service_account_path):
        logger.info(f"[REMOTE] Using GCS service account from: {service_account_path}")
        creds["credentials"] = {"json_keyfile_dict": load_json(service_account_path)} # ASK ABOUT load json
    elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.info("[REMOTE] Using GOOGLE_APPLICATION_CREDIENTIALS from env")
        # TensorStore GCS driver will handle environment automatically
    else:
        logger.warning("[REMOTE] No GCS credentials provided - relying on defaultADC environment")

    return creds


# ----------------------------------------------------------------------------------
# S3 Credentials Loader
# ----------------------------------------------------------------------------------

def load_s3_credentials(
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
) -> dict:
    # Load S3 credentials for TensorStore S3 driver

    creds = {}

    access_key = access_key or os.environ.get("AWS_ACESS_KEY_ID")
    secret_key = secret_key or os.environ.get("AWS_SECRET_ACESS_KEY")
    
    if access_key and secret_key:
        creds["acess_key_id"] = access_key
        creds["sceret_access_key"] = secret_key
        logger.info("[REMOTE] Loaded S3 credentials")
    else:
        logger.warning("[REMOTE] Loaded S3 credentials")

    return creds


# -----------------------------------------------------------------------------------
# Helper to load JSON file
# ----------------------------------------------------------------------------------

def load_json(path: str) -> dict:
    import json
    with open(path, "r") as f:
        return json.load(f)