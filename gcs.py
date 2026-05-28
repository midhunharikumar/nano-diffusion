"""
GCS checkpoint uploader.

Uploads each saved checkpoint to gs://{bucket}/nano_diffusion/{run_name}/
and keeps a latest.pt pointer at the same prefix for easy resuming.

Authentication — any of the standard GCP credential chains work:
  • GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json  (service account file)
  • gcloud auth application-default login             (interactive / CI)
  • Workload Identity / instance metadata             (GCE / GKE / Cloud Run)
"""
from __future__ import annotations
from pathlib import Path


class GCSCheckpointUploader:
    def __init__(self, bucket: str, run_name: str):
        """
        bucket   : GCS bucket name, e.g. "checkpoints"
        run_name : used to build the blob prefix nano_diffusion/{run_name}/
        """
        from google.cloud import storage  # lazy import — only needed when GCS is on
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)
        self._prefix = f"nano_diffusion/{run_name}/"
        print(f"GCS uploader ready → gs://{bucket}/{self._prefix}")

    def upload(self, local_path: str | Path) -> str:
        """
        Upload local_path and refresh latest.pt at the same prefix.
        Returns the gs:// URI of the uploaded file.
        Exceptions are caught and printed so a GCS hiccup never kills training.
        """
        local_path = Path(local_path)
        try:
            blob_name = self._prefix + local_path.name
            self._bucket.blob(blob_name).upload_from_filename(str(local_path))
            self._bucket.blob(self._prefix + "latest.pt").upload_from_filename(str(local_path))
            uri = f"gs://{self._bucket.name}/{blob_name}"
            print(f"  uploaded → {uri}")
            return uri
        except Exception as exc:
            print(f"  warning: GCS upload failed ({exc})")
            return ""
