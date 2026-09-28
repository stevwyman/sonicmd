from __future__ import annotations

import os

SAMPLE_RATE = 16000
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def configure_ssl() -> None:
    """Point urllib/OpenSSL at certifi. macOS Python often has an empty default CA store."""
    try:
        import certifi

        ca = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", ca)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
        os.environ.setdefault("CURL_CA_BUNDLE", ca)
    except Exception:
        return


configure_ssl()
