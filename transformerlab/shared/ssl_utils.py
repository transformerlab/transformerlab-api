from __future__ import annotations

import datetime as _dt
import ipaddress as _ip
from pathlib import Path
from typing import Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from filelock import FileLock

from transformerlab.shared.constants import WORKSPACE_DIR

__all__ = [
    "CERT_DIR",
    "CERT_PATH",
    "KEY_PATH",
    "ensure_persistent_self_signed_cert",
]

CERT_DIR: Path = Path(WORKSPACE_DIR) / "certs"
CERT_PATH: Path = CERT_DIR / "server-cert.pem"
KEY_PATH: Path = CERT_DIR / "server-key.pem"


def ensure_persistent_self_signed_cert() -> Tuple[str, str]:
    lock = CERT_DIR / ".cert.lock"
    with FileLock(str(lock)):
        if CERT_PATH.exists() and KEY_PATH.exists():
            return str(CERT_PATH), str(KEY_PATH)
        CERT_DIR.mkdir(parents=True, exist_ok=True)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"TransformerLab-Selfhost")
        ])
        cert_builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(_dt.datetime.utcnow() - _dt.timedelta(days=1))
            .not_valid_after(_dt.datetime.utcnow() + _dt.timedelta(days=3650))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.DNSName(u"localhost"),
                        x509.IPAddress(_ip.IPv4Address("127.0.0.1")),
                        x509.IPAddress(_ip.IPv6Address("::1")),
                    ]
                ),
                critical=False,
            )
        )
        cert = cert_builder.sign(key, hashes.SHA256())
        CERT_PATH.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        KEY_PATH.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )
        return str(CERT_PATH), str(KEY_PATH)
