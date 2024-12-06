import datetime
import os
import ssl
from typing import Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from codegate.config import Config


class CertificateManager:
    def __init__(self):
        self.cfg = Config.load()

    def generate_certificates(self) -> Tuple[str, str]:
        """Generate self-signed certificates with proper extensions for HTTPS proxy"""

        # Create certificates directory if it doesn't exist
        if not os.path.exists(self.cfg.certs):
            os.makedirs(self.cfg.certs)

        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )

        # Generate public key
        public_key = private_key.public_key()

        # Create certificate builder
        builder = x509.CertificateBuilder()

        # Add name attributes
        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "Proxy Pilot CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Proxy Pilot"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Development"),
            x509.NameAttribute(NameOID.COUNTRY_NAME, "UK"),
        ])

        # Basic certificate information
        builder = builder.subject_name(name)
        builder = builder.issuer_name(name)
        builder = builder.public_key(public_key)
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.datetime.utcnow())
        builder = builder.not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        )

        # Add certificate extensions
        builder = builder.add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("*.github.com"),
                x509.DNSName("*.githubusercontent.com"),
                x509.DNSName("*.githubcopilot.com"),
                x509.DNSName("api.github.com"),
                x509.DNSName("github.com"),
                x509.DNSName("api.githubcopilot.com"),
                x509.DNSName("copilot-proxy.githubusercontent.com"),
                x509.DNSName("default.exp-tas.com"),
                x509.DNSName("*.exp-tas.com"),
            ]),
            critical=False,
        )

        # Add key usage extensions
        builder = builder.add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True,
        )

        # Add extended key usage extension
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([
                ExtendedKeyUsageOID.SERVER_AUTH,
                ExtendedKeyUsageOID.CLIENT_AUTH,
                ExtendedKeyUsageOID.CODE_SIGNING,
                ExtendedKeyUsageOID.EMAIL_PROTECTION,
            ]),
            critical=False,
        )

        # Add basic constraints extension
        builder = builder.add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )

        # Add key identifier extensions
        builder = builder.add_extension(
            x509.SubjectKeyIdentifier.from_public_key(public_key),
            critical=False,
        )

        builder = builder.add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(public_key),
            critical=False,
        )

        # Sign the certificate
        certificate = builder.sign(
            private_key=private_key,
            algorithm=hashes.SHA256(),
        )

        # Write private key
        key_path = os.path.join(self.cfg.certs, "server.key")
        with open(key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Write certificate
        cert_path = os.path.join(self.cfg.certs, "server.crt")
        with open(cert_path, "wb") as f:
            f.write(certificate.public_bytes(serialization.Encoding.PEM))

        return cert_path, key_path

    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with secure configuration"""
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(self.cfg.cert_file, self.cfg.key_file)
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        ssl_context.options |= (
            ssl.OP_NO_SSLv2 |
            ssl.OP_NO_SSLv3 |
            ssl.OP_NO_COMPRESSION |
            ssl.OP_CIPHER_SERVER_PREFERENCE
        )
        ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
        return ssl_context

    def ensure_certificates_exist(self) -> None:
        """Ensure SSL certificates exist, generate if they don't"""
        if not (os.path.exists(self.cfg.cert_file) and os.path.exists(self.cfg.key_file)):
            self.generate_certificates()
