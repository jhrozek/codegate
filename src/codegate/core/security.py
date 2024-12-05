from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime
import os
import ssl
from typing import Tuple

from codegate.config import Config

def generate_certificates() -> Tuple[str, str]:
    """Generate self-signed certificates with proper extensions for HTTPS proxy"""
    # Create certificates directory if it doesn't exist
    if not os.path.exists(settings.CERT_DIR):
        os.makedirs(settings.CERT_DIR)

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

    # Add key usage extensions
    builder = builder.add_extension(
        x509.KeyUsage(
            digital_signature=True,
            content_commitment=False,
            key_encipherment=True,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=False,
            crl_sign=False,
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
    key_path = os.path.join(settings.CERT_DIR, "server.key")
    with open(key_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    # Write certificate
    cert_path = os.path.join(settings.CERT_DIR, "server.crt")
    with open(cert_path, "wb") as f:
        f.write(certificate.public_bytes(serialization.Encoding.PEM))

    print("Certificates generated successfully in the 'certs' directory")
    print("\nTo trust these certificates:")
    print("\nOn macOS:")
    print("sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain certs/server.crt")
    print("\nOn Windows (PowerShell as Admin):")
    print("Import-Certificate -FilePath \"certs\\server.crt\" -CertStoreLocation Cert:\\LocalMachine\\Root")
    print("\nOn Linux:")
    print("sudo cp certs/server.crt /usr/local/share/ca-certificates/proxy-pilot.crt")
    print("sudo update-ca-certificates")
    print("\nFor VSCode, add to settings.json:")
    print('''{
    "http.proxy": "https://localhost:8989",
    "http.proxySupport": "on",
    "github.copilot.advanced": {
        "debug.testOverrideProxyUrl": "https://localhost:8989",
        "debug.overrideProxyUrl": "https://localhost:8989"
    }
}''')

    return cert_path, key_path

def create_ssl_context() -> ssl.SSLContext:
    """Create SSL context with secure configuration"""
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(settings.CERT_FILE, settings.KEY_FILE)
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
    ssl_context.options |= (
        ssl.OP_NO_SSLv2 | 
        ssl.OP_NO_SSLv3 | 
        ssl.OP_NO_COMPRESSION |
        ssl.OP_CIPHER_SERVER_PREFERENCE
    )
    ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
    return ssl_context

def ensure_certificates_exist() -> None:
    """Ensure SSL certificates exist, generate if they don't"""
    print("Checking certs")
    if not (os.path.exists(settings.CERT_FILE) and os.path.exists(settings.KEY_FILE)):
        generate_certificates()

def get_ssl_context() -> ssl.SSLContext:
    """Get SSL context with certificates"""
    ensure_certificates_exist()
    return create_ssl_context()

def get_cert_files() -> Tuple[str, str]:
    """Get certificate and key file paths"""
    return settings.CERT_FILE, settings.KEY_FILE
