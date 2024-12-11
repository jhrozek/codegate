from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime
import os
import ssl
import structlog
from typing import Tuple, Dict
from codegate.config import Config

logger = structlog.get_logger("codegate")

class CertificateAuthority:
    logger.debug("Initializing Certificate Authority class: CertificateAuthority")
    
    def __init__(self):
        self._ca_cert = None
        self._ca_key = None
        self._cert_cache: Dict[str, Tuple[str, str]] = {}
        self._load_or_generate_ca()

    def _load_or_generate_ca(self):
        """Load existing CA certificate and key or generate new ones"""
        logger.debug("Loading or generating CA certificate and key: fn: _load_or_generate_ca")
        ca_cert = Config.get_config().ca_cert
        ca_key = Config.get_config().ca_key

        if os.path.exists(ca_cert) and os.path.exists(ca_key):
            # Load existing CA certificate and key
            with open(ca_cert, "rb") as f:
                logger.debug(f"Loading CA certificate from {ca_cert}")
                self._ca_cert = x509.load_pem_x509_certificate(f.read())
            with open(ca_key, "rb") as f:
                logger.debug(f"Loading CA key from {ca_key}")
                self._ca_key = serialization.load_pem_private_key(f.read(), password=None)
        else:
            # Generate new CA certificate and key
            logger.debug("Generating new CA certificate and key")
            self._ca_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
            )

            name = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, "CodeGate CA"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CodeGate"),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "CodeGate"),
                x509.NameAttribute(NameOID.COUNTRY_NAME, "UK"),
            ])

            builder = x509.CertificateBuilder()
            builder = builder.subject_name(name)
            builder = builder.issuer_name(name)
            builder = builder.public_key(self._ca_key.public_key())
            builder = builder.serial_number(x509.random_serial_number())
            builder = builder.not_valid_before(datetime.datetime.utcnow())
            builder = builder.not_valid_after(
                datetime.datetime.utcnow() + datetime.timedelta(days=3650)  # 10 years
            )

            builder = builder.add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True,
            )

            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,  # This is a CA
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False
                ),
                critical=True,
            )

            self._ca_cert = builder.sign(
                private_key=self._ca_key,
                algorithm=hashes.SHA256(),
            )

            # Save CA certificate and key
            if not os.path.exists(Config.get_config().certs_dir):
                logger.debug(f"Creating directory: {Config.get_config().certs_dir}")
                os.makedirs(Config.get_config().certs_dir)

            with open(ca_cert, "wb") as f:
                logger.debug(f"Saving CA certificate to {ca_cert}")
                f.write(self._ca_cert.public_bytes(serialization.Encoding.PEM))

            with open(ca_key, "wb") as f:
                logger.debug(f"Saving CA key to {ca_key}")
                f.write(self._ca_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))

    def get_domain_certificate(self, domain: str) -> Tuple[str, str]:
        """Get or generate a certificate for a specific domain"""
        logger.debug(f"Getting domain certificate for domain: {domain} fn: get_domain_certificate")
        if domain in self._cert_cache:
            return self._cert_cache[domain]

        # Generate new certificate for domain
        logger.debug(f"Generating private key for domain: {domain}")
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,  # 2048 bits is sufficient for domain certs
        )

        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, domain),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Proxy Pilot Generated"),
        ])

        builder = x509.CertificateBuilder()
        builder = builder.subject_name(name)
        builder = builder.issuer_name(self._ca_cert.subject)
        builder = builder.public_key(key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.datetime.utcnow())
        builder = builder.not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        )

        # Add domain to SAN
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(domain)]),
            critical=False,
        )

        # Add extended key usage
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([
                ExtendedKeyUsageOID.SERVER_AUTH,
                ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=False,
        )

        # Basic constraints (not a CA)
        builder = builder.add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )

        logger.debug(f"Signing certificate for domain: {domain}")
        certificate = builder.sign(
            private_key=self._ca_key,
            algorithm=hashes.SHA256(),
        )

        # Save domain certificate and key
        logger.debug(f"Saving certificate and key for domain: {domain}")
        domain_cert_path = Config.get_config().certs_dir + f"/{domain}.crt"
        domain_key_path = Config.get_config().certs_dir + f"/{domain}.key"

        with open(domain_cert_path, "wb") as f:
            logger.debug(f"Saving certificate to {domain_cert_path}")
            f.write(certificate.public_bytes(serialization.Encoding.PEM))

        with open(domain_key_path, "wb") as f:
            logger.debug(f"Saving key to {domain_key_path}")
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        self._cert_cache[domain] = (domain_cert_path, domain_key_path)
        return domain_cert_path, domain_key_path

    def generate_certificates(self) -> Tuple[str, str]:
        """Generate self-signed certificates with proper extensions for HTTPS proxy"""
        logger.debug("Generating certificates fn: generate_certificates")

        if not os.path.exists(Config.get_config().certs_dir):
            logger.debug(f"Creating directory: {Config.get_config().certs_dir}")
            os.makedirs(Config.get_config().certs_dir)

        # Generate private key
        logger.debug("Generating private key for CA")
        ca_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
        )

        # Generate public key
        logger.debug("Generating public key for CA")
        ca_public_key = ca_private_key.public_key()

        # Add name attributes
        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "CodeGate CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CodeGate"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "CodeGate"),
            x509.NameAttribute(NameOID.COUNTRY_NAME, "UK"),
        ])

        # Create certificate builder
        builder = x509.CertificateBuilder()

        # Basic certificate information
        builder = builder.subject_name(name)
        builder = builder.issuer_name(name)
        builder = builder.public_key(ca_public_key)
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.datetime.utcnow())
        builder = builder.not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        )

        builder = builder.add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )

        # Add key usage extensions
        builder = builder.add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,  # This is a CA
                crl_sign=True,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True,
        )

        logger.debug("Signing CA certificate")
        ca_cert = builder.sign(
            private_key=ca_private_key,
            algorithm=hashes.SHA256(),
        )

        # Save CA certificate and key
        with open(Config.get_config().ca_cert, "wb") as f:
            logger.debug(f"Saving CA certificate to {Config.get_config().ca_cert}")
            f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

        with open(Config.get_config().ca_key, "wb") as f:
            logger.debug(f"Saving CA key to {Config.get_config().ca_key}")
            f.write(ca_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # CA generated, now generate server certificate

        ## Generate new certificate for domain
        logger.debug("Generating private key for server")
        server_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,  # 2048 bits is sufficient for domain certs
        )

        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "CodeGate Server"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "CodeGate"),
        ])

        # Add extended key usage extension
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(name)
        builder = builder.issuer_name(ca_cert.subject)
        builder = builder.public_key(server_key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.datetime.utcnow())
        builder = builder.not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        )

        # Add domain to SAN
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )

        # Add extended key usage
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([
                ExtendedKeyUsageOID.SERVER_AUTH,
                ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=False,
        )

        # Basic constraints (not a CA)
        builder = builder.add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )

        logger.debug("Signing server certificate")
        server_cert = builder.sign(
            private_key=ca_private_key,
            algorithm=hashes.SHA256(),
        )

        with open(Config.get_config().server_cert, "wb") as f:
            logger.debug(f"Saving server certificate to {Config.get_config().server_cert}")
            f.write(server_cert.public_bytes(serialization.Encoding.PEM))

        with open(Config.get_config().server_key, "wb") as f:
            logger.debug(f"Saving server key to {Config.get_config().server_key}")
            f.write(server_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Print instructions for trusting the certificates
        print("Certificates generated successfully in the 'certs' directory")
        print("\nTo trust these certificates:")
        print("\nOn macOS:")
        print("`sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain certs/ca.crt")
        print("\nOn Windows (PowerShell as Admin):")
        print("Import-Certificate -FilePath \"certs\\ca.crt\" -CertStoreLocation Cert:\\LocalMachine\\Root")
        print("\nOn Linux:")
        print("sudo cp certs/ca.crt /usr/local/share/ca-certificates/codegate.crt")
        print("sudo update-ca-certificates")
        print("\nFor VSCode, add to settings.json:")
        print('''{
    "http.proxy": "https://localhost:8990",
    "http.proxySupport": "on",
    "github.copilot.advanced": {
        "debug.testOverrideProxyUrl": "https://localhost:8990",
        "debug.overrideProxyUrl": "https://localhost:8990"
    }
}''')
        logger.debug("Certificates generated successfully")
        return server_cert, server_key

    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with secure configuration"""
        logger.debug("Creating SSL context fn: create_ssl_context")
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        logger.debug(f"Loading server certificate for ssl_context from: {Config.get_config().server_cert}")
        ssl_context.load_cert_chain(Config.get_config().server_cert, Config.get_config().server_key)
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        ssl_context.options |= (
            ssl.OP_NO_SSLv2 | 
            ssl.OP_NO_SSLv3 | 
            ssl.OP_NO_COMPRESSION |
            ssl.OP_CIPHER_SERVER_PREFERENCE
        )
        ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
        logger.debug("SSL context created successfully")
        return ssl_context

    def ensure_certificates_exist(self) -> None:
        """Ensure SSL certificates exist, generate if they don't"""
        logger.debug("Ensuring certificates exist. fn ensure_certificates_exist")
        if not (os.path.exists(Config.get_config().server_cert) and os.path.exists(Config.get_config().server_key)):
            logger.debug("Certificates not found, generating new certificates")
            self.generate_certificates()

    def get_ssl_context(self) -> ssl.SSLContext:
        """Get SSL context with certificates"""
        logger.debug("Getting SSL context fn: get_ssl_context")
        self.ensure_certificates_exist()
        return self.create_ssl_context()

    def get_cert_files(self) -> Tuple[str, str]:
        """Get certificate and key file paths"""
        logger.debug("Getting certificate and key file paths fn: get_cert_files")   
        return Config.get_config().server_cert, Config.get_config().server_key

# Initialize the Certificate Authority
ca = CertificateAuthority()
