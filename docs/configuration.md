# Configuration System

The configuration system in Codegate is managed through the `Config` class in `config.py`. It supports multiple configuration sources with a clear priority order.

## Configuration Priority (highest to lowest)

1. CLI arguments
2. Environment variables
3. Config file (YAML)
4. Default values (including default prompts from prompts/default.yaml)

## Default Configuration Values

- Port: 8989
- Host: "localhost"
- Log Level: "INFO"
- Log Format: "JSON"
- Prompts: Default prompts from prompts/default.yaml
- Provider URLs:
  - vLLM: "http://localhost:8000"
  - OpenAI: "https://api.openai.com/v1"
  - Anthropic: "https://api.anthropic.com/v1"
  - Ollama: "http://localhost:11434"
- Certificate Configuration:
  - Certs Directory: "./certs"
  - CA Certificate: "ca.crt"
  - CA Key: "ca.key"
  - Server Certificate: "server.crt"
  - Server Key: "server.key"

## Configuration Methods

### From File

Load configuration from a YAML file:

```python
config = Config.from_file("config.yaml")
```

Example config.yaml:
```yaml
port: 8989
host: localhost
log_level: INFO
log_format: JSON
provider_urls:
  vllm: "https://vllm.example.com"
  openai: "https://api.openai.com/v1"
  anthropic: "https://api.anthropic.com/v1"
  ollama: "http://localhost:11434"
certs_dir: "./certs"
ca_cert: "ca.crt"
ca_key: "ca.key"
server_cert: "server.crt"
server_key: "server.key"
```

### From Environment Variables

Environment variables are automatically loaded with these mappings:

- `CODEGATE_APP_PORT`: Server port
- `CODEGATE_APP_HOST`: Server host
- `CODEGATE_APP_LOG_LEVEL`: Logging level
- `CODEGATE_LOG_FORMAT`: Log format
- `CODEGATE_PROMPTS_FILE`: Path to prompts YAML file
- `CODEGATE_PROVIDER_VLLM_URL`: vLLM provider URL
- `CODEGATE_PROVIDER_OPENAI_URL`: OpenAI provider URL
- `CODEGATE_PROVIDER_ANTHROPIC_URL`: Anthropic provider URL
- `CODEGATE_PROVIDER_OLLAMA_URL`: Ollama provider URL
- `CODEGATE_CERTS_DIR`: Directory for certificate files
- `CODEGATE_CA_CERT`: CA certificate file name
- `CODEGATE_CA_KEY`: CA key file name
- `CODEGATE_SERVER_CERT`: Server certificate file name
- `CODEGATE_SERVER_KEY`: Server key file name

```python
config = Config.from_env()
```

## Configuration Options

### Provider URLs

Provider URLs can be configured in several ways:

1. In Configuration File:
   ```yaml
   provider_urls:
     vllm: "https://vllm.example.com"  # /v1 path is added automatically
     openai: "https://api.openai.com/v1"
     anthropic: "https://api.anthropic.com/v1"
     ollama: "http://localhost:11434"  # /api path is added automatically
   ```

2. Via Environment Variables:
   ```bash
   export CODEGATE_PROVIDER_VLLM_URL=https://vllm.example.com
   export CODEGATE_PROVIDER_OPENAI_URL=https://api.openai.com/v1
   export CODEGATE_PROVIDER_ANTHROPIC_URL=https://api.anthropic.com/v1
   export CODEGATE_PROVIDER_OLLAMA_URL=http://localhost:11434
   ```

3. Via CLI Flags:
   ```bash
   codegate serve --vllm-url https://vllm.example.com --ollama-url http://localhost:11434
   ```

Note: 
- For the vLLM provider, the /v1 path is automatically appended to the base URL if not present.
- For the Ollama provider, the /api path is automatically appended to the base URL if not present.

### Certificate Configuration

Certificate files can be configured in several ways:

1. In Configuration File:
   ```yaml
   certs_dir: "./certs"
   ca_cert: "ca.crt"
   ca_key: "ca.key"
   server_cert: "server.crt"
   server_key: "server.key"
   ```

2. Via Environment Variables:
   ```bash
   export CODEGATE_CERTS_DIR=./certs
   export CODEGATE_CA_CERT=ca.crt
   export CODEGATE_CA_KEY=ca.key
   export CODEGATE_SERVER_CERT=server.crt
   export CODEGATE_SERVER_KEY=server.key
   ```

3. Via CLI Flags:
   ```bash
   codegate serve --certs-dir ./certs --ca-cert ca.crt --ca-key ca.key --server-cert server.crt --server-key server.key
   ```

### Log Levels

Available log levels (case-insensitive):

- `ERROR`
- `WARNING`
- `INFO`
- `DEBUG`

### Log Formats

Available log formats (case-insensitive):

- `JSON`
- `TEXT`

### Prompts Configuration

Prompts can be configured in several ways:

1. Default Prompts:
   - Located in prompts/default.yaml
   - Loaded automatically if no other prompts are specified

2. In Configuration File:
   ```yaml
   # Option 1: Direct prompts definition
   prompts:
     my_prompt: "Custom prompt text"
     another_prompt: "Another prompt text"

   # Option 2: Reference to prompts file
   prompts: "path/to/prompts.yaml"
   ```

3. Via Environment Variable:
   ```bash
   export CODEGATE_PROMPTS_FILE=path/to/prompts.yaml
   ```

4. Via CLI Flag:
   ```bash
   codegate serve --prompts path/to/prompts.yaml
   ```

### Prompts File Format

Prompts files should be in YAML format with string values:

```yaml
prompt_name: "Prompt text content"
another_prompt: "More prompt text"
```

Access prompts in code:
```python
config = Config.load()
prompt = config.prompts.prompt_name
```

## Error Handling

The configuration system uses a custom `ConfigurationError` exception for handling configuration-related errors, such as:

- Invalid port numbers (must be between 1 and 65535)
- Invalid log levels
- Invalid log formats
- YAML parsing errors
- File reading errors
- Invalid prompt values (must be strings)
- Missing or invalid prompts files
