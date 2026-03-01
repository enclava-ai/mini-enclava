# Enclava

**Confidential AI Platform for businesses**

Enclava is a comprehensive AI platform that makes privacy practical. It provides easy to create openai compatible chatbots and API endpoints with knowledge base access (RAG). All in a completely confidential way through [privatemode.ai](https://privatemode.ai)

## Key Features

- **AI Chatbots** - Customizable chatbots with prompt templates and RAG integration (openai compatible)
- **RAG System** - Document upload, processing, and semantic search with Qdrant
- **TEE Security** - Privacy-protected LLM inference via confidential computing
- **OpenAI Compatible** - Standard API endpoints for seamless integration with existing tools 
- **Budget Management** - Built-in spend tracking and usage limits

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Git
- [privatemode.ai](https://privatemode.ai) api key

### 1. Clone Repository

```bash
git clone <repository-url>
cd enclava
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
vim .env
```

**Required Configuration:**
```bash
# Security
JWT_SECRET=your-super-secret-jwt-key-here-change-in-production

# PrivateMode.ai API Key (optional but recommended)
PRIVATEMODE_API_KEY=your-privatemode-api-key

# Base URL for CORS and frontend
BASE_URL=localhost
```

### 3. Deploy with Docker

```bash
# Start all services
docker compose up --build

# Or run in background
docker compose up --build -d
```

### 4. Access Application

- **Main Application**: http://localhost
- **API Documentation**: http://localhost/docs (backend API)
- **Qdrant Dashboard**: http://localhost:56333/dashboard

### 5. Default Login

- **Username**: `admin`
- **Password**: `admin123`

*Change default credentials immediately in production!*


## Configuration

### Environment Variables

See `.env.example` for all available configuration options.

## Attestation Policy Artifacts

On every tagged release (`v*.*.*`), GitHub Actions now generates a digest-pinned
attestation policy for the pushed container image.

- Workflow: `.github/workflows/mini-build.yml`
- Generator script: `.github/scripts/generate_attestation_policy.py`
- Release assets:
  - `mini-enclava-attestation-policy-<tag>.json`
  - `mini-enclava-attestation-policy-<tag>.json.sha256`
  - `mini-enclava-attestation-policy-<tag>.json.sigstore.json`
  - `mini-enclava-attestation-policy-<tag>.json.verify.txt`

This lets clients download the policy directly from the release assets and
verify attestation without manual policy generation.

Policy signatures use keyless Sigstore (`cosign`) with GitHub OIDC identity.
Clients can verify authenticity with:

```bash
cosign verify-blob \
  --bundle mini-enclava-attestation-policy-<tag>.json.sigstore.json \
  --certificate-identity "https://github.com/<org>/<repo>/.github/workflows/mini-build.yml@refs/tags/<tag>" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  mini-enclava-attestation-policy-<tag>.json
```


## Support

- **Documentation**: [docs.enclava.ai](https://docs.enclava.ai)
- **Issues**: Use the GitHub issue tracker
- **Security**: Report security issues privately



---
