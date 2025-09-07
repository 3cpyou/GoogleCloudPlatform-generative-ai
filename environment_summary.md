# Environment Summary

This document provides a comprehensive summary of the development environment as of the last update. It is intended to be a reference to avoid re-verifying configurations.

---

### 1. System Information

- **Operating System:** Darwin (macOS)
- **Architecture:** arm64 (Apple Silicon)
- **Kernel Version:** `Darwin Craigs-MacBook-Air.local 24.6.0 Darwin Kernel Version 24.6.0: Mon Jul 14 11:30:34 PDT 2025; root:xnu-11417.140.69~1/RELEASE_ARM64_T8103 arm64`

---

### 2. Python & Conda Environment

- **Conda Version:** 25.7.0
- **Active Conda Environment:** `ad3gem`
- **Environment Location:** `/Users/craigcharity/miniconda3/envs/ad3gem`
- **Python Version:** 3.12.11

---

### 3. Google Cloud Configuration

- **Project ID:** `ad3-sam`
- **Authentication:** Service Account Credentials (`GOOGLE_APPLICATION_CREDENTIALS` is set to `/Users/craigcharity/ad3sam/credentials/ad3pulse-service.json`)
- **Active gcloud Account:** `ad3pulse@ad3-sam.iam.gserviceaccount.com`
- **Firestore Database:** `ad3sam-database`
- **Region:** `us-central1`

---

### 4. Installed Python Packages

*This is a snapshot of the packages installed in the `ad3gem` environment. This list may change as dependencies are updated.*

| Package                       | Version        |
| ----------------------------- | -------------- |
| agent-starter-pack            | 0.14.0         |
| annotated-types               | 0.7.0          |
| anyio                         | 4.10.0         |
| arrow                         | 1.3.0          |
| attrs                         | 25.3.0         |
| backoff                       | 2.2.1          |
| binaryornot                   | 0.4.4          |
| blinker                       | 1.9.0          |
| cachetools                    | 5.5.2          |
| certifi                       | 2025.8.3       |
| chardet                       | 5.2.0          |
| charset-normalizer            | 3.4.3          |
| choreographer                 | 1.0.10         |
| click                         | 8.2.1          |
| cookiecutter                  | 2.6.0          |
| docstring_parser              | 0.17.0         |
| flasgger                      | 0.9.7.1        |
| Flask                         | 3.1.2          |
| flask-sock                    | 0.7.0          |
| google-api-core               | 2.25.1         |
| google-auth                   | 2.40.3         |
| google-cloud-aiplatform       | 1.111.0        |
| google-cloud-bigquery         | 3.36.0         |
| google-cloud-core             | 2.4.3          |
| google-cloud-resource-manager | 1.14.2         |
| google-cloud-storage          | 2.19.0         |
| google-crc32c                 | 1.7.1          |
| google-genai                  | 1.33.0         |
| google-resumable-media        | 2.7.2          |
| googleapis-common-protos      | 1.70.0         |
| grpc-google-iam-v1            | 0.14.2         |
| grpcio                        | 1.74.0         |
| grpcio-status                 | 1.74.0         |
| h11                           | 0.16.0         |
| httpcore                      | 1.0.9          |
| httpx                         | 0.28.1         |
| idna                          | 3.10           |
| itsdangerous                  | 2.2.0          |
| Jinja2                        | 3.1.6          |
| jsonschema                    | 4.25.1         |
| jsonschema-specifications     | 2025.4.1       |
| kaleido                       | 1.0.0          |
| logistro                      | 1.1.0          |
| markdown-it-py                | 4.0.0          |
| MarkupSafe                    | 3.0.2          |
| mdurl                         | 0.1.2          |
| mistune                       | 3.1.4          |
| narwhals                      | 2.3.0          |
| numpy                         | 2.3.2          |
| orjson                        | 3.11.3         |
| packaging                     | 25.0           |
| pandas                        | 2.3.2          |
| pip                           | 25.2           |
| plotly                        | 6.3.0          |
| proto-plus                    | 1.26.1         |
| protobuf                      | 6.32.0         |
| pyasn1                        | 0.6.1          |
| pyasn1_modules                | 0.4.2          |
| pydantic                      | 2.11.7         |
| pydantic_core                 | 2.33.2         |
| Pygments                      | 2.19.2         |
| python-dateutil               | 2.9.0.post0    |
| python-slugify                | 8.0.4          |
| pytz                          | 2025.2         |
| PyYAML                        | 6.0.2          |
| referencing                   | 0.36.2         |
| requests                      | 2.32.5         |
| rich                          | 14.1.0         |
| rpds-py                       | 0.27.1         |
| rsa                           | 4.9.1          |
| setuptools                    | 80.9.1         |
| shapely                       | 2.1.1          |
| simple-websocket              | 1.1.0          |
| simplejson                    | 3.20.1         |
| six                           | 1.17.0         |
| sniffio                       | 1.3.1          |
| SQLAlchemy                    | 2.0.43         |
| sqlparse                      | 0.5.3          |
| tabulate                      | 0.9.0          |
| tenacity                      | 9.1.2          |
| text-unidecode                | 1.3            |
| types-python-dateutil         | 2.9.0.20250822 |
| typing_extensions             | 4.15.0         |
| typing-inspection             | 0.4.1          |
| tzdata                        | 2025.2         |
| urllib3                       | 2.5.0          |
| vanna                         | 0.7.9          |
| websockets                    | 15.0.1         |
| Werkzeug                      | 3.1.3          |
| wheel                         | 0.45.1         |
| wsproto                       | 1.2.0          |
