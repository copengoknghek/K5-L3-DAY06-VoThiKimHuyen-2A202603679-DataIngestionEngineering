# Chạy pipeline Day 06 (repo này)

## Yêu cầu

- Docker Desktop (PostgreSQL + pgvector)
- Python 3.11+
- `starter/.env`: `DATABASE_URL`, `GEMINI_API_KEY` (nếu dùng Gemini)

## Setup

```powershell
cd starter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # điền GEMINI_API_KEY
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

## Tải corpus (nếu chưa có PDF)

```powershell
cd starter/data
curl.exe -L -o college-success.pdf "https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/College_Success_web.pdf"
```

## Kiểm tra benchmark

```powershell
python scripts/verify_benchmark.py
```

## Chạy ingest + benchmark

```powershell
$env:EMBEDDING_PROVIDER='gemini'   # hoặc local nếu hết quota
python -m src.run
```

**Kết quả:** `results/metrics.json`, `traces_A|B|C.json`, `artifacts/experiment_log.csv`

## Audit

```powershell
python scripts/generate_audit.py
```

Nhãn coherence/grounding: `results/manual_labels.json` (chỉnh tay nếu cần).
