# Ghi chú lỗi Gemini (2026-11-13)

- **Lỗi:** `429 RESOURCE_EXHAUSTED` — vượt **1.000 request embed/ngày** (free tier) trên model `gemini-embedding-001`.
- **Retry:** API gợi ý chờ vài giờ (~16–20h) trước khi reset quota **theo ngày**.
- **Hết quota ngày ≠ hết quota phút:** khi thấy `per day limit exceeded`, chờ reset — throttle 0.72s/giây cũng không giúp.
- **Đã lưu một phần:** `embed_cache_gemini_partial.json` (~451 vector, strategy A).
- **Pipeline hoàn chỉnh (Gemini):** đã chạy thành công với **API key project mới** — xem `results/metrics.json`, `ingest_stats.json`, `artifacts/experiment_log.csv` (model `gemini-embedding-001`, 768d).
- **Dry-run local (không dùng làm kết quả chính):** `sentence-transformers/all-MiniLM-L6-v2` khi quota Gemini hết — chỉ kiểm pipeline.

## Chạy lại bằng Gemini khi quota reset

```powershell
cd starter
# Xóa index cũ và cache local (cùng dimension 768)
Remove-Item ..\results\embed_cache.json -ErrorAction SilentlyContinue
python -c "import os; from dotenv import load_dotenv; import psycopg; load_dotenv('.env'); c=psycopg.connect(os.environ['DATABASE_URL']); cur=c.cursor(); cur.execute('DROP TABLE IF EXISTS chunks CASCADE'); c.commit()"
$env:EMBEDDING_PROVIDER='gemini'
python -m src.run
```

**Lưu ý:** Chỉ chạy **một** tiến trình `src.run` để tránh đốt quota gấp đôi.
