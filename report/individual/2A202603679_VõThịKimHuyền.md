# Báo cáo cá nhân — Day 6

> **Đổi tên file** thành `<MSSV>_<HoTen>.md` trước khi nộp. Điền `STUDENT.md`.

## Dữ liệu và pipeline

- **Chủ đề / corpus:** OpenStax *College Success* (Amy Baldwin, 2020), trang PDF **25–100**, **75** trang nội dung (ngưỡng ≥400 ký tự/trang). Manifest: `data/manifest.json`.
- **Nguồn PDF:** CloudFront mirror (`starter/data/college-success.pdf`). License: CC BY 4.0 trong PDF; attribution *Access for free at openstax.org*.
- **Chuẩn hóa:** Unicode NFKC; bỏ trang nội dung thấp (vd. trang 62). Metadata **chapter** parse từ heading OpenStax + forward-fill.
- **Embedding (kết quả chính thức):** Google **Gemini API**, `gemini-embedding-001`, **768** chiều (MRL), `task_type` RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY, cosine similarity.
- **Vector store:** PostgreSQL 16 + **pgvector 0.8.7** (`127.0.0.1:55433`), bảng `chunks`, metadata `document_id`, `source`, `page`, `chapter`.
- **Lệnh:** `cd starter` → `docker compose -p day6_student_starter up -d --wait` → `python -m src.run` (một tiến trình, `EMBEDDING_PROVIDER=gemini`).
- **Quota / dry-run:** Lần đầu hết quota free tier ~1.000 embed/ngày (project cũ). Đã chạy thử **local** để kiểm pipeline; **số liệu nộp** lấy từ lần chạy **Gemini** đầy đủ với API key project mới (xem `results/gemini_quota_note.md`).
- **Chi phí:** free tier; ~**791** lần gọi embed mới (451 + 340 unique) + 20 câu × 3 strategy query embed. Thời gian embed A ~512s, B ~383s.

## Cấu hình và benchmark

| Strategy | Chunk | Filter |
|---|---|---|
| A | fixed 500 ký tự, overlap 50 | không |
| B | by_paragraph, max 500 | không |
| C | giống B | `chapter` khi câu có `filter` |

- **Benchmark:** 20 câu (`benchmark/questions.json`), **15 dev / 5 test**; 4 câu có metadata filter (3 dev, 1 test — q19).
- **Tiêu chí chọn (trước khi xem test):** ưu tiên **Evidence Hit@3 trên dev**, rồi coherence/grounding, rồi tài nguyên.
- **Chốt trên dev:** **Strategy A** — Evidence Hit@3 = **14/15** (cao hơn B/C **12/15**).

## Kết quả retrieval

| Strategy | Split | Hit@1 | Evidence Hit@3 | Chunks | Ingest (s) |
|---|---|---|---|---:|---:|
| A | dev | 11/15 | **14/15** | 451 | 513 |
| A | test | 3/5 | 4/5 | | |
| B | dev | 9/15 | 12/15 | 415 | 384 |
| B | test | 3/5 | 4/5 | | |
| C | dev | 9/15 | 12/15 | 415 | 0.9* |
| C | test | 3/5 | 4/5 | | |

\*C tái dùng chunk + vector của B; không gọi embed thêm.

**Tổng thể (20 câu):** A Hit@1 14/20, Hit@3 18/20; B và C Hit@1 12/20, Hit@3 16/20.

Trace: `results/traces_A.json`, `traces_B.json`, `traces_C.json` · Metrics: `results/metrics.json` · Log: `artifacts/experiment_log.csv`.

## Audit metadata, coherence, grounding

| Chỉ số | Giá trị |
|---|---|
| Metadata đủ & đúng (`source`, `page`, `chapter` nếu cần) trên **5 câu test × top-3 × 3 strategy** | **26/45** (`results/audit_metadata.json`, `python scripts/generate_audit.py`) |
| Filter: top-3 tuân thủ `chapter` (câu test có filter) | **3/3** |
| Coherence (10 vị trí nguồn, thang 0–2, tay) | A **1.6**, B **2.0**, C **2.0** — `results/manual_labels.json` |
| Grounding test (5 câu × top-3, thang 0–2, tay) | A **1.5**, B/C **1.7** |

**Giải thích metadata 26/45:** Kiểm tra yêu cầu **trúng `gold_page`**; chunk có evidence đúng nghĩa vẫn có thể ở trang lệch (đặc biệt chunk cố định A), nên thấp hơn Hit@3.

## Phân tích và kết luận

- **A–B (chunking):** A nhiều chunk hơn (451 vs 415). **Gemini** cho dev Hit@3 **cao hơn với A (14/15)** so với B (12/15) — khác với thử local trước đó; embedding semantic mạnh hơn giúp fixed-window vẫn truy được evidence ngắn. Trade-off: coherence trung bình A thấp hơn (1.6 vs 2.0) vì cắt giữa bảng/đoạn.
- **B–C (metadata filter):** Hit@1/Hit@3 **giống B** trên 20 câu (filter chỉ áp dụng 4 câu). Trên **3 câu filter test**, audit **3/3** top-3 thuộc đúng `chapter` — filter hoạt động khi câu hỏi cung cấp filter.
- **Ví dụ lỗi / học được:** q17 — top-1 có thể là trang outline chương 2 (43) thay vì gold 44; metadata audit ghi `ok: false` dù nội dung liên quan. Chi tiết: `manual_labels.failure_examples`.
- **Chiến lược chọn (dev):** **A** theo Evidence Hit@3; trên test cả A/B/C đều 4/5 Hit@3 — không đổi quyết định sau test.
- **Hạn chế:** Benchmark nhỏ (5 test); audit metadata nghiêm `gold_page`; quota API; thời gian embed; coherence/grounding do tác giả chấm (nguy cơ thiên lệch).
- **Công cụ AI:** Dùng Cursor/AI hỗ trợ code và pipeline; học viên chạy lại, kiểm tra benchmark (`verify_benchmark.py`) và trace thực tế.
