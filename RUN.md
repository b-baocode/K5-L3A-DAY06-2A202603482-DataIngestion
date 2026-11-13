# Cách chạy bài làm — Ngô Gia Bảo (2A202603482)

Bài làm của Ngô Gia Bảo (2A202603482). Báo cáo: [report/individual/2A202603482_NgoGiaBao.md](report/individual/2A202603482_NgoGiaBao.md).

- **Corpus:** `starter/data/openstax-intro-intellectual-property.pdf`. Đây là chương 4–5 của OpenStax *Introduction to Intellectual Property*, giấy phép CC BY 4.0. Nguồn, mã SHA-256 và cách trích trang ghi trong [data/manifest.json](data/manifest.json).
- **Benchmark:** [benchmark/questions.json](benchmark/questions.json) (20 câu, 15 dev / 5 test).
- **Cấu hình:** [starter/config/strategies.json](starter/config/strategies.json).

Chạy từ thư mục `starter/`, trên Windows PowerShell:

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-lab.txt   # file riêng của bài làm; requirements.txt của starter giữ nguyên
Copy-Item .env.example .env            # DATABASE_URL đã có sẵn cho Docker local
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py          # mong đợi: Setup OK
python scripts/verify_benchmark.py     # kiểm chứng evidence trên trang gold, mong đợi: errors = []
python -m src.ingest --reset           # nạp A và B vào pgvector (~7 phút CPU, lần đầu tải model ~1,5 GB)
python -m src.ingest                   # nạp lại: mọi chunk 'unchanged', không nhân đôi
python -m src.run --split dev          # chạy dev, ghi results/selection.json
python -m src.run --split test         # chỉ chạy được sau khi đã có selection.json
python scripts/fill_audit_labels.py    # điền nhãn audit (AI đề xuất + kiểm tra metadata tự động)
```

Model embedding là `google/embeddinggemma-2`, chạy local, không cần API key, chi phí 0 USD.

**Lưu ý cho người chấm:** thư mục `results/` mà SUBMISSION.md yêu cầu nằm ở **`starter/results/`**, theo cấu trúc của starter. Các file trong đó:
- Thống kê nạp: `ingest_stats.json`, `ingest_runs.jsonl`, `ingest.log`.
- Kết quả truy xuất: `traces_dev.jsonl`, `traces_test.jsonl`, `metrics_dev.json`, `metrics_test.json`, `selection.json`.
- Kiểm chứng benchmark: `benchmark_verification.json`.
- Nhãn chấm tay: `audit_coherence.csv`, `audit_grounding.csv`, `audit_metadata.csv`.
