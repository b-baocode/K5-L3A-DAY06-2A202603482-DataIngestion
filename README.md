# Day 06 · Data Ingestion Engineering

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.

---

## Bài làm: cách setup và chạy

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
