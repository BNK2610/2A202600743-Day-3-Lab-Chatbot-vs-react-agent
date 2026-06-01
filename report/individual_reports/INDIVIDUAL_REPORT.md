# Báo Cáo Cá Nhân: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Bùi Ngọc Khánh
- **Student ID**: 2A202600743
- **Date**: 2026-06-01
- **Nhóm**: Vietnam Trip Planner Agent

---

## I. Đóng Góp Kỹ Thuật

Phần đóng góp chính của tôi là cụm **hotel/transport/cost** và phần evaluation tổng hợp. Tôi phụ trách một phần dữ liệu, một phần tool/module, một nhóm test case và phân tích kết quả cuối.

### 1. Module / file phụ trách

| Nhóm việc | File / module | Đóng góp cụ thể |
| :--- | :--- | :--- |
| Data | `data/hotels.csv` | Chuẩn bị dữ liệu khách sạn, giá/đêm, số phòng, vị trí |
| Data | `data/local_transport.csv` | Chuẩn bị chi phí taxi/bus/ride-hailing từ sân bay về trung tâm |
| Data/eval | `data/eval_ground_truth.csv` | Chuẩn hóa required tools, expected facts, expected behavior |
| Tool/module | `search_hotels` trong `src/tools/travel_tools.py` | Thiết kế tool tìm khách sạn theo city/budget/rooms |
| Tool/module | `estimate_local_transport` trong `src/tools/travel_tools.py` | Thiết kế tool ước tính chi phí di chuyển nội đô |
| Tool/module | `calculate_total_cost` trong `src/tools/travel_tools.py` | Thiết kế tool tính tổng chi phí cơ bản từ các thành phần |
| Evaluation | `run_eval.py` | Chạy chatbot và agent, xuất CSV/JSON kết quả |
| Telemetry | `src/telemetry/logger.py`, `src/telemetry/metrics.py` | Tổng hợp log, latency, token, loop count, cost estimate |
| Test/eval | T17-T22 trong `data/test_scenarios.csv` | Thiết kế case cost, hotel no-match, missing info và full multi-tool planning |

### 2. Thiết kế test case

Bộ test gồm 22 case, trong đó tôi trực tiếp phụ trách nhóm T17-T22 và phối hợp để thống nhất toàn bộ ground truth. Các nhóm case gồm:

- Tra cứu chuyến bay đơn giản: T01, T13
- Không có dữ liệu hoặc không thỏa ngân sách: T02, T08, T12, T14, T19
- Thiếu thông tin bắt buộc: T11, T20
- Multi-step weather/outfit/activity: T03, T07, T15, T16
- Tính tổng chi phí: T04, T17
- Full multi-tool planning: T21, T22

Mục tiêu là không chỉ kiểm tra final answer, mà còn kiểm tra trace: agent có gọi đúng tool không, có bịa dữ liệu không, có xử lý failure status không.

### 3. Kết quả evaluation chính

Nguồn: `eval_results_20260601_222128.json`

| Hệ thống | Pass | Review | Fail | Tỉ lệ pass |
| :--- | ---: | ---: | ---: | ---: |
| Chatbot baseline | 9 | 11 | 2 | 40.9% |
| Agent v2 | 15 | 7 | 0 | 68.2% |

Agent v2 không có fail cứng. Các case review chủ yếu do thiếu tool theo ground truth, không phải do answer hoàn toàn sai.

---

## II. Debugging Case Study

### Case: T21 bị `review` vì thiếu `calculate_total_cost`

**Input**:

```text
Tôi muốn đi từ Hà Nội vào TP.HCM ngày 2026-06-10 cho 2 người, vé dưới 2 triệu mỗi người, khách sạn dưới 1 triệu một đêm, nếu trời mưa thì gợi ý hoạt động phù hợp, tính taxi từ sân bay về trung tâm và tính tổng chi phí cơ bản.
```

**Kết quả evaluation**:

```text
agent_verdict = review
agent_score_notes = missing tools: calculate_total_cost; missing expected number 2000000
agent_tools_called = search_trips|get_weather|search_hotels|recommend_activities|estimate_local_transport
agent_loop_count = 8
```

### Chẩn đoán

Agent đã gọi hầu hết các tool cần thiết:

- `search_trips`: lấy chuyến bay HAN -> SGN.
- `get_weather`: lấy thời tiết TP.HCM.
- `search_hotels`: lấy khách sạn dưới 1 triệu.
- `recommend_activities`: lấy hoạt động phù hợp khi mưa.
- `estimate_local_transport`: lấy taxi sân bay về trung tâm.

Nhưng agent không gọi `calculate_total_cost`. Nó tự cộng chi phí trong Final Answer. Về mặt người dùng, câu trả lời khá hữu ích; nhưng về mặt eval-by-trace, trace chưa đạt vì thiếu tool tính tổng.

### Root cause

- Prompt nói `calculate_total_cost` dùng các giá đã lấy từ Observation, nhưng chưa bắt buộc đủ mạnh.
- Task T21 dài, agent chạm `max_steps=8`.
- LLM có xu hướng tự tính phép cộng đơn giản thay vì gọi tool.

### Bài học

Final answer đúng chưa đủ. Với agent, trace mới là bằng chứng hệ thống có hành động đúng hay không. Nếu requirement nói phải dùng tool tính tổng, tự cộng trong câu trả lời vẫn bị xem là thiếu grounding theo trace.

---

## III. Nhận Xét Cá Nhân: Chatbot vs ReAct Agent

Qua số liệu, chatbot baseline có latency thấp hơn nhưng độ tin cậy thấp hơn. Chatbot trả lời nhanh vì chỉ cần một LLM call, nhưng khi hỏi dữ liệu cụ thể nó thường trả lời chung chung hoặc yêu cầu người dùng tự kiểm tra.

Agent v2 tốn nhiều token và latency hơn, nhưng pass nhiều hơn:

- Chatbot pass 40.9%.
- Agent v2 pass 68.2%.
- Agent v2 không có fail cứng.

Điểm tôi thấy rõ nhất là agent cần được đánh giá bằng trace. Một câu trả lời nghe hợp lý có thể vẫn bị review nếu thiếu tool. Ngược lại, một case có validation error giữa chừng chưa chắc fail nếu agent sửa được và tiếp tục gọi tool đúng.

---

## IV. Hướng Cải Tiến

1. Thêm metric `tool_coverage`: phần trăm required tools đã được gọi.
2. Tách `review` thành hai loại: answer đúng nhưng thiếu trace, và answer sai thật.
3. Thêm evaluator đọc cả `agent_trace_json` để đánh giá grounding tốt hơn.
4. Cho phép `max_steps` động: case full-plan có thể dùng 10-12 steps, case đơn giản giữ 6-8 steps.
5. Tự động sinh bảng báo cáo từ JSON eval để giảm lỗi copy thủ công.

---

## Tự Đánh Giá Đóng Góp

Tôi đóng góp khoảng **33.3%** công việc nhóm, tập trung vào cụm hotel/transport/cost, test case T17-T22, evaluation, telemetry và phân tích kết quả cho báo cáo.
