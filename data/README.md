# Dữ Liệu Mock: Vietnam Trip Planner Agent

Thư mục này chứa bộ dữ liệu CSV dùng cho domain của Lab 3:

```text
Vietnam Trip Planner Agent
```

Bộ dữ liệu được thiết kế nhỏ, dễ đọc, dễ sửa và có kết quả cố định. Mục tiêu là dùng cùng một bộ câu hỏi để so sánh:

```text
1. Chatbot baseline: trả lời trực tiếp, không dùng tool.
2. ReAct Agent: suy nghĩ, gọi tool, đọc observation, rồi trả lời.
```

## Vai Trò Từng File

- `city_aliases.csv`: map tên thành phố người dùng hay gọi, ví dụ "Hà Nội", "Sài Gòn", "Đà Nẵng", sang mã sân bay và tên thành phố chuẩn.
- `trips.csv`: dữ liệu chuyến bay giả lập, gồm nơi đi, nơi đến, ngày bay, hãng bay, giá vé, số ghế còn lại và phí hành lý.
- `weather.csv`: dữ liệu thời tiết theo thành phố và ngày.
- `hotels.csv`: dữ liệu khách sạn, giá mỗi đêm, số phòng còn lại và vị trí gần trung tâm.
- `activities.csv`: dữ liệu hoạt động du lịch theo thành phố, loại hoạt động, giá và điều kiện thời tiết phù hợp.
- `baggage_rules.csv`: quy định hành lý theo từng hãng bay.
- `local_transport.csv`: ước tính chi phí di chuyển từ sân bay về trung tâm thành phố.
- `test_scenarios.csv`: danh sách câu hỏi dùng để chạy evaluation cho chatbot và agent.
- `eval_ground_truth.csv`: đáp án chuẩn dạng checklist, dùng để chấm chatbot và agent đúng/sai.

## Test Scenarios Và Ground Truth

Hai file này đi theo cặp:

```text
test_scenarios.csv      = câu hỏi để đưa vào chatbot/agent
eval_ground_truth.csv   = tiêu chí chấm cho từng câu hỏi
```

Ví dụ:

```text
T02 trong test_scenarios.csv:
Tìm chuyến HAN -> SGN ngày 2026-06-10 dưới 1 triệu.

T02 trong eval_ground_truth.csv:
Không có chuyến nào dưới 1 triệu. Hệ thống phải báo không có kết quả phù hợp,
không được bịa ra vé 900.000 VND.
```

Ground truth không bắt câu trả lời phải giống từng chữ. Nó chỉ kiểm tra các ý quan trọng:

- Có dùng đúng dữ liệu trong CSV không.
- Có gọi đúng tool cần thiết không.
- Có xử lý đúng case không có dữ liệu không.
- Có tránh bịa giá vé, thời tiết, khách sạn hay route không.

## Các Edge Case Có Chủ Ý

Những case này được cố tình tạo ra để phục vụ phần failure analysis và cải tiến Agent v1 -> Agent v2:

- Không có chuyến `HAN -> SGN` dưới `1000000` VND vào ngày `2026-06-10`.
- Không có route `HAN -> HUI` trong `trips.csv`.
- Chuyến rẻ nhất `HAN -> PQC` ngày `2026-06-12` chỉ còn 1 ghế, trong khi một số test yêu cầu 2 người.
- Test case `T11` thiếu ngày đi; agent phải hỏi lại thay vì tự đoán ngày.
- Test case `T12` không có dữ liệu thời tiết cho city/date được hỏi; agent không được tự đoán thời tiết.
- Test case `T21` và `T22` yêu cầu gọi nhiều tool để lập kế hoạch đầy đủ.

## Phân Bố Test Case

- Tra cứu đơn giản: `T01`, `T06`, `T10`, `T13`
- Case ngân sách / thiếu dữ liệu: `T02`, `T08`, `T12`, `T14`, `T19`
- Lập kế hoạch nhiều bước: `T03`, `T15`, `T16`
- Full multi-tool planning: `T21`, `T22`
- Tính chi phí: `T04`, `T17`
- Xử lý ràng buộc / thiếu thông tin: `T05`, `T11`, `T20`
- Gợi ý hoạt động và khách sạn: `T07`, `T09`, `T18`

## Cách Dùng Trong Bài Lab

Quy trình đề xuất:

```text
1. Chạy từng câu hỏi trong test_scenarios.csv bằng chatbot baseline.
2. Chạy cùng câu hỏi đó bằng ReAct Agent.
3. Dùng eval_ground_truth.csv để chấm cả hai hệ thống.
4. Ghi lại kết quả: pass/fail, latency, token, loop count, error code.
5. Chọn ít nhất 1 trace thành công và 1 trace lỗi để đưa vào group report.
6. Dùng trace lỗi để cải tiến Agent v1 thành Agent v2.
```

## Ví Dụ Case Thành Công

```text
T03:
Tôi muốn đi từ Hà Nội vào TP.HCM ngày 2026-06-10 cho 2 người dưới 2 triệu mỗi vé.
Nếu trời mưa thì nên mang gì?
```

Agent nên gọi:

```text
search_trips -> get_weather -> recommend_outfit
```

Kết quả đúng cần dựa trên:

- Chuyến `HAN -> SGN` dưới 2 triệu có `VJ121` và `VN207`.
- Thời tiết TP.HCM ngày `2026-06-10`: nhiệt độ cao 32C, xác suất mưa 70%.
- Gợi ý đồ phù hợp mưa: ô gấp, áo mưa, giày dễ khô, áo mỏng.

## Ví Dụ Case Lỗi

```text
T02:
Tìm chuyến HAN -> SGN ngày 2026-06-10 dưới 1 triệu.
```

Trong data không có chuyến nào thỏa điều kiện. Agent v1 có thể bị lỗi nếu vẫn trả lời như có vé 900.000 VND. Agent v2 cần sửa để:

- Báo không có chuyến phù hợp.
- Không bịa dữ liệu.
- Có thể gợi ý tăng ngân sách hoặc chọn chuyến rẻ nhất hiện có là `VJ121` giá `1750000`.

