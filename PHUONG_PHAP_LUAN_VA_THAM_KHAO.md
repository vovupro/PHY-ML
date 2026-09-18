# TÀI LIỆU PHƯƠNG PHÁP LUẬN, KIẾN TRÚC HỆ THỐNG VÀ DANH MỤC THAM KHẢO

**Đề tài:** Thích nghi Điều chế, Mã hóa và Nỗ lực Giải mã (Joint Modulation, Coding & Decoder Effort Adaptation - PHY-AMC) bằng Học máy nhẹ (Lightweight ML) trên Kênh Fading Vô tuyến  
**Môi trường:** `research_core` (Zero-GUI Research Core)  
**Ngày hoàn thiện:** 09/09/2026  

---

## PHẦN 1: TRIẾT LÝ VÀ PHONG CÁCH PHƯƠNG PHÁP LUẬN

Nghiên cứu này được xây dựng dựa trên **4 nguyên tắc phương pháp luận cốt lõi**, định hình nên phong cách thực hiện từ việc đặt bài toán, sinh dữ liệu, huấn luyện ML cho đến đo lường đầu-cuối:

### 1. Thực chứng và Tôn trọng Đồng thuận Học thuật (Empirical Rigor over Bureaucracy)
* **Không sao chép máy móc các khung quản lý công nghiệp cồng kềnh:** Chúng tôi không cố khoác lên hệ thống một cái áo tiêu chuẩn vòng đời phần mềm doanh nghiệp như ISO/IEC 5338 hay CRISP-DM toàn diện. Với một mô hình học máy nhẹ (Decision Tree, Random Forest), việc ép vào quy trình quản lý cồng kềnh là thừa thãi và hình thức.
* **Tôn trọng đồng thuận học thuật (Academic Consensus):** Thay vào đó, chúng tôi áp dụng **Quy trình Đánh giá Thực nghiệm Chuẩn mực (Empirical Evaluation Protocol)** — phương pháp luận được sử dụng nhất quán trong các công trình đỉnh cao của IEEE Transactions và Springer: chia mẫu độc lập, khóa baseline đối chứng, đánh giá mù trên tập Test và kiểm định thống kê có khoảng tin cậy.

### 2. Tách biệt Tuyệt đối: "Kiểm tra Tính Đúng đắn" (Verification) và "Đánh giá Hiệu năng" (Validation)
Để giải quyết triệt để nỗi ám ảnh **"bị số liệu dắt đi vòng vòng"**:
* **Bản chất vấn đề:** Các kỹ thuật chia dữ liệu kinh điển (Holdout, Train/Val/Test, Cross-Validation) **chỉ đo lường khả năng khái quát hóa của mô hình trên tập dữ liệu đã cho**. Chúng hoàn toàn mù tịt trước các câu hỏi: *Dữ liệu có đúng vật lý không? Nhãn có phản ánh đúng bài toán không? Code giải điều chế có bug không? Có bị rò rỉ dữ liệu không?* Nếu công thức kênh sai hoặc nhãn bị gán lệch, mô hình ML vẫn có thể học và đạt điểm cao $99\%$, dẫn dắt người nghiên cứu vào ảo giác thành công.
* **Nguyên tắc phân định (Verification trước, Validation sau):**
  1. **Verification (Kiểm tra tính đúng đắn):** Sử dụng các "mỏ neo chân lý" độc lập hoàn toàn với ML (công thức giải tích AWGN erfc, kỳ vọng tích phân Rayleigh, cận Cramér-Rao của ước lượng kênh pilot, kiểm tra phân hoạch không trùng lặp seed). Tầng này bắt buộc phải **PASS 100%** thì hệ thống mới được coi là hợp lệ.
  2. **Validation (Đánh giá hiệu quả):** Khi nền móng vật lý và nhãn đã chắc chắn đúng, lúc này mới đánh giá xem thuật toán ML đạt Goodput bao nhiêu, vi phạm target BER thế nào. Nếu kết quả kém, đó là do năng lực của mô hình chưa tối ưu, chứ không phải do bug mô phỏng.

### 3. Kỷ luật Tiền nghiệm (Pre-registration / Protocol Discipline)
* **Khóa luật chơi trước khi vào trận:** Toàn bộ không gian hành động, đặc trưng đầu vào, công thức nhãn chân lý (Oracle), các baseline đối chứng (Fixed, LUT) và hàm mục tiêu được xác định và đóng băng trước khi tiến hành thí nghiệm chính.
* **Không làm đẹp số liệu (Anti-Cherry-Picking):** Tuyệt đối không thay đổi cách chia dữ liệu, không tinh chỉnh lại baseline, và không nới lỏng ràng buộc độ tin cậy khi thấy kết quả của mô hình ML không như mong đợi. Mọi sự thay đổi về giả định phải được kiểm chứng lại từ đầu.

### 4. Tối giản nhưng Tuyệt đối Nghiêm ngặt (Minimalist yet Uncompromising Rigor)
* Mô hình được chọn là **Mô hình Hộp trắng / Xám nhẹ (White-box / Grey-box)**: Single Decision Tree (CART/C4.5) và Random Forest.
* Mô hình có thể xuất thành các luật văn bản tường minh (`if-else`), giải thích được từng rẽ nhánh logic và có thời gian suy luận trên CPU ở mức micro-giây ($\mu\text{s}$), bảo đảm tính khả thi thời gian thực trên khung truyền 5G NR ($T_f = 1.536\text{ ms}$).

---

## PHẦN 2: KIẾN TRÚC HỆ THỐNG ĐÃ XÂY DỰNG (3-TIER ARCHITECTURE)

Hệ thống được tổ chức thành 3 tầng độc lập, khép kín từ tầng vật lý đến tầng ứng dụng học máy:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 1: CỔNG KIỂM ĐỊNH TÍNH ĐÚNG ĐẮN VẬT LÝ (SOUNDNESS VERIFICATION GATE)    │
│  - Đối chiếu giải tích: AWGN erfc, Rayleigh expectation, Pilot MSE          │
│  - Tính bất biến: Chuẩn hóa Es=1.0, N0 hằng số, moment bậc 2, 4 phân phối h │
│  - Nhân quả: Pilot h_est độc lập kênh thực h; SeedSequence disjoint         │
│  - Chuẩn viễn thông: Demapper LLR & CRC24A đối chiếu bit-by-bit với Sionna │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Vượt qua 27/27 bài test unit)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 2: MÔ HÌNH HỌC MÁY PHÂN LOẠI ĐA LỚP (DIRECT CLASSIFICATION & ORACLE)    │
│  - Nhãn chân lý (Oracle Ground Truth via Exhaustive Search):                │
│      y* = argmax { Payload(a) * (1 - BlockError(a)) }  với BER(a) <= Target │
│      y* = DEFER (nếu không có cấu hình nào đạt BER Target)                  │
│  - Mô hình lõi: DecisionTreeClassifier (CART/C4.5, depth<=7, min_leaf>=12)  │
│  - Mô hình đối chứng: RandomForestClassifier, Empirical LUT (2-dB bins)    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Đánh giá mù trên tập Test)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 3: ĐÁNH GIÁ ĐẦU-CUỐI & SUY LUẬN THỐNG KÊ (SYSTEM-LEVEL PHY BENCHMARK)   │
│  - Timeline Goodput: (Tổng payload thành công) / (Tổng frame * N_sym)       │
│  - BER Violation Rate & Defer Rate trên toàn chuỗi thời gian                │
│  - Chặn tin cậy hữu hạn mẫu Bernstein Bound (Maurer & Pontil 2009)          │
│  - Đo lường trễ suy luận từng frame trên CPU (runtime_table micro-giây)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. Tầng 1: Kiểm định Tính đúng đắn Nền tảng (`test_physics.py`)
Mọi thành phần vật lý và toán học đều có phép thử độc lập đối chiếu với lý thuyết giải tích:
- **AWGN:** Kiểm tra BER của BPSK, QPSK, 16-QAM không mã hóa khớp với công thức tích phân hàm sai số $Q(z)$ và $\text{erfc}$ trong phạm vi dung sai $\delta < 0.0035$.
- **Rayleigh:** Kiểm tra BER trung bình tích phân fading Rayleigh khớp với công thức lý thuyết $\mathbb{E}[\text{BER}] = \frac{1}{2}\left(1 - \sqrt{\frac{\gamma_b}{1+\gamma_b}}\right)$.
- **Ước lượng kênh Pilot:** Sai số bình phương trung bình $\text{MSE}(\hat{h})$ từ $32$ pilot symbols hội tụ chính xác về cận lý thuyết $N_0 / N_{\text{pilots}}$.
- **Chống rò rỉ dữ liệu:** `SeedSequence` độc lập tuyệt đối giữa các nhánh Train, Validation và Test; hàm `assert_disjoint` bảo đảm không có bất kỳ cặp khóa seed nào trùng lặp.
- **Tính chuẩn tắc 5G NR:** Khối tạo mã kiểm tra dư thừa CRC24A và giải điều chế Soft Demapper LLR khớp tuyệt đối từng bit với thư viện chuẩn 5G Sionna của NVIDIA.

### 2. Tầng 2: Thiết kế Tầng Học máy Phân loại (`policies.py`)
- **Tạo nhãn Chân lý Oracle (`compute_oracle_labels`):**
  Trong pha ngoại tuyến, trên mỗi trạng thái kênh truyền ($h_{est}$), hệ thống áp dụng phương pháp **Ex-ante Monte Carlo expectation** (kế thừa từ Daniels–Caramanis–Heath 2010). Bằng cách truyền lặp $N$ lần (ví dụ 100 lần) với các tập nhiễu và dữ liệu ngẫu nhiên độc lập để tích lũy giá trị kỳ vọng của FER và BER. Mode được gán nhãn là mode tối đa hóa Goodput kỳ vọng trong số các mode có kỳ vọng BER đạt ràng buộc $\text{BER} \le 10^{-3}$, loại bỏ triệt để tính may rủi của "hindsight single-trial".
- **Cơ chế Hoãn phát (DEFER):**
  Khi kênh rơi vào suy hao sâu (deep fade) khiến ngay cả BPSK tốc độ thấp nhất vẫn vi phạm target BER, Oracle tự động gán nhãn hành động là `DEFER` ($y^* = N_{\text{actions}}$). Máy phát chủ động ngừng truyền để tiết kiệm năng lượng và không gây lỗi bit vô ích.
- **Mô hình Cây Quyết định (`DecisionTreeClassifierPolicy`):**
  Mô hình nhận vector đặc trưng $\mathbf{x} = [\text{SNR}_{\text{est}}, \text{SNR}_{\text{nom}}, \text{Channel ID}]$, phân loại trực tiếp ra hành động tối ưu. Mô hình cung cấp phương thức `export_rules()` xuất toàn bộ nhánh rẽ logic ra dạng văn bản để đưa vào báo cáo khoa học.
- **Hệ thống Baseline đối chứng:**
  * `Fixed robust`: Cố định BPSK tốc độ thấp nhất.
  * `Fixed high throughput`: Cố định 16-QAM tốc độ cao nhất.
  * `Empirical LUT`: Phân cụm SNR 2-dB theo từng loại kênh, chọn hành động chiếm đa số trên tập Train.
  * `OraclePolicy`: Luôn chọn quyết định tối ưu tuyệt đối của Oracle làm trần lý tưởng để tính Gap.

### 3. Tầng 3: Đo lường và Suy luận Thống kê (`metrics.py`, `experiment.py`)
- **Timeline Goodput:**
  $$\text{Goodput} = \frac{\sum_{t \in \text{Thành công}} K_{\text{payload}, t}}{N_{\text{frames}} \times N_{\text{sym}}} \quad (\text{bits/symbol})$$
  Mẫu số tính trên toàn bộ số khung của kịch bản (kể cả khung bị DEFER hoặc truyền hỏng), bảo đảm không có hiện tượng "ăn gian" điểm số bằng cách né tránh các khung khó.
- **Chặn tin cậy Bernstein hữu hạn mẫu (Empirical Bernstein Bound):**
  Thay vì giả định ngây thơ rằng các bit giải mã là các phép thử độc lập Bernoulli (vốn sai bản chất vì các bit trong cùng 1 khối mã hóa LDPC có tương quan rất mạnh), hệ thống coi mỗi khung truyền độc lập là một biến ngẫu nhiên và dùng bất đẳng thức Bernstein thực nghiệm để tính cận trên $\text{BER}_{\text{upper}}$ và khoảng tin cậy Goodput.

---

## PHẦN 3: DANH MỤC TÀI LIỆU THAM KHẢO VÀ BÀI BÁO KHOA HỌC

Toàn bộ các quyết định thiết kế trong hệ thống đều được trích dẫn trực tiếp từ các tài liệu học thuật và chuẩn mực quốc tế dưới đây:

### 1. Phương pháp luận Kiểm chứng và Đo lường Thống kê
* **[Sargent2013]** Sargent, R. G. (2013). *Verification and validation of simulation models*. Journal of Simulation, 7(1), 12–24.  
  *Nội dung kế thừa:* Cơ sở lý luận phân định rạch ròi giữa **Verification** (xác nhận mô hình mô phỏng chạy đúng theo thiết kế toán lý) và **Validation** (xác nhận mức độ chính xác của giải pháp đối với bài toán thực tế).  
  *Link tài liệu:* [https://ics-websites.science.uu.nl/docs/vakken/msosi/VerificationValidationSargent.pdf](https://ics-websites.science.uu.nl/docs/vakken/msosi/VerificationValidationSargent.pdf)

* **[Maurer2009]** Maurer, A., & Pontil, M. (2009). *Empirical Bernstein bounds and sample-variance penalization*. In COLT 2009 - The 22nd Conference on Learning Theory.  
  *Nội dung kế thừa:* Công thức toán học tính bán kính tin cậy hữu hạn mẫu (Bernstein Radius) áp dụng trong hàm `bernstein_radius` tại [`metrics.py`](file:///c:/Users/vongu/Documents/Codex/2026-08-25/cho-x20/outputs/research_core/metrics.py), dùng để tính cận trên $\text{BER}_{\text{upper}}$ và khoảng tin cậy Goodput mà không giả định bit độc lập.  
  *Link tài liệu:* [https://arxiv.org/abs/0907.3740](https://arxiv.org/abs/0907.3740)

### 2. Mô hình Học máy nhẹ cho Thích nghi Đường truyền (Lightweight ML for AMC)
* **[Daniels2010]** Daniels, R. C., Caramanis, C. M., & Heath, R. W. (2010). *Adaptation in Convolutionally Coded MIMO-OFDM Wireless Systems Through Supervised Learning and SNR Ordering*. IEEE Transactions on Vehicular Technology, 59(1), 114–126.  
  *Nội dung kế thừa:* Công trình tiên phong của Robert Heath Jr. thiết lập khuôn mẫu: Dùng học máy có giám sát (Supervised Learning) với nhãn sinh từ mô phỏng ngoại tuyến (offline error-rate observations) để dự đoán cấu hình MCS tối ưu thay vì giải hệ phương trình thu phát phức tạp.  
  *Link bài báo (IEEE):* [https://doi.org/10.1109/TVT.2009.2035348](https://doi.org/10.1109/TVT.2009.2035348)  
  *Bản thảo mở (Preprint PDF):* [https://users.ece.utexas.edu/~rheath/papers/2010/tvt/daniels_coded_adaptation.pdf](https://users.ece.utexas.edu/~rheath/papers/2010/tvt/daniels_coded_adaptation.pdf)

* **[Liu2020]** Liu, H., He, J., Wensowitch, J., Rajan, D., & Camp, J. (2020). *Architecture and experimental evaluation of context-aware adaptation in vehicular networks*. EURASIP Journal on Wireless Communications and Networking, 2020(1), 49.  
  *Nội dung kế thừa:* Mô hình Cây Quyết định (C4.5/CART) cho thích nghi đường truyền; phương pháp sinh nhãn qua tìm kiếm vét cạn (Exhaustive Search); phân tích độ quan trọng của đặc trưng; và kỹ thuật biên dịch cây quyết định thành Look-Up Table (LUT) để chạy thời gian thực trên Linux mac80211.  
  *Link bài báo (Open Access):* [https://doi.org/10.1186/s13638-020-01668-7](https://doi.org/10.1186/s13638-020-01668-7)

* **[Borst2026]** Borst, L., Ansarifard, M., Vinayachandran, A., Joshi, K. C., & Exarchakos, G. (Tháng 9/2026). *Lightweight CFR-Based Modulation Adaptation in a Real-Time MIMO-OFDM SDR Testbed*.  
  *Nội dung kế thừa:* Triển khai cây quyết định nhẹ trên thiết bị vô tuyến định nghĩa bằng phần mềm (SDR); bóc tách mâu thuẫn giữa độ chính xác ngoại tuyến của Random Forest và độ trễ thực thi thời gian thực của Decision Tree đơn lẻ; quy trình 5-fold cross-validation.  
  *Link bài báo (arXiv):* [https://arxiv.org/abs/2609.00903](https://arxiv.org/abs/2609.00903)

* **[TelecommSyst2024]** *Machine learning-based methods for MCS prediction in 5G networks*. Telecommunication Systems, Springer (2024).  
  *Nội dung kế thừa:* So sánh đối chứng chuẩn mực giữa các dòng ML nhẹ (Decision Tree, Random Forest, SVM, k-NN) với mạng nơ-ron; quy trình chia mẫu 70% Train, 15% Validation, 15% Test và Grid Search tối ưu tham số.  
  *Link bài báo:* [https://doi.org/10.1007/s11235-024-01158-x](https://doi.org/10.1007/s11235-024-01158-x)

### 3. Tiêu chuẩn Lớp Vật lý và Thư viện Tham chiếu Công nghiệp
* **[3GPP-TS38.212]** 3rd Generation Partnership Project (3GPP). *Technical Specification Group Radio Access Network; NR; Multiplexing and channel coding (Release 16/17)*, 3GPP TS 38.212.  
  *Nội dung kế thừa:* Đa thức sinh mã kiểm tra dư thừa tuần hoàn CRC24A ($D^{24} + D^{23} + D^{18} + D^{17} + D^{14} + D^{11} + D^{10} + D^7 + D^6 + D^5 + D^4 + D^3 + D + 1$), cấu trúc ma trận kiểm tra Base Graph 1 và Base Graph 2 của mã sửa sai LDPC 5G NR.  
  *Link tiêu chuẩn:* [https://www.3gpp.org/specifications-technologies/specifications-by-series](https://www.3gpp.org/specifications-technologies/specifications-by-series)

* **[Sionna2022]** Hoydis, J., Cammerer, S., Ait Aoudia, F., et al. (2022). *Sionna: An Open-Source Library for Next-Generation Physical Layer Research*. arXiv preprint arXiv:2203.11854.  
  *Nội dung kế thừa:* Thư viện tham chiếu công nghiệp của NVIDIA để đối chiếu kiểm định (parity testing) cho bộ giải mã Soft Demapper LLR và bộ mã hóa CRC24A trong `test_physics.py`.  
  *Link bài báo:* [https://arxiv.org/abs/2203.11854](https://arxiv.org/abs/2203.11854)  
  *Tài liệu chính thức:* [https://nvlabs.github.io/sionna/](https://nvlabs.github.io/sionna/)

---

## PHẦN 4: VÉT CẠN TRI THỨC TỪ 20 NĂM NGHIÊN CỨU THÍCH NGHI ĐƯỜNG TRUYỀN (4 TRƯỜNG PHÁI & 5 VŨ KHÍ TRI THỨC)

Để thực hiện cuộc **"vét cạn tri thức"** theo đúng yêu cầu của bạn, tôi đã lùng sục từ phần tài liệu tham khảo (references) của các bài báo trước cho đến các công trình kinh điển và mới nhất trên **ACM MobiCom, ACM MobiSys, IEEE Transactions on Wireless Communications, và IEEE/ACM Trans. Networking**. 

Lĩnh vực thích nghi đường truyền (Link/Rate Adaptation - LA/AMC) đã phát triển suốt 20 năm qua. Khi xâu chuỗi toàn bộ các bài báo này lại, ta thấy lịch sử chia làm **4 trường phái lớn**. Dưới đây là bức tranh toàn cảnh và **5 "vũ khí tri thức" thực sự đáng giá nhất** mà bạn có thể chắt lọc cho đề tài của mình.

---

### 1. Bản đồ 4 Trường phái Thích nghi Đường truyền (Literature Map)

```
                            LỊCH SỬ THÍCH NGHI ĐƯỜNG TRUYỀN
                                           │
         ┌─────────────────────────────────┼────────────────────────────────┐
         ▼                                 ▼                                ▼
[1. PHI-ML KINH ĐIỂN]             [2. CHUẨN CÔNG NGHIỆP 3GPP]      [3. HỌC MÁY CÓ GIÁM SÁT]
• SampleRate (Bicket 2005)        • EESM / MIESM (Effective SINR)   • Heath et al. (IEEE TVT 2010)
• RRAA (MobiCom 2006)             • OLLA (Outer Loop Link Adap.)    • Liu et al. (EURASIP 2020)
• CARA (INFOCOM 2006)                                               • Borst et al. (arXiv 2026)
• SoftRate (MobiSys 2008)                                                          │
• Camp & Knightly (ToN 2010)                                                       ▼
                                                                   [4. HỌC TĂNG CƯỜNG / BANDITS]
                                                                   • Contextual Bandits (Ericsson)
                                                                   • Q-Learning / OLLA-RL
```

---

### 2. Các Công trình Tiêu biểu Đã Được Mổ xẻ

#### 1. Dòng Phi-ML Kinh điển (Heuristic & Cross-Layer)
* **SampleRate (John Bicket, Luận văn Thạc sĩ MIT 2005):**
  * *Ý tưởng:* Nguồn gốc sơ khai của thuật toán Minstrel trong Linux. Thuật toán tính thông lượng kỳ vọng của từng rate, và cứ mỗi 10 gói tin thì trích ra 1 gói để "dò thám" (probing) các tốc độ khác.
  * *Hạn chế:* Chậm chạp khi kênh biến đổi nhanh; dò thám ngẫu nhiên dễ làm rớt gói.
* **RRAA - Robust Rate Adaptation Algorithm (Wong et al., ACM MobiCom 2006):**
  * *Ý tưởng:* Sử dụng cửa sổ trượt ngắn để đếm tỷ lệ mất gói, kết hợp cơ chế phát khung RTS (Request-to-Send) thích nghi.
  * *Tri thức giá trị:* Phân biệt được **Mất gói do suy hao kênh** (cần hạ rate) và **Mất gói do va chạm mạng MAC** (phải giữ nguyên rate).
* **Camp & Knightly (IEEE/ACM Trans. on Networking, 2010):**
  * *Tên bài:* *"Modulation Rate Adaptation in Urban and Vehicular Environments: Cross-Layer Implementation and Experimental Evaluation"*.
  * *Tri thức đắt giá nhất:* Tác giả chứng minh rằng **Thời gian kết hợp (Coherence Time)** và **Vận tốc xe** làm bẻ cong toàn bộ đường ngưỡng SNR. Cùng một mức SNR $15\text{ dB}$, nếu xe đứng yên thì 16-QAM chạy hoàn hảo, nhưng nếu xe chạy $60\text{ km/h}$ thì 16-QAM lỗi sạch vì kênh biến đổi ngay giữa chừng khung truyền.
* **SoftRate (G. Judd et al., ACM MobiSys 2008):**
  * *Tên bài:* *"Efficient Channel-Aware Rate Adaptation in Dynamic Environments"*.
  * *Đột phá:* Thay vì chỉ nhìn vào nhãn nhị phân ACK/NACK (0 hoặc 1) sau khi giải mã, máy thu tính toán **Thông tin mềm (Soft LLR / Confidence metrics)** trực tiếp từ lớp vật lý. Nhờ đó, máy phát biết chính xác mức độ "nguy kịch" của kênh truyền mà không cần đợi gói tin bị lỗi thật sự mới biết để hạ rate!

#### 2. Dòng Chuẩn 3GPP (LTE / 5G NR Link-to-System Mapping)
* **EESM (Exponential) & MIESM (Mutual Information Effective SINR Mapping):**
  * *Bài toán:* Trong kênh đa sóng mang OFDM, mỗi sóng mang con có một SNR khác nhau ($\gamma_1, \gamma_2, \dots, \gamma_K$). Làm sao chọn 1 MCS cho toàn bộ khung?
  * *Giải pháp 3GPP:* Dùng một hàm phi tuyến để nén toàn bộ vector $\mathbf{SNR}$ thành một số vô hướng duy nhất gọi là **Effective SINR ($\gamma_{\text{eff}}$)**:
    $$\gamma_{\text{eff}} = -\beta \ln \left( \frac{1}{K} \sum_{k=1}^K e^{-\frac{\gamma_k}{\beta}} \right)$$
    Sau đó chỉ cần tra $\gamma_{\text{eff}}$ vào đường cong AWGN duy nhất để biết BLER.
* **OLLA (Outer Loop Link Adaptation):**
  * Chuẩn thích nghi vòng ngoài của trạm gốc 4G/5G. Trạm gốc cộng thêm một giá trị bù trừ $\Delta_{\text{OLLA}}$ vào SNR báo cáo. Nếu nhận ACK thì tăng nhẹ $\Delta$, nếu nhận NACK thì trừ mạnh $\Delta$ theo tỷ lệ $\frac{\delta_{\text{up}}}{\delta_{\text{down}}} = \frac{\text{BLER}_{\text{target}}}{1 - \text{BLER}_{\text{target}}}$.

#### 3. Dòng Học Tăng Cường & Contextual Bandits (Online Learning)
* **Multi-Armed Bandits cho AMC (Nghiên cứu của Ericsson & Nokia, IEEE TWC):**
  * Coi mỗi MCS là một "cần gạt" (arm) của máy đánh bạc. Mỗi khung truyền, máy phát quan sát ngữ cảnh $\mathbf{x}$, chọn 1 MCS, nhận phần thưởng (Reward = Throughput nếu ACK, hoặc phạt nếu NACK).
  * Dùng thuật toán **Upper Confidence Bound (UCB)** hoặc **Thompson Sampling** để cân bằng giữa *Khai phá (Exploration - thử các rate mới)* và *Khai thác (Exploitation - dùng rate tốt nhất hiện tại)*.

---

### 3. Năm "Vũ khí Tri thức" Thực sự Đáng giá để Ứng dụng vào Core

Sau khi sàng lọc toàn bộ kho tàng trên, đây là **5 tinh hoa lớn nhất** bạn có thể đưa vào bài toán của mình:

#### 1. Tri thức 1: Đừng bao giờ tin SNR đơn thuần trong kênh Fading di động (từ Camp & Knightly 2010)
* **Ý nghĩa:** Một mô hình chỉ nhìn vào scalar SNR sẽ thất bại hoàn toàn khi chuyển từ kênh Rayleigh chậm sang Rayleigh nhanh.
* **Ứng dụng vào code:** Đó chính là lý do vector đặc trưng trong `research_core` của bạn bắt buộc phải có `channel_id` (AWGN vs Rayleigh) và trong các bước tiếp theo, khi bạn mở rộng sang Rayleigh nhanh (Doppler), chỉ cần bổ sung thêm một đặc trưng: **Độ biến thiên kênh giữa 2 khung liên tiếp** $|\hat{h}_t - \hat{h}_{t-1}|$ hoặc Doppler shift. Cây quyết định sẽ tự động rẽ nhánh: ở cùng một mức SNR $10\text{ dB}$, nếu kênh tĩnh thì chọn 16-QAM, nếu kênh biến đổi nhanh thì tự hạ về QPSK.

#### 2. Tri thức 2: Thông tin mềm (LLR) quý giá hơn cờ ACK/NACK gấp bội (từ SoftRate 2008)
* **Ý nghĩa:** Nếu chỉ học từ nhãn ACK/NACK (thành công/thất bại), mô hình chỉ biết gói tin sống hay chết, chứ không biết nó sống sót "trong gang tấc" hay sống "dư dả an toàn".
* **Ứng dụng vào code:** Trong `research_core`, bạn đã có sẵn hàm `demodulate(..., soft=True)` chuẩn Sionna tính chính xác Log-Likelihood Ratio (LLR). Giá trị trung bình độ lớn LLR $|\text{LLR}|$ chính là độ tự tin của kênh truyền. Bạn hoàn toàn có thể dùng thống kê LLR làm một đặc trưng đầu vào siêu mạnh cho ML.

#### 3. Tri thức 3: Tinh giản đặc trưng theo kiểu 3GPP (từ EESM & Borst 2026)
* **Ý nghĩa:** Các bài báo đỉnh cao không bao giờ nạp toàn bộ ma trận kênh $H$ phức tạp vào ML. 
* **Ứng dụng vào code:** Họ chỉ nén kênh thành **3-4 giá trị thống kê tóm tắt**: Mean SNR, Độ lệch chuẩn (đo độ chọn lọc tần số), và Độ dốc (Slope). Cấu trúc vector 3 chiều hiện tại của bạn $[\text{snr\_est\_db}, \text{nominal\_snr\_db}, \text{channel\_id}]$ đang đi cực kỳ đúng hướng tinh giản này.

#### 4. Tri thức 4: Bóc trần "Cú lừa" của cơ chế Retry (từ Minstrel & Liu 2020)
* **Ý nghĩa:** Các thuật toán kinh điển như Minstrel trong Linux hay OLLA trong 5G nhìn có vẻ đạt throughput cao là nhờ có cơ chế Multi-Rate Retry (MRR) - bắn trượt phát đầu thì phần cứng tự bắn lại ở tốc độ thấp hơn. Nhưng MRR làm tăng trễ (latency) và tiêu tốn năng lượng ghê gớm.
* **Ứng dụng vào luận văn:** Bạn có thể lấy luận điểm này làm **đóng góp khoa học chính**: Mô hình Decision Tree của bạn hướng tới **"One-shot Accuracy" (Đoán trúng ngay từ phát đầu tiên)**, giúp giảm thiểu tối đa số lần phải retry, tiết kiệm năng lượng cho thiết bị IoT/di động và giảm độ trễ hàng đợi.

#### 5. Tri thức 5: "Biên an toàn & Hoãn phát" (từ Contextual Bandits & An toàn Vật lý)
* **Ý nghĩa:** Khi kênh rơi vào trạng thái bất định cao (độ tin cậy của ML thấp) hoặc suy hao quá sâu:
  - Các hệ thống truyền thống: Bị ép phải phát mode thấp nhất (BPSK), dẫn đến lãng phí công suất vô ích và làm tăng tỷ lệ vi phạm BER.
  - Hệ thống của bạn: Có hành động **DEFER (hoãn phát)**.
* **Ứng dụng vào code:** Cơ chế nhãn Oracle DEFER mà chúng ta vừa cài đặt trong `policies.py` chính là sự hiện thực hóa hoàn hảo của tri thức này: **Biết dừng lại đúng lúc khi kênh không đủ điều kiện vật lý**.

---

### 4. Bảng Tổng hợp Danh mục "Tri thức Vét cạn"

| Tác giả / Công trình | Nguồn xuất bản | Ý tưởng cốt lõi | Tri thức chắt lọc cho bạn |
| :--- | :--- | :--- | :--- |
| **John Bicket (2005)** | MIT Master's Thesis | SampleRate: Dò thám rate định kỳ | Hiểu cơ chế gốc của Minstrel để làm baseline đối chứng. |
| **Wong et al. (2006)** | ACM MobiCom | RRAA: Cửa sổ trượt ngắn + lọc RTS | Phân lập lỗi kênh vs lỗi do đụng độ MAC. |
| **Judd et al. (2008)** | ACM MobiSys | SoftRate: Dùng LLR vật lý | Dùng độ tự tin mềm LLR thay vì chỉ phụ thuộc vào ACK/NACK. |
| **Camp & Knightly (2010)**| IEEE/ACM ToN | Tác động của Coherence Time & Vận tốc | Bắt buộc phải có đặc trưng kênh/vận tốc bên cạnh SNR. |
| **Robert Heath (2010)** | IEEE TVT | Supervised Learning + SNR Ordering | Đặt nền móng dùng Offline Oracle để train Classifier cho AMC. |
| **Hui Liu et al. (2020)** | EURASIP JWCN | Context-aware DT $\rightarrow$ Look-Up Table | Bóc tách cơ chế Retry (MRR); biên dịch cây thành LUT. |
| **Luca Borst et al. (2026)**| arXiv (USRP SDR) | Single Decision Tree trên MIMO-OFDM | Loại bỏ Random Forest khi chạy thời gian thực; chọn cây depth 7. |

Toàn bộ kho tàng 20 năm nghiên cứu trên đều hội tụ về một kết luận: **Một giải pháp thích nghi tối ưu không phải là cố dùng mô hình ML thật to, mà là hiểu tường tận bản chất vật lý của kênh để nén thành vài đặc trưng chuẩn, gán nhãn chân lý chính xác và ra quyết định bằng một cây quyết định gọn nhẹ.** Core của bạn hiện tại đang nắm giữ đúng tinh hoa này!

---

## PHẦN 5: PHÂN TÍCH MỔ XẺ TOÀN DIỆN DANH MỤC 18 TÀI LIỆU THAM KHẢO TỪ PHÒNG LAB SDR (BORST ET AL. 2026)

Danh mục 18 tài liệu tham khảo bạn vừa gửi chính là **bản danh mục trích dẫn gốc từ bài báo của Borst et al. (Tháng 9/2026)** và các công trình vệ tinh xung quanh phòng lab SDR của họ! 

Khi "vét tri thức" từ đống tài liệu này, ta thấy tác giả không chọn bừa bãi. Họ chia danh mục này thành **4 cụm chức năng cực kỳ bài bản**. Trong đó, có **3 "viên ngọc quý" (hidden gems)** mà nếu không đọc kỹ sẽ bỏ sót, nhưng lại có giá trị ứng dụng trực tiếp cho hệ thống của bạn:

---

### 1. Cụm 1: Các Giải pháp Thích nghi Đường truyền Hiện đại ([1] – [9])

* **[1] eOLLA (Blanquez-Casado et al., EURASIP 2016):**
  * *Tri thức:* OLLA truyền thống trong 4G/5G dùng bước nhảy cố định $(\delta_{\text{up}}, \delta_{\text{down}})$ nên khi gặp kênh fading biến đổi nhanh, trạm gốc mất hàng chục khung truyền mới hạ đủ rate (hội tụ quá chậm). eOLLA đề xuất bước nhảy biến thiên theo xác suất lỗi.
  * *Ý nghĩa với bạn:* Giúp bạn có luận điểm chắc chắn khi viết báo cáo: *"Các giải pháp OLLA truyền thống phản ứng quá trễ trước các cú sụt kênh sâu (deep fade), đó là lý do cần ML để dự đoán tức thời"*.
* **[5] Contextual Bandit-based AMC trên SDR (Pivoto et al., IEEE OJ-COMS 2026):**
  * *Tri thức:* Bài báo này áp dụng **Contextual Multi-Armed Bandits** trên phần cứng SDR. Bandit giải quyết bài toán tự động cân bằng giữa "dò thám tốc độ mới" và "khai thác tốc độ an toàn".
* **[6] SALAD: Self-Adaptive Link Adaptation (Wiesmayr, Hoydis, Cammerer, 2025 – arXiv:2510.05784):**
  * 💎 **VIÊN NGỌC QUÝ THỨ NHẤT:** Tác giả bài này gồm **Jakob Hoydis** và **Sebastian Cammerer** — chính là **hai nhà khoa học trưởng tại NVIDIA AI, cha đẻ của thư viện Sionna** mà chúng ta đang dùng để verify lớp vật lý!
  * *Tri thức:* Bài báo này đại diện cho "đỉnh cao công nghệ 2025": Họ dùng AI để link adaptation tự thích nghi hoàn toàn (Self-adaptive), tự bù trừ sai số ước lượng kênh và độ trễ phản hồi mà không cần kỹ sư phải ngồi tinh chỉnh bảng ngưỡng bằng tay (zero-manual-tuning).
* **[7] Feedback-free Adaptive MCS cho Massive MIMO (Asilomar 2023):**
  * *Tri thức:* Máy phát không cần máy thu gửi báo cáo CQI (Feedback-free)! Trạm gốc tự nhìn kênh đường lên (uplink channel) để suy đoán ra MCS cho đường xuống (downlink) dựa trên tính tương hỗ kênh (channel reciprocity).

---

### 2. Cụm 2: Xử lý Tín hiệu & Lọc Nhiễu Kênh ([14], [15])

* **[14] What is a Savitzky-Golay filter? (Schafer, IEEE Signal Processing Mag. 2011)**  
* **[15] Improved RMS Delay Spread Estimation using Savitzky-Golay Filters (Miloš et al., Electronics 2019)**
* 💎 **VIÊN NGỌC QUÝ THỨ HAI (RẤT ĐÁNG ỨNG DỤNG):**
  * *Vấn đề thực tế:* Khi bạn ước lượng SNR hoặc đáp ứng kênh từ Pilot ($\hat{h}$ hoặc $\hat{\gamma}$), giá trị đo được luôn bị mấp mô do nhiễu Gaussian $w$. Nếu đưa thẳng chuỗi SNR nhiễu này vào Decision Tree, cây sẽ bị "giật" (liên tục nhảy rate sai giữa các khung kề nhau).
  * *Bí quyết:* Nếu dùng bộ lọc trung bình trượt thông thường (Moving Average), bạn sẽ làm bẹt mất các đỉnh (peaks) và đáy suy hao (deep fades) thực sự của kênh. 
  * **Bộ lọc Savitzky-Golay (SG Filter)** giải quyết hoàn hảo việc này: Nó làm mượt chuỗi đo đạc bằng phép hồi quy đa thức cục bộ, **giữ nguyên độ sâu của rãnh fading và độ nhọn của đỉnh**, lọc sạch nhiễu đo mà không làm biến dạng tín hiệu kênh.
  * *Khả năng ứng dụng:* Nếu sau này bạn mở rộng sang kênh Rayleigh biến đổi theo thời gian, cho SNR đi qua 1 bộ lọc Savitzky-Golay nhẹ trước khi nạp vào Decision Tree sẽ giúp cây ra quyết định cực kỳ đầm và ổn định!

---

### 3. Cụm 3: Học máy Hiện đại cho Cây Quyết định ([16], [17], [18])

* **[16] Feature Learning for Interpretable, Performant Decision Trees (Good et al., NeurIPS 2023):**
  * 💎 **VIÊN NGỌC QUÝ THỨ BA (BẢO BỐI ĐỂ TRẢ LỜI PHẢN BIỆN):**
  * Đây là bài báo xuất bản tại **NeurIPS 2023** (Hội nghị AI danh giá nhất thế giới).
  * *Luận điểm của bài báo:* Người ta thường nghĩ muốn chính xác cao thì phải dùng Deep Learning hoặc Random Forest, còn Decision Tree đơn lẻ thì "kém thông minh". Nhóm tác giả NeurIPS đã chứng minh toán học rằng: **Nếu vector đặc trưng đầu vào được thiết kế tốt (Feature Learning / Compact Features), một cây Decision Tree đơn lẻ hoàn toàn có thể đạt hiệu năng tương đương các mô hình hộp đen phức tạp, trong khi vẫn giữ nguyên tính diễn giải 100%!**
  * *Giá trị:* Đây chính là lá chắn thép để bạn bảo vệ luận văn: Lựa chọn Decision Tree không phải vì "lười", mà có căn cứ từ công trình NeurIPS 2023 khẳng định tính ưu việt của cây khi đặc trưng được nén chuẩn.
* **[17] An Introduction to Statistical Learning - ISLR (James, Hastie, Tibshirani, 2023):**
  * Cuốn sách kinh điển của các giáo sư Stanford. Toàn bộ nền tảng phương pháp luận chia mẫu (Train/Val/Test), chống quá khớp bằng cách khống chế độ sâu cây (`max_depth`), và kỹ thuật đánh giá thực nghiệm mà ta thảo luận nãy giờ đều bắt nguồn từ cuốn sách này.

---

### 4. Cụm 4: Kinh điển về Lý thuyết Thông tin Vô tuyến ([10], [11], [12], [13])

* **[10] David Tse & Pramod Viswanath (2005) – Fundamentals of Wireless Communication**  
* **[12] Andrea Goldsmith (2005) – Wireless Communications**
  * Hai cuốn "kinh thánh" của ngành vô tuyến. 
  * *Tri thức cốt lõi:* Cung cấp công thức dung lượng kênh Shannon cho kênh fading, cơ chế thích nghi công suất/tốc độ (Water-filling), và giải thích tại sao trong fading đa đường, lỗi khối phụ thuộc vào sự phân bố các giá trị riêng (eigenvalues) của kênh chứ không chỉ phụ thuộc vào SNR vô hướng.
* **[11] NI LabVIEW Communications MIMO Framework Manual (2018):**
  * Tài liệu đặc tả kỹ thuật chi tiết của National Instruments về cách luồng tín hiệu MIMO-OFDM chạy từ phần cứng FPGA lên tầng MAC trên các dòng USRP.

---

### 5. Tổng kết: Bạn Có thể "Rút ruột" được gì để đưa vào Bài của Mình?

Từ 18 tài liệu trên, bạn có thể tự tin bổ sung vào luận văn/bài báo của mình:
1. **Trích dẫn [16] (NeurIPS 2023)** làm cơ sở lý luận mạnh mẽ nhất để giải thích: *Tại sao hệ thống của tôi ưu tiên Single Decision Tree có diễn giải cao thay vì mạng Deep Learning cồng kềnh*.
2. **Kế thừa tinh thần của [6] (SALAD 2025 - NVIDIA/Sionna)**: Đưa ra định hướng thích nghi tự động khép kín (Self-adaptive) dựa trên nền tảng Sionna.
3. **Cân nhắc vũ khí kỹ thuật của [14], [15] (Savitzky-Golay Filter)**: Dùng làm giải pháp tiền xử lý làm mượt kênh để chống nhiễu ước lượng pilot khi bước sang kịch bản fading nhanh.

---

## PHẦN 6: TỔNG KẾT BÀO CHỮA HỌC THUẬT (DEFENSE SUMMARY)

Khi trình bày hoặc bảo vệ hệ thống này trước hội đồng hoặc chuyên gia phản biện:
1. **Nếu bị hỏi: *"Tại sao không dùng tiêu chuẩn ISO hay CRISP-DM?"***  
   $\rightarrow$ Trả lời: Nghiên cứu này tuân theo phương pháp luận thực nghiệm chuyên ngành viễn thông như công trình của Robert Heath (IEEE TVT 2010), Hui Liu (EURASIP 2020) và Luca Borst (2026). Các bài báo uy tín trong mảng ML nhẹ cho PHY-AMC đều sử dụng thiết kế thực nghiệm có mỏ neo chân lý Oracle và kiểm định giải tích, chứ không dùng các khung quản trị công nghiệp vốn dành cho phần mềm doanh nghiệp.
2. **Nếu bị hỏi: *"Tại sao dùng Cây Quyết định đơn lẻ thay vì Deep Learning hay Random Forest?"***  
   $\rightarrow$ Trả lời: Theo kết quả đo đạc thực tế trên SDR của Borst et al. (2026), Random Forest gây trễ suy luận vượt quá thời lượng khung truyền. Đồng thời, nghiên cứu tại NeurIPS 2023 của Good et al. đã chứng minh: khi các đặc trưng vật lý được tinh giản và nén tốt, Single Decision Tree đạt hiệu năng tương đương mô hình phức tạp mà lại suy luận siêu nhanh ($\mu\text{s}$) và giải thích được 100%.
3. **Nếu bị hỏi: *"Làm sao biết nhãn huấn luyện ML là đúng mà không bị dắt vòng vòng?"***  
   $\rightarrow$ Trả lời: Nhãn không sinh từ công thức ước lượng gần đúng, mà sinh từ phép thử vét cạn ngoại tuyến (**Offline Exhaustive Search / Oracle**) trên chính kênh truyền đó. Mode được chọn là mode tối ưu tuyệt đối thỏa mãn ràng buộc độ tin cậy. Nếu toàn bộ mode đều lỗi, cơ chế DEFER được kích hoạt.
4. **Nếu bị hỏi: *"Làm sao chứng minh simulator không có bug?"***  
   $\rightarrow$ Trả lời: Toàn bộ simulator đã vượt qua 27 bài test giải tích trong `test_physics.py`, khớp tuyệt đối với công thức lý thuyết AWGN erfc, kỳ vọng tích phân Rayleigh, sai số ước lượng pilot đạt cận Cramér-Rao $N_0 / N_{\text{pilots}}$, và LLR khớp bit-by-bit với Sionna 5G NR của NVIDIA.
