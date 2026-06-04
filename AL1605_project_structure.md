# AL-1605-0740 — SPECTRA → ANSYS Heat Flux Assignment
## Python/Tk 移植專案結構文件
**Version 2.2 · Albert Sheng · 2026-06-04**
**依據 12-Rule Template 設計**

---

## 根目錄說明

根目錄名稱 `ANSYS heat flux assignment/` 含空格，Python package 不可含空格。
解決方式：根目錄僅存放 `main.py`、`requirements.txt`、`README.md` 及 `tests/`，
內部主 package 命名為合法的 `heatflux/`。`main.py` 位於根目錄，`sys.path` 在此，
所有 `from heatflux.xxx import ...` 均正常運作。

---

## 完整資料夾結構

```
ANSYS heat flux assignment/
│
├── main.py
├── requirements.txt
├── README.md
├── heatflux.log              ← 執行時產生，rotating（5 MB × 3），不提交
│
├── heatflux/
│   ├── __init__.py
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── app_window.py
│   │   ├── geometry_frame.py
│   │   └── progress_dialog.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── ansys_reader.py
│   │   ├── spectra_reader.py
│   │   └── output_writer.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── ansys_node.py
│   │   ├── ansys_node_store.py          ← 新增：numpy-backed 大量節點儲存
│   │   ├── ansys_heatflux_element.py
│   │   ├── heatflux_result_store.py     ← 新增：numpy structured array 結果儲存
│   │   ├── spectra_node.py
│   │   └── spectra_element.py
│   ├── math_core/
│   │   ├── __init__.py
│   │   ├── geometry.py
│   │   ├── bilinear_interpolation.py
│   │   ├── coordinate_transform.py      ← 新增 map_to_mrad_batch()
│   │   ├── grazing_angle.py
│   │   ├── unit_conversion.py
│   │   └── spatial_search.py            ← 新增 find_elements_batch()
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── mapping_pipeline.py          ← 新增向量化路徑；保留逐元素路徑供 debug
│   └── config/
│       ├── __init__.py
│       ├── app_logger.py            ← 新增：rotating file logger 初始化
│       ├── ansys_cache.py
│       ├── spectra_cache.py
│       └── session_backup.py
│
└── tests/
    ├── __init__.py
    ├── test_geometry.py
    ├── test_bilinear_interpolation.py
    ├── test_coordinate_transform.py
    ├── test_grazing_angle.py
    ├── test_unit_conversion.py
    ├── test_spatial_search.py
    ├── test_ansys_reader.py
    ├── test_spectra_reader.py
    ├── test_output_writer.py
    ├── test_ansys_node_store.py         ← 新增
    └── test_mapping_pipeline.py         ← 新增
```

---

## 層次依賴規則

```
gui        →  pipeline  →  math_core
                        →  model
           →  io        →  model
           →  config

math_core  →  model（唯讀）
model      →  （無任何依賴）
tests      →  可引用所有層；絕不引用 gui
```

`gui/` 是唯一允許 `import tkinter` 的層。
`math_core/` 與 `model/` 零 UI 依賴，可在無顯示環境執行或 CLI batch 呼叫。

---

## 各檔案職責說明

### 根目錄

| 檔案 | 職責 |
|------|------|
| `main.py` | 唯一入口點。建立 `tk.Tk()` root，實例化 `MainWindow`，進入 mainloop。不含任何業務邏輯。 |
| `requirements.txt` | 唯一外部依賴：`numpy>=1.24`。tkinter 為 Python 標準庫內建。 |
| `README.md` | 安裝步驟、執行方式、檔案格式說明。 |

---

### `heatflux/gui/` — 使用者介面層

| 檔案 | 對應 VBA | 職責 |
|------|----------|------|
| `app_window.py` | `form1` (UserForm) | 主視窗 (`ttk.Frame`)。包含四個區塊：ANSYS 檔案、SPECTRA 檔案、幾何設定、輸出設定。左右面板 5:5（`uniform="panels"`），預設 1080×1040。管理按鈕啟用/停用狀態序列（Upload → Map）。Footer 含兩個 label：左 = `footer_status_var`，右 = `warn_strip_var`（橘色 12 秒自動消失）。**ANSYS/SPECTRA 卡片**：path entry 旁有 `Browse...` 按鈕；entry 下方有 `Selected: <filename>` 小字 label；entry 在 `<Map>`/`<FocusOut>` 時 `xview_moveto(1.0)` 顯示檔名而非根目錄。**Cache 控制列**：兩張卡片各有 `Cache source:` label + Delete current / Delete all / Browse... 三顆按鈕；hint label 由 `_refresh_cache_hint()` 根據 `*_cache_file_exists()` 在 path 變動時自動更新。ANSYS / SPECTRA 各有獨立 cache browser Toplevel。 |
| `geometry_frame.py` | `form1` 幾何輸入區 | Source / Target / Side 三點各 3 個 `ttk.Entry`（共 9 欄）。「Update geometry」按鈕觸發重新計算。顯示計算後的正交基底向量 ê_X, ê_Y, ê_Z（唯讀標籤）。 |
| `progress_dialog.py` | `UserForm1` | Modal `tk.Toplevel`，內含 `ttk.Progressbar`。由 `mapping_pipeline.py` 透過 callback 更新進度值。 |

---

### `heatflux/io/` — 檔案讀寫層

效能改進說明請見本文件「I/O 效能設計」章節。

| 檔案 | 對應 VBA | 職責 |
|------|----------|------|
| `ansys_reader.py` | `readAnsys.bas` | State-machine parser。節點區段：收集原始行至 list，解析完畢後一次用 numpy 向量化建立 `AnsysNodeStore`。Heat flux 區段：逐行解析。Element 區段：僅計數，不儲存。**Log**：解析耗時、各 section 轉換（DEBUG）、flux=0 或 node=0（WARNING）、跳過的 hf 行含原因（WARNING）。 |
| `spectra_reader.py` | `readSpectra.bas` | 跳過前 2 行 header。解析三欄（xmrad, ymrad, powerDensity [kW/mrad²]）。自動偵測 `total_column`。**Log**：解析耗時、grid 尺寸、峰值 PD；grid < 3×3 時 WARNING。 |
| `output_writer.py` | `writeAnsysHeatFluxElements()` | **修正 VBA Bug #1**：輸出剛好 5 欄。套用 `total_power_ratio` 縮放。**Log**：輸出路徑、行數、耗時；目標檔已存在時 WARNING（覆蓋）。 |

---

### `heatflux/model/` — 純資料結構層

| 檔案 | 對應 VBA | 職責 |
|------|----------|------|
| `ansys_node.py` | `ansysCoordinate.cls` | `@dataclass AnsysNode`：`node_id: int`，`x/y/z: float` [mm]。用於小模型、測試、及單點存取介面。 |
| `ansys_node_store.py` | ─（新增）| `class AnsysNodeStore`：內部以兩個 numpy array 儲存所有節點（`node_ids: int32 (N,)`，`xyz: float64 (N,3)`）。維護 `id→index` dict 作為查找表。提供 `get_xyz(node_id)` 單點查詢及 `get_xyz_batch(node_ids)` 批次查詢。**記憶體用量比 dict[int, AnsysNode] 少 10 倍**（500,000 節點：200MB → 20MB）。 |
| `ansys_heatflux_element.py` | `ansysHeatFlux.cls`（資料部分）| `@dataclass AnsysHeatFluxElement`：`element_id`，`corner_nodes: list[AnsysNode]`（長度 4），`midside_nodes: list[AnsysNode]`（長度 4），`surface_area_mm2: float`，及所有計算結果欄位。用於逐元素存取介面。 |
| `heatflux_result_store.py` | ─（新增）| `class HeatFluxResultStore`：以 numpy structured array（`RESULT_DTYPE`）儲存所有熱通量元素的計算結果。欄位：`element_id, x, y, z, xmrad, ymrad, distance_mm, grazing_rad, normal_pd, projected_pd, total_power_w`。支援 `to_output_array()` 直接傳給 `numpy.savetxt`。 |
| `spectra_node.py` | `powerDensityNode.cls` | `@dataclass SpectraNode`：`node_id: int`，`xmrad/ymrad: float`，`power_density: float` [kW/mrad²]。無方法。 |
| `spectra_element.py` | `powerDensityElement.cls` | `@dataclass SpectraElement`：`nodes: list[SpectraNode]`（長度 4），`area_mrad2: float`，`a0/a1/a2/a3: float`。提供 `interpolate(xmrad, ymrad) -> float` 方法。 |

> **Node layout 約定（與 VBA 相同）：**
> ```
> Node1(x1,y2) ── Node3(x2,y2)
>      |                  |
> Node2(x1,y1) ── Node4(x2,y1)
> ```

---

### `heatflux/math_core/` — 純函式計算層

| 檔案 | 對應 VBA | 職責 |
|------|----------|------|
| `geometry.py` | `sourceGeometry.cls` | `class SourceGeometry`。`from_points()` classmethod 建構正交基底。`e_x / e_y / e_z: np.ndarray` shape (3,)。 |
| `bilinear_interpolation.py` | `powerDensityElement.calculcateInterpolateCoefficients()` + §5.2 | `bilinear_coefficients()` → `(a0,a1,a2,a3)`。 |
| `coordinate_transform.py` | `ansysHeatFlux.mapAnsysCoordinateToMrad()` + §5.1 | 單點：`map_to_mrad(centroid, source, geom) -> (xmrad, ymrad, distance_mm)`。**批次（新增）：`map_to_mrad_batch(centroids, source, geom) -> (xmrad_arr, ymrad_arr, dist_arr)`**，全程 numpy，無 Python for loop，速度提升 50–200x。 |
| `grazing_angle.py` | `ansysHeatFlux.getGrazingAngleInRadian()` + §5.3 | `get_grazing_angle_rad(corner_nodes, beam_vec) -> float`。**修正 VBA Bug #2**：正確 `v1×v2`，移除 `Abs()` 補丁。 |
| `unit_conversion.py` | §5.4 | `kw_mrad2_to_w_mm2(pd, distance_mm) -> float`。公式：`pd × 1e9 / distance_mm²`。 |
| `spatial_search.py` | `fallIntoThisSpectraElement.cls` | 單點：`find_element(xmrad, ymrad) -> int | None`，O(log N)。**批次（新增）：`find_elements_batch(xmrad_arr, ymrad_arr) -> np.ndarray`**，一次 `searchsorted` 處理全部元素，回傳 -1 代表在 grid 外，速度提升 50–100x。 |

---

### `heatflux/pipeline/` — 業務流程協調層

| 檔案 | 對應 VBA | 職責 |
|------|----------|------|
| `mapping_pipeline.py` | `main.bas storeAnsysInterpolateCoordinateAndPower()` | 提供兩條執行路徑：`run_mapping(..., vectorized=True)`（預設，向量化，快速）及 `run_mapping(..., vectorized=False)`（逐元素，供 debug/驗證）。**Log**：開始條件（元素數、source 點）、完成耗時、out-of-grid 數量及百分比；out-of-grid > 5% 時 WARNING。 |

---

### `heatflux/config/` — 設定持久化層

| 檔案 | 對應 VBA | 職責 |
|------|----------|------|
| `app_logger.py` | ─（新增）| `setup_logging(log_path)` — 一次性初始化 rotating file handler，寫入 `heatflux.log`（5 MB × 3 份）。DEBUG 等級，完全靜默。由 `main.py` 在 Tk 啟動前呼叫。所有模組用 `logging.getLogger(__name__)` 取得 logger。 |
| `ansys_cache.py` | ─（新增）| ANSYS parse cache：`save_*` / `load_*` / `delete_*` / `clear_all_*` / `has_valid_*_parse_cache`（完整 load 驗證）/ `ansys_cache_file_exists`（輕量：僅 `Path.exists()`，供 UI hint 用）/ `list_*_entries`（cache browser）。Cache key = sha256(resolved source path)，存於 `.cache/ansys/*.npz`。 |
| `spectra_cache.py` | ─（新增）| 結構與 ANSYS 對等：`save_*` / `load_*` / `delete_*` / `clear_all_*` / `has_valid_spectra_parse_cache` / `spectra_cache_file_exists`（輕量）/ `list_spectra_parse_cache_entries`。存於 `.cache/spectra/*.npz`。 |
| `session_backup.py` | `backupModule.bas` | `save_backup() / load_backup()`。格式：`.backup.json`。儲存/載入路徑記錄到 log。 |

---

### `tests/` — 測試層

| 檔案 | 驗證目標 | 關鍵斷言 |
|------|----------|----------|
| `test_geometry.py` | `SourceGeometry.from_points()` | `‖ê_X‖=1`；正交性；右手系 |
| `test_bilinear_interpolation.py` | `bilinear_coefficients()` | 全部 4 個邊界條件 |
| `test_coordinate_transform.py` | `map_to_mrad()` + `map_to_mrad_batch()` | 批次結果與逐點結果完全一致 |
| `test_grazing_angle.py` | `get_grazing_angle_rad()` | 無 `abs()` 仍回傳正值（Bug #2 驗證）|
| `test_unit_conversion.py` | `kw_mrad2_to_w_mm2()` | 反平方律；VBA 公式等價性 |
| `test_spatial_search.py` | `SpectraGrid` 單點及批次 | 邊界條件；批次結果與逐點一致；效能 < 10ms/1000次 |
| `test_ansys_reader.py` | `ansys_reader.py` | Token offset；狀態機轉換；節點數一致性 |
| `test_ansys_node_store.py` | `AnsysNodeStore` | 批次查詢與逐點一致；記憶體用量驗證 |
| `test_spectra_reader.py` | `spectra_reader.py` | Grid 維度偵測；node layout；邊界條件 |
| `test_output_writer.py` | `output_writer.py` | **每行剛好 5 欄**（Bug #1 驗證）；批次輸出與逐行輸出結果相同 |
| `test_mapping_pipeline.py` | `mapping_pipeline.py` | 向量化路徑與逐元素路徑對同一輸入產生相同結果（誤差 < 1e-10）|

---

## I/O 效能設計

### 問題根源

VBA 原始碼與天真的 Python 移植都面臨相同的擴充性問題：

| 資料規模（ALS-U 典型）| 數量 |
|---|---|
| ANSYS 節點數 | 50,000 – 500,000 |
| ANSYS heat flux 元素數 | 5,000 – 50,000 |
| SPECTRA grid | 61×61 – 201×201 節點 |

在此規模下，逐行讀寫與逐物件儲存的代價是：
- **逐行 `readline()` + `dict` insert**：每個節點一次 Python 函式呼叫，N=500K 時約耗時 10–30 秒
- **逐行 `f.write()`**：每行一次系統呼叫，N=50K 行時 I/O 等待顯著
- **dict[int, AnsysNode dataclass]**：每個 Python object 約 400 bytes，500K 節點 ≈ 200 MB，遠超實際需要
- **Python for loop 做 mapping**：50K 元素 × 每元素多步計算，約 30–60 秒

### 解決方案與效益

#### 方案 A — numpy 向量化節點解析（ansys_reader.py）

**做法：** 節點區段不逐行 insert，改為收集原始數值字串至 list，
解析完畢後一次呼叫 `numpy.fromstring()` 或 `numpy.loadtxt()` 建立 `(N,4)` array。

**效益：**
- 速度：10–50 倍提升（消除 N 次 Python 函式呼叫）
- 記憶體：節點座標以 `float64 (N,3)` 連續 array 儲存，cache-friendly

#### 方案 B — AnsysNodeStore（model/ansys_node_store.py）

**做法：** 以兩個 numpy array 取代 `dict[int, AnsysNode]`：
- `node_ids: np.ndarray(int32, N)` — 節點 ID
- `xyz: np.ndarray(float64, N×3)` — 座標
- `_id_to_idx: dict[int, int]` — ID 到陣列索引的查找表（一次建立，O(1) 查詢）

**效益：**

| 指標 | dict[int, AnsysNode] | AnsysNodeStore |
|---|---|---|
| 記憶體（500K 節點）| ~200 MB | ~20 MB |
| 單點查詢速度 | O(1) | O(1)（相當）|
| 批次查詢（8點/元素）| 8 次 dict lookup | 1 次 numpy index，快 50–100x |
| 向量化計算相容性 | 否（需逐一解包）| 是（直接傳入 numpy 運算）|

**為何必要：** heat flux 解析時每個元素需查詢 8 個節點座標；
mapping pipeline 的形心計算需要全部元素的所有角點座標。
使用 `get_xyz_batch()` 可將這些操作向量化，是後續計算加速的基礎。

#### 方案 C — 批次輸出 buffer（output_writer.py）

**做法：** 以 `CHUNK=10,000` 行為單位 buffer，呼叫 `f.writelines(buffer)`，
而非每行一次 `f.write()`。若使用 `HeatFluxResultStore`，則直接用 `numpy.savetxt`。

**效益：**
- 系統呼叫次數：N 次 → N/10,000 次
- 速度：5–20 倍提升（I/O wait 大幅減少）
- 對 50K 元素：從約 5 秒降至 < 0.5 秒

#### 方案 D — mapping 主迴圈向量化（pipeline/mapping_pipeline.py）

**做法：** 將五個計算步驟全部改為 numpy array 操作：

```
形心計算：  corner_xyz.mean(axis=1)          → centroids (N,3)
mrad 轉換：  map_to_mrad_batch()              → xmrad, ymrad, dist (N,)
Grid 搜尋：  find_elements_batch()            → elem_idx (N,)，-1=在外
插值：       vectorized bilinear              → normal_pd (N,)
單位換算：   pd * 1e9 / dist**2 * sin(theta) → projected_pd (N,)
```

**效益：**

| 路徑 | 50K 元素耗時（估計）|
|---|---|
| Python for loop（逐元素）| 30–60 秒 |
| numpy 向量化 | 0.5–2 秒 |
| **加速比** | **30–100x** |

**為何同時保留逐元素路徑：** 向量化程式碼較難除錯。
`vectorized=False` 路徑與原始 VBA 邏輯一對一對應，
作為向量化路徑的正確性基準（`test_mapping_pipeline.py` 驗證兩者結果誤差 < 1e-10）。

---

## VBA → Python 對照總表

| VBA 模組 / 類別 | Python 檔案 | 備註 |
|---|---|---|
| `main.bas`（globals + orchestration）| `pipeline/mapping_pipeline.py` + `main.py` | globals 改為函式參數傳遞 |
| `readAnsys.bas` | `io/ansys_reader.py` | 節點解析向量化 |
| `readSpectra.bas` | `io/spectra_reader.py` | |
| `backupModule.bas` | `config/session_backup.py` | .txt → .json |
| `form1` | `gui/app_window.py` + `gui/geometry_frame.py` | |
| `UserForm1` | `gui/progress_dialog.py` | |
| `ansysCoordinate.cls` | `model/ansys_node.py` + `model/ansys_node_store.py` | 新增大量節點 numpy 儲存 |
| `ansysElement.cls` | ─（不移植）| VBA 中僅計數不儲存；Python 維持同策略（Rule 2）|
| `ansysHeatFlux.cls` | `model/ansys_heatflux_element.py` + `model/heatflux_result_store.py` + `math_core/grazing_angle.py` + `math_core/coordinate_transform.py` | 計算方法拆至 math_core；新增 numpy 結果儲存 |
| `powerDensityNode.cls` | `model/spectra_node.py` | |
| `powerDensityElement.cls` | `model/spectra_element.py` + `math_core/bilinear_interpolation.py` | |
| `fallIntoThisSpectraElement.cls` | `math_core/spatial_search.py` | O(N)→O(log N)；新增批次查詢 |
| `sourceGeometry.cls` | `math_core/geometry.py` | |
| `oneAnsysSpectraHeatFluxType.cls` | ─（不移植）| 空骨架 class（Rule 2）|
| `testProgress.bas` | ─（不移植）| stub（Rule 2）|
| `readGeometry.bas` | ─（不移植）| 非主路徑（Rule 2）|

---

## 已修正的 VBA 已知 Bugs

| Bug | VBA 位置 | Python 修正位置 | 修正方式 |
|-----|----------|-----------------|----------|
| **#1 HIGH**：輸出多 6 個零欄 | `main.bas` `Dim arrayString(10)` | `io/output_writer.py` | 直接輸出 5 欄；`test_output_writer.py` 驗證 |
| **#2 MEDIUM**：cross product 方向反、靠 `Abs()` 補救 | `ansysHeatFlux.cls` `getGrazingAngleInRadian()` | `math_core/grazing_angle.py` | 重寫為正確 `v1×v2`；移除 `Abs()` 補丁；`test_grazing_angle.py` 驗證 |
| **#3 MEDIUM**：`dotProduct` 索引 `v1[3]` 應為 `v1[2]` | `ansysHeatFlux.cls` `dotProduct()` | `math_core/` 各 inline dot product | VBA 中從未被呼叫；Python 版直接寫正確，不實作該函式 |
| **#4 LOW**：typo `averagePowreDensity` 造成死迴圈 | `powerDensityElement.cls` | `model/spectra_element.py` | 不實作死迴圈；直接正確計算平均值 |
| **#5 LOW**：`getVZ()` 內 `ReDim getVY(2)` 拼錯 | `sourceGeometry.cls` | `math_core/geometry.py` | Python 不需要 `ReDim`；問題自然消失 |

---

*文件 v2.0 依據 AL-1605-0740 系統文件 v6.0、spectra_ansys_flowchart.html 及效能分析產生*
*設計原則遵循 12-Rule Template (CLAUDE.md)*
