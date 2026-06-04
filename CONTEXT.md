# CONTEXT.md — AL-1605-0740 SPECTRA → ANSYS Heat Flux Assignment
## Claude Code 業務背景文件
**Version 2.0**
**此文件供 Claude Code 開發時使用，補充 CLAUDE.md（12 rules）未涵蓋的領域知識。**

---

## 1. 工具用途

這是一個**同步輻射熱分析前處理工具**，橋接兩個獨立軟體領域：

| 領域 | 軟體 | 資料 |
|------|------|------|
| 光子束功率分佈 | **SPECTRA**（Tanaka & Kitamura） | 二維角度網格上的功率密度圖 [kW/mrad²] |
| 結構/熱有限元素分析 | **ANSYS APDL** | 網格節點、元素、熱通量面元素 |

工具讀取兩份資料集，定義幾何座標轉換，產生 **ANSYS External Data** `.inp` 檔，
將插值後的投影熱通量 [W/mm²] 映射到 FEA 模型中每個熱通量面節點。

**應用背景：** ALS-U（Advanced Light Source Upgrade）波動器/擺動器前端熱分析。
預設輸出檔名：`EPU66-27A power for ansys.inp`（橢圓極化波動器束線元件）。

---

## 2. 核心資料流（含效能改進路徑）

```
ANSYS .dat 檔                         SPECTRA .dta 檔
    │                                       │
    ▼  numpy 向量化解析                     ▼  逐行解析（資料量小）
io/ansys_reader.py                   io/spectra_reader.py
    │                                       │
    ▼                                       ▼
AnsysNodeStore          +        list[SpectraNode]
(numpy array, ~20MB)             list[SpectraElement]（含 bilinear 係數）
list[AnsysHeatFluxElement]
    │                                       │
    └──────────────────┬────────────────────┘
                       ▼
          pipeline/mapping_pipeline.py
          向量化路徑（預設）：
            形心 numpy.mean → map_to_mrad_batch
            → find_elements_batch → 批次插值
            → 批次單位換算 → HeatFluxResultStore
                       │
                       ▼  批次 buffer 輸出
          io/output_writer.py → EPU66-27A power for ansys.inp
```

---

## 3. 資料規模與效能需求

### 3.1 ALS-U 典型規模

| 資料類型 | 數量級 | 說明 |
|---|---|---|
| ANSYS 節點數 | 50,000 – 500,000 | 大型模型可達百萬 |
| ANSYS heat flux 元素數 | 5,000 – 50,000 | 熱通量面的子集 |
| SPECTRA grid | 61×61 – 201×201 節點 | 3,721 – 40,401 點 |
| 輸出 .inp 行數 | = heat flux 元素數 | 每元素一行 |

### 3.2 天真實作的瓶頸

| 瓶頸 | 天真做法 | 問題 |
|---|---|---|
| 節點讀取 | 逐行 readline + dict insert | N=500K 時 10–30 秒，200MB 記憶體 |
| mapping 主迴圈 | Python for loop，逐元素計算 | N=50K 時 30–60 秒 |
| 輸出寫檔 | 逐行 f.write() | N=50K 行時 I/O wait 顯著 |

### 3.3 改進後效能目標

| 操作 | 目標耗時（50K 元素）|
|---|---|
| ANSYS 節點讀取 | < 3 秒 |
| mapping 計算 | < 2 秒 |
| 輸出寫檔 | < 0.5 秒 |
| **總端對端** | **< 10 秒** |

---

## 4. I/O 與記憶體設計（必須遵守）

### 4.1 節點儲存：AnsysNodeStore（model/ansys_node_store.py）

**為何必要：**
- `dict[int, AnsysNode]` 每個 Python object 約 400 bytes；500K 節點 ≈ 200 MB
- 每次 heat flux 解析需查詢 8 個節點座標（共 N_hf × 8 次 dict lookup）
- dict of objects 無法直接傳入 numpy 向量計算

**設計：**
```python
class AnsysNodeStore:
    # node_ids: np.ndarray(int32,  shape=(N,))
    # xyz:      np.ndarray(float64, shape=(N,3))
    # _id_to_idx: dict[int, int]  — 一次建立，O(1) 查詢

    def get_xyz(self, node_id: int) -> np.ndarray:          # shape (3,)
    def get_xyz_batch(self, node_ids: list[int]) -> np.ndarray:  # shape (K,3)
```

**效益：**
- 記憶體：200 MB → 20 MB（10 倍）
- `get_xyz_batch(8 nodes)` vs 8次 dict lookup：快 50–100 倍
- 形心計算可直接用 `corner_xyz.mean(axis=1)`（numpy，無 Python loop）

### 4.2 節點向量化解析（io/ansys_reader.py）

**為何必要：**
- 逐行 `dict[str(id)] = AnsysNode(x,y,z)` 在 500K 行時有 500K 次 Python 函式呼叫
- numpy 批次解析可消除這些呼叫

**做法：**
```python
# 節點區段：收集原始行，解析完畢後一次建立 array
raw_lines = []  # 在 node section 內收集
# ...解析結束後...
data = np.array([[float(t) for t in line.split()] for line in raw_lines])
# data shape: (N, 4+)，取前 4 欄
node_store = AnsysNodeStore(
    node_ids=data[:, 0].astype(np.int32),
    xyz=data[:, 1:4]
)
```

### 4.3 結果儲存：HeatFluxResultStore（model/heatflux_result_store.py）

**為何必要：**
- `list[AnsysHeatFluxElement]` 每個 Python dataclass 物件的欄位散落在記憶體各處
- 向量化 pipeline 的輸出是 numpy array，需要對應的儲存結構
- `numpy.savetxt` 可以一次寫出整個結果，取代逐元素 format + write

**設計：**
```python
RESULT_DTYPE = np.dtype([
    ('element_id',    np.int32),
    ('x',             np.float64),
    ('y',             np.float64),
    ('z',             np.float64),
    ('xmrad',         np.float64),
    ('ymrad',         np.float64),
    ('distance_mm',   np.float64),
    ('grazing_rad',   np.float64),
    ('normal_pd',     np.float64),
    ('projected_pd',  np.float64),
    ('total_power_w', np.float64),
])

class HeatFluxResultStore:
    def __init__(self, n: int):
        self.data = np.zeros(n, dtype=RESULT_DTYPE)

    def to_output_array(self, total_power_ratio: float) -> np.ndarray:
        """回傳 (N,5) array 供 numpy.savetxt 直接使用"""
```

### 4.4 批次輸出（io/output_writer.py）

**為何必要：**
- 每次 `f.write()` 是一次系統呼叫（kernel context switch）
- 50K 次系統呼叫的 overhead 在機械硬碟上可達 5–10 秒

**做法：**
```python
# 方案 A：buffer（適用 list[AnsysHeatFluxElement]）
CHUNK = 10_000
lines = []
for elem in elements:
    lines.append(format_line(elem))
    if len(lines) >= CHUNK:
        f.writelines(lines)
        lines.clear()

# 方案 B：numpy.savetxt（適用 HeatFluxResultStore，最快）
np.savetxt(f, result_store.to_output_array(ratio),
           fmt='%.4E', delimiter=',')
```

### 4.5 mapping 向量化（pipeline/mapping_pipeline.py）

**為何必要：**
- Python for loop 每次迭代有約 100ns 的 interpreter overhead
- 50K 元素 × 每元素多個 Python 呼叫 ≈ 30–60 秒
- numpy 向量化將同樣計算壓縮至 0.5–2 秒

**批次 mrad 轉換（math_core/coordinate_transform.py）：**
```python
def map_to_mrad_batch(
    centroids: np.ndarray,   # (N,3) mm
    source: np.ndarray,      # (3,)  mm
    geometry: SourceGeometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = centroids - source                        # (N,3)
    R = np.linalg.norm(v, axis=1)                 # (N,)
    xmrad = np.arcsin(v @ geometry.e_x / R) * 1000
    ymrad = np.arcsin(v @ geometry.e_y / R) * 1000
    return xmrad, ymrad, R
```

**批次 grid 搜尋（math_core/spatial_search.py）：**
```python
def find_elements_batch(
    self,
    xmrad: np.ndarray,   # (N,)
    ymrad: np.ndarray    # (N,)
) -> np.ndarray:         # (N,) int，-1 = 在 grid 外
    xi = np.searchsorted(self.x_boundaries, xmrad, side='right') - 1
    yi = np.searchsorted(self.y_boundaries, ymrad, side='right') - 1
    xi = np.clip(xi, 0, self.n_col - 2)
    yi = np.clip(yi, 0, self.n_row - 2)
    out = (xmrad < self.x_min) | (xmrad > self.x_max) | \
          (ymrad < self.y_min) | (ymrad > self.y_max)
    indices = yi * (self.n_col - 1) + xi + 1   # 1-based
    indices[out] = -1
    return indices
```

**兩條 pipeline 路徑：**
```python
def run_mapping(..., vectorized: bool = True):
    if vectorized:
        return _run_mapping_vectorized(...)   # numpy，快
    else:
        return _run_mapping_sequential(...)   # Python loop，供 debug
```

逐元素路徑（`vectorized=False`）保留作為正確性基準，
`test_mapping_pipeline.py` 驗證兩者結果誤差 < 1e-10。

---

## 5. 數學公式（實作必須嚴格依照此節）

### 5.1 角度座標轉換

對每個 ANSYS 熱通量元素的形心 **P** = (x, y, z) [mm]：

```
v = P − S = (x−xₛ,  y−yₛ,  z−zₛ)
R = |v|   [mm]
xmrad = arcsin( v · ê_X / R ) × 1000   [mrad]
ymrad = arcsin( v · ê_Y / R ) × 1000   [mrad]
```

有效範圍：`|angle| < 5 mrad`，`sin(θ) ≈ θ` 誤差 < 0.004%。使用 arcsin（與 VBA 一致）。

### 5.2 雙線性插值

Node layout 約定：
```
Node1(x1,y2) ── Node3(x2,y2)
     |                  |
Node2(x1,y1) ── Node4(x2,y1)
```

φ(x,y) = a₀ + a₁·x + a₂·y + a₃·x·y，Δ = (x₂−x₁)(y₂−y₁)

```
a₀ = (f₂·x₂·y₂ − f₁·x₂·y₁ − f₄·x₁·y₂ + f₃·x₁·y₁) / Δ
a₁ = (−f₂·y₂ + f₁·y₁ + f₄·y₂ − f₃·y₁) / Δ
a₂ = (−f₂·x₂ + f₁·x₂ + f₄·x₁ − f₃·x₁) / Δ
a₃ = (f₂ − f₁ − f₄ + f₃) / Δ
```

邊界條件（4 個，必須全部通過）：φ(x₁,y₁)=f₂，φ(x₁,y₂)=f₁，φ(x₂,y₁)=f₄，φ(x₂,y₂)=f₃

### 5.3 掠射角計算

```
v₁ = node1 − node2
v₂ = node3 − node2
n  = v₁ × v₂   （正確外法線，Python 版）
n̂  = n / |n|
θ_grazing = π/2 − arccos( n̂ · v̂ )
```

⚠️ Bug #2 修正：Python 版使用 `v₁ × v₂`（外法線），不使用 `abs()`。

### 5.4 單位換算

```
flux [W/mm²] = φ [kW/mrad²] × 10⁹ / R_mm²
φ_projected  = sin(θ_grazing) × φ_normal
P [W]        = A_surface [mm²] × φ_projected
```

### 5.5 正交基底

```python
e_z_raw = target - source
e_x_raw = side - source
e_y_raw = np.cross(e_z_raw, e_x_raw)
e_x = e_x_raw / np.linalg.norm(e_x_raw)
e_y = e_y_raw / np.linalg.norm(e_y_raw)
e_z = e_z_raw / np.linalg.norm(e_z_raw)
```

### 5.6 四邊形面積

```
area = 0.5 * (|v12 × v13| + |v13 × v14|)
v12 = node2−node1，v13 = node3−node1，v14 = node4−node1
```

### 5.7 SPECTRA Grid 元素索引（1-based）

```
j = ceil(elem_idx / (n_col-1))
i = elem_idx - (j-1)*(n_col-1)
index1=(j-1)*n_col+i, index2=j*n_col+i, index3=(j-1)*n_col+i+1, index4=j*n_col+i+1
```

---

## 6. 輸入/輸出檔案格式

### 6.1 ANSYS APDL 輸入檔（*.dat）

| 觸發字串 | 動作 |
|----------|------|
| `"Nodes for the whole assembly"` | 切換到 node 模式，跳過後 3 行 |
| `/com,\*{1,} Elements for` | 切換到 element 模式，跳過後 4 行，**只計數不儲存** |
| `/com,\*{1,} Create "Heat Flux"` | 切換到 heat flux 模式，跳過後 4 行 |
| `"-1"` | 終止當前區段 |

節點行：token[0]=nodeID, token[1..3]=x,y,z [mm]
熱通量行：token[0]=element ID；token[5..8]=4 角點；token[9..12]=4 中點

### 6.2 SPECTRA 輸出檔（*.dta / *.data / *.dta2）

前 2 行 header 跳過。欄位：xmrad  ymrad  powerDensity[kW/mrad²]。
Grid 維度自動偵測：ymrad 第一次改變時，前一個索引 = n_col。

### 6.3 ANSYS External Data 輸出檔（*.inp）

每行剛好 **5 欄**（Bug #1 已修正）：x, y, z, pd×ratio, pd，格式 `{:.4E}`。

⚠️ VBA Bug #1：`Dim arrayString(10)` 多出 6 個零欄。Python 版不得複製此行為。

---

## 7. 已知 Bug 完整清單

### Bug #1 — 輸出多 6 個零欄 ⚠️ HIGH（必須修正）
- **修正位置：** `io/output_writer.py`
- **修正方式：** 直接輸出 5 欄；`test_output_writer.py` 驗證

### Bug #2 — Cross product 方向反 ⚠️ MEDIUM（必須修正）
- **修正位置：** `math_core/grazing_angle.py`
- **修正方式：** `np.cross(v1, v2)`，不使用 `abs()`；`test_grazing_angle.py` 驗證

### Bug #3 — dotProduct 索引錯誤（VBA 從未呼叫）
- **Python 處理：** 直接用 `np.dot()`，不實作自定義函式

### Bug #4 — 累加迴圈 typo（死迴圈）
- **Python 處理：** 直接計算 `sum(nodes)/4`，不實作迴圈

### Bug #5 — ReDim 變數名稱錯誤
- **Python 處理：** Python 無 ReDim，問題自然消失

---

## 8. 資料結構

### 8.1 核心 dataclass

```python
@dataclass
class AnsysNode:
    node_id: int; x: float; y: float; z: float   # mm

@dataclass
class AnsysHeatFluxElement:
    element_id: int
    corner_nodes: list[AnsysNode]    # 長度 4
    midside_nodes: list[AnsysNode]   # 長度 4
    surface_area_mm2: float = 0.0
    # pipeline 填入：
    x: float=0.0; y: float=0.0; z: float=0.0
    xmrad: float=0.0; ymrad: float=0.0
    distance_from_source_mm: float=0.0
    grazing_angle_rad: float=0.0
    normal_power_density_kw_mrad2: float=0.0
    projected_power_density_w_mm2: float=0.0
    total_power_w: float=0.0

@dataclass
class SpectraNode:
    node_id: int; xmrad: float; ymrad: float; power_density: float  # kW/mrad²

@dataclass
class SpectraElement:
    nodes: list[SpectraNode]   # 長度 4，VBA node layout
    area_mrad2: float
    a0: float; a1: float; a2: float; a3: float
    def interpolate(self, xmrad, ymrad): return self.a0+self.a1*xmrad+self.a2*ymrad+self.a3*xmrad*ymrad
```

### 8.2 spatial_search.py searchsorted 邊界處理

```python
def find_element(self, xmrad, ymrad):
    if xmrad < self.x_min or xmrad > self.x_max: return None
    if ymrad < self.y_min or ymrad > self.y_max: return None
    xi = min(np.searchsorted(self.x_boundaries, xmrad, side='right')-1, self.n_col-2)
    yi = min(np.searchsorted(self.y_boundaries, ymrad, side='right')-1, self.n_row-2)
    return yi*(self.n_col-1) + xi + 1   # 1-based
```

---

## 9. mapping_pipeline.py 介面規格

```python
def run_mapping(
    hf_elements: list[AnsysHeatFluxElement],
    spectra_elements: list[SpectraElement],
    grid: SpectraGrid,
    geometry: SourceGeometry,
    source: np.ndarray,                          # (3,) mm
    progress_cb: Callable[[int,int],None] | None = None,
    vectorized: bool = True
) -> list[AnsysHeatFluxElement]:
    """
    vectorized=True：numpy 全程，快速（預設）
    vectorized=False：Python loop，供 debug 及正確性驗證
    元素落在 grid 外：projected_power_density_w_mm2 保持 0.0，不報錯。
    """
```

---

## 10. GUI 按鈕啟用序列

```
啟動：[Upload ANSYS] 可用；其餘停用
載入 ANSYS 後：[Upload SPECTRA] 啟用；顯示節點數/元素數/flux元素數
載入 SPECTRA 後（執行 mapping）：[Create file][Update geometry] 啟用；顯示 grid 統計
Create 後：顯示輸出元素數/總功率；[View location] 啟用
Exit：→ .backup.json
```

---

## 11. session_backup.py 格式

```json
{
  "timestamp": "2025-01-15T14:32:00",
  "ansys_file": "/path/to/model.dat",
  "spectra_file": "/path/to/power.dta",
  "geometry": {"source":[0,0,0], "target":[0,0,10000], "side":[1,0,0]},
  "output": {"directory":"/path/to/out", "filename":"EPU66-27A power for ansys.inp", "total_power_ratio":1.0},
  "stats": {"ansys_nodes":12345, "ansys_elements":6789, "ansys_heatflux_elements":234,
            "spectra_rows":61, "spectra_cols":61, "spectra_peak_kw_mrad2":12.34, "spectra_total_power_kw":5.67}
}
```

---

## 12. 不移植項目（Rule 2）

| VBA 模組 | 原因 |
|---|---|
| `ansysElement.cls` | 只計數不儲存，Python 維持同策略 |
| `oneAnsysSpectraHeatFluxType.cls` | 空骨架 |
| `testProgress.bas` | stub |
| `readGeometry.bas` | 非主路徑 |
| `ExportAllModule.bas` | Python 不需要 |

---

## 13. 外部依賴

```
numpy>=1.24
```

tkinter 為標準庫內建。pytest 為開發依賴，不列入 requirements.txt。

---

---

## 14. Logging 架構（2026-06-04 新增）

### 14.1 設計原則

- **完全靜默**：所有 log 寫入 `heatflux.log`，不輸出 console，不增加 modal dialog
- **非阻斷警告**：GUI 只在 footer 警告條顯示重要警告（橘色，12 秒後自動消失）
- 等級分工：DEBUG = 內部流程；INFO = 重要事件；WARNING = 需注意但不阻斷；ERROR = 例外含 traceback

### 14.2 初始化

```python
# main.py — Tk 啟動前呼叫
from heatflux.config.app_logger import setup_logging
setup_logging()   # 寫入 heatflux.log，5 MB × 3 份 rotating
```

所有模組的 module-level logger：
```python
import logging
_log = logging.getLogger(__name__)
```

### 14.3 各模組 log 事件速查

| 模組 | INFO | WARNING |
|------|------|---------|
| `ansys_reader.py` | 檔案大小、解析耗時、counts | flux=0、node=0、跳過 hf 行（含原因） |
| `spectra_reader.py` | grid 尺寸、峰值 PD、耗時 | grid < 3×3 |
| `output_writer.py` | 輸出路徑、行數、耗時 | 目標檔已存在（覆蓋） |
| `mapping_pipeline.py` | 開始條件、耗時、out-of-grid 統計 | out-of-grid > 5% |
| `session_backup.py` | save/load 路徑 | — |
| `app_window.py` | 使用者操作、mapping 完成統計 | UI 警告條觸發（ERROR：mapping 例外含 traceback） |

### 14.4 GUI 警告條

```python
# app_window.py
self.warn_strip_var = tk.StringVar(value="")   # footer 右側，橘色
self._warn_after_id: str | None = None

def _post_warning(self, msg: str, duration_ms: int = 12000) -> None:
    # 取消舊計時 → 設文字 → 12s 後自動清除
    ...

def _clear_warning(self) -> None:
    self.warn_strip_var.set("")
    self._warn_after_id = None
```

觸發條件：mapping 完成後 `out_of_grid_count / len(mapped) > 0.05`。

*v2.1：新增 §14 logging 架構。*
