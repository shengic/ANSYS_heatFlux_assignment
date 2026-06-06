import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ============ 頁面配置 ============
st.set_page_config(layout="wide", page_title="3D Surface & Contour Viewer")

st.markdown("""
<h1 style="font-size: 20px; font-family: Georgia;">⚡ 3D Surface Plot and Contour of SPECTRA Power Density</h1>
""", unsafe_allow_html=True)

st.markdown(r"""
<p style="font-size: 16px; color: gray; font-family: Georgia;">
File path: K:\ANSYS heat flux assignment\test sample\CU18-15A fixed mask1-0.data
</p>
""", unsafe_allow_html=True)

# ============ 側邊欄控制面板 ============
st.sidebar.markdown("""
<h3 style="font-family: Georgia;">⚙️ 繪圖參數設定</h3>
""", unsafe_allow_html=True)

# 1. 資料來源
st.sidebar.markdown("""
<h4 style="font-family: Georgia;">1️⃣ 資料來源</h4>
""", unsafe_allow_html=True)
data_file = st.sidebar.text_input(
    "資料檔案路徑:",
    value=os.environ.get("SPECTRA_DATA_FILE") or r"K:\ANSYS heat flux assignment\test sample\CU18-15A fixed mask1-0.data",
    help="支援 .data, .txt, .csv 等純文字格式"
)

skip_rows = st.sidebar.number_input("跳過的標頭列數:", value=2, min_value=0, max_value=10)

# 2. 軸縮放控制
st.sidebar.markdown("""
<h4 style="font-family: Georgia;">2️⃣ 軸縮放</h4>
""", unsafe_allow_html=True)
st.sidebar.write("**預設值：X=1, Y=1, Z=1（無縮放）**")

x_scale = st.sidebar.slider(
    "X 軸縮放 (theta_x):",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.1,
    format="%.2f"
)

y_scale = st.sidebar.slider(
    "Y 軸縮放 (theta_y):",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.1,
    format="%.2f"
)

z_scale = st.sidebar.slider(
    "Z 軸縮放 (Power Density):",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.1,
    format="%.2f"
)

# 3. 色階與外觀
st.sidebar.markdown("""
<h4 style="font-family: Georgia;">3️⃣ 色階與外觀</h4>
""", unsafe_allow_html=True)
colorscale_option = st.sidebar.selectbox(
    "色彩對比 (Colorscale)",
    options=["Turbo", "Inferno", "Viridis", "Plasma", "Hot", "Cool", "Rainbow"],
    index=0,
    help="Turbo & Inferno 更好區分功率密度動態範圍"
)

opacity_value = st.sidebar.slider(
    "曲面透明度 (Opacity)",
    0.1, 1.0, 0.85,
    step=0.05
)

# 4. 網格與等高線
st.sidebar.markdown("""
<h4 style="font-family: Georgia;">4️⃣ 網格與等高線</h4>
""", unsafe_allow_html=True)

show_mesh_lines = st.sidebar.checkbox(
    "顯示曲面網格線 (Mesh Lines)",
    value=True,
    help="微弱白線，模擬 FEM 網格效果"
)

show_contours_z = st.sidebar.checkbox(
    "顯示 Z 軸等高線投影",
    value=True,
    help="在 XY 平面投射等高線"
)

use_log_scale = st.sidebar.checkbox(
    "使用對數色階",
    value=False,
    help="當功率跨多個數量級時啟用"
)

# ============ 固定參數 ============
MESH_REFINEMENT = 2  # 網格細化倍數
CONTOUR_DENSITY = 15  # 等高線密度

# ============ 網格插值函數 ============
def refine_mesh(x_coords, y_coords, z_matrix, refinement_factor):
    """使用雙三次樣條插值細化網格"""
    if refinement_factor == 1:
        return x_coords, y_coords, z_matrix
    
    try:
        from scipy.interpolate import RectBivariateSpline
        
        # 建立原始網格的樣條
        spl = RectBivariateSpline(
            y_coords, x_coords, z_matrix,
            kx=min(3, len(x_coords) - 1),
            ky=min(3, len(y_coords) - 1)
        )
        
        # 建立細化的座標網格
        new_len_x = len(x_coords) + (len(x_coords) - 1) * (refinement_factor - 1)
        new_len_y = len(y_coords) + (len(y_coords) - 1) * (refinement_factor - 1)
        
        x_fine = np.linspace(x_coords.min(), x_coords.max(), new_len_x)
        y_fine = np.linspace(y_coords.min(), y_coords.max(), new_len_y)
        
        # 使用樣條進行插值評估
        z_fine = spl(y_fine, x_fine, grid=True)
        
        # 確保沒有 NaN
        z_fine = np.nan_to_num(z_fine)
        
        return x_fine, y_fine, z_fine
    
    except Exception as e:
        st.warning(f"⚠️ 網格細化失敗: {e}。使用原始網格。")
        return x_coords, y_coords, z_matrix

# ============ 數據載入 ============
@st.cache_data
def load_data(file_path, skip_rows):
    """讀取資料檔案"""
    try:
        filepath = Path(file_path)
        
        if not filepath.exists():
            st.error(f"❌ 檔案不存在: {filepath}")
            return None
        
        with open(filepath) as f:
            first_line = f.readline()
            delimiter = ',' if ',' in first_line else r'\s+'
        
        df = pd.read_csv(
            file_path,
            sep=delimiter,
            skiprows=skip_rows,
            names=['theta_x', 'theta_y', 'P_Density'],
            engine='python'
        )
        
        df = df[pd.to_numeric(df['theta_x'], errors='coerce').notnull()]
        df = df[pd.to_numeric(df['theta_y'], errors='coerce').notnull()]
        df = df[pd.to_numeric(df['P_Density'], errors='coerce').notnull()]
        
        df = df.astype(float)
        
        if len(df) == 0:
            st.error("❌ 檔案中無有效資料")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"❌ 讀取檔案失敗: {e}")
        return None


# ============ 繪圖 ============
st.subheader("📈 3D 曲面與等高線投影")

df = load_data(data_file, skip_rows)

if df is not None and len(df) > 0:
    try:
        x_coords = np.sort(df['theta_x'].unique())
        y_coords = np.sort(df['theta_y'].unique())
        
        pivot_df = df.pivot(index='theta_y', columns='theta_x', values='P_Density')
        pivot_df = pivot_df.fillna(0)
        z_matrix = pivot_df.values
        
        with st.spinner(f"🔄 細化網格 (×{MESH_REFINEMENT})..."):
            x_coords, y_coords, z_matrix = refine_mesh(
                x_coords, y_coords, z_matrix, MESH_REFINEMENT
            )
        
        x_coords_scaled = x_coords * x_scale
        y_coords_scaled = y_coords * y_scale
        z_matrix_scaled = z_matrix * z_scale
        
        z_min = float(z_matrix_scaled.min())
        z_max = float(z_matrix_scaled.max())
        
        if use_log_scale:
            z_clipped = np.where(z_matrix_scaled > 0, z_matrix_scaled, z_min + 1e-6)
            z_for_color = np.log10(z_clipped)
            colorbar_title = "Power (log10)"
        else:
            z_for_color = z_matrix_scaled
            colorbar_title = "Power Density (kW/mrad²)"
        
        fig = go.Figure(data=[go.Surface(
            x=x_coords_scaled,
            y=y_coords_scaled,
            z=z_matrix_scaled,
            colorscale=colorscale_option,
            surfacecolor=z_for_color,
            opacity=opacity_value,
            
            contours={
                "x": {
                    "show": show_mesh_lines,
                    "color": "rgba(255,255,255,0.25)",
                    "width": 1
                },
                "y": {
                    "show": show_mesh_lines,
                    "color": "rgba(255,255,255,0.25)",
                    "width": 1
                },
                "z": {
                    "show": show_contours_z,
                    "usecolormap": True,
                    "project": {"z": True},
                    "highlightcolor": "white",
                    "size": max(0.01, (z_max - z_min) / CONTOUR_DENSITY)
                }
            },
            
            colorbar=dict(
                title={
                    'text': colorbar_title,
                    'side': 'top'
                },
                thickness=15,
                len=0.7
            ),
            
            hovertemplate=(
                "<b>θx:</b> %{x:.4g} mrad<br>" +
                "<b>θy:</b> %{y:.4g} mrad<br>" +
                "<b>Power:</b> %{z:.4g} kW/mrad²<br>" +
                "<extra></extra>"
            )
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f"θx (mrad) × {x_scale}",
                yaxis_title=f"θy (mrad) × {y_scale}",
                zaxis_title=f"Power Density (kW/mrad²) × {z_scale}",
                
                aspectratio=dict(x=1, y=1, z=0.55),
                
                xaxis=dict(
                    gridcolor="rgba(128,128,128,0.2)",
                    showbackground=True,
                    backgroundcolor="rgb(235, 235, 235)"
                ),
                yaxis=dict(
                    gridcolor="rgba(128,128,128,0.2)",
                    showbackground=True,
                    backgroundcolor="rgb(235, 235, 235)"
                ),
                zaxis=dict(
                    gridcolor="rgba(128,128,128,0.2)",
                    showbackground=True,
                    backgroundcolor="rgb(245, 245, 245)",
                    range=[z_min, z_max]
                ),
            ),
            
            margin=dict(l=0, r=0, b=0, t=40),
            height=850,
            hovermode="closest"
        )
        
        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"❌ 繪圖時發生錯誤: {e}")
        import traceback
        st.error(traceback.format_exc())

else:
    st.warning("⏳ 等待有效的資料檔案...")