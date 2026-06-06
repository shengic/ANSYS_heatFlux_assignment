#!/usr/bin/env python3
"""
Interactive 3D Power Density Visualization App
- Switch between 3D Scatter and 3D Surface Color Map
- Dynamic X, Y, Z axis scaling with sliders
- Real-time plot updates
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from pathlib import Path
from scipy.interpolate import griddata

# Page configuration
st.set_page_config(
    page_title="3D Power Density Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 3D Power Density Visualizer")
st.markdown("Interactive visualization with dynamic axis scaling and plot type selection")

# ============ SIDEBAR CONTROLS ============
st.sidebar.header("⚙️ Configuration")

# File path input
default_data_path = os.environ.get("DATA_FILE_PATH", r"K:\ANSYS heat flux assignment\results\CU15-19A power for ansys.inp")
data_file = st.sidebar.text_input(
    "Data file path:",
    value=default_data_path,
    help="Full path to your .inp or .txt file"
)

skip_rows = st.sidebar.number_input("Skip header rows:", value=0, min_value=0)

# Plot type selection
st.sidebar.subheader("2️⃣ Plot Type")
plot_type = st.sidebar.radio(
    "Select visualization:",
    ["3D Scatter", "3D Surface Color Map"],
    help="Scatter: individual points | Surface: interpolated mesh"
)

# Color scale selection
st.sidebar.subheader("3️⃣ Colormap Options")
colorscale = st.sidebar.selectbox(
    "Color scale:",
    ["Turbo", "Inferno", "Viridis", "Hot", "Plasma", "Cividis", "Blues"],
    index=0
)

use_log_scale = st.sidebar.checkbox(
    "Use log scale for power density",
    value=False,
    help="Enable if power spans multiple decades (e.g., 1-1e6)"
)

# Axis scaling sliders
st.sidebar.subheader("4️⃣ Axis Scaling")
st.sidebar.write("**Adjust scale factors (default: X=1, Y=1, Z=0.01)**")

x_scale = st.sidebar.slider(
    "X-axis scale:",
    min_value=0.001,
    max_value=2.0,
    value=1.0,
    step=0.01,
    format="%.3f"
)

y_scale = st.sidebar.slider(
    "Y-axis scale:",
    min_value=0.001,
    max_value=2.0,
    value=1.0,
    step=0.01,
    format="%.3f"
)

z_scale = st.sidebar.slider(
    "Z-axis scale:",
    min_value=0.001,
    max_value=1.0,
    value=0.01,
    step=0.001,
    format="%.4f"
)

# Point/Surface styling
st.sidebar.subheader("5️⃣ Styling")

point_size = st.sidebar.slider(
    "Point size (Scatter only):",
    min_value=1,
    max_value=15,
    value=3,
    step=1
)

opacity = st.sidebar.slider(
    "Opacity:",
    min_value=0.1,
    max_value=1.0,
    value=0.85,
    step=0.05
)

# Interpolation method for surface
if plot_type == "3D Surface Color Map":
    interpolation_method = st.sidebar.selectbox(
        "Interpolation method:",
        ["linear", "cubic", "nearest"],
        help="Method for gridding scattered data"
    )
    grid_resolution = st.sidebar.slider(
        "Grid resolution:",
        min_value=20,
        max_value=100,
        value=50,
        step=5,
        help="Higher = finer mesh, but slower"
    )

# ============ MAIN CONTENT ============

@st.cache_data
def load_data(filepath, skip_rows):
    """Load and cache data"""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            st.error(f"❌ File not found: {filepath}")
            return None
        
        # Auto-detect delimiter
        with open(filepath) as f:
            first_line = f.readline()
            delimiter = ',' if ',' in first_line else r'\s+'
        
        df = pd.read_csv(filepath, delimiter=delimiter, skiprows=skip_rows, header=None, engine='python')
        data = df.iloc[:, :4].values
        
        if data.shape[1] < 4:
            st.error(f"❌ Expected 4+ columns, got {data.shape[1]}")
            return None
        
        return data
    
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return None


def create_scatter_plot(data, x_scale, y_scale, z_scale, colorscale, use_log_scale, point_size, opacity):
    """Create 3D scatter plot"""
    x = data[:, 0] * x_scale
    y = data[:, 1] * y_scale
    z = data[:, 2] * z_scale
    power = data[:, 3]
    
    # Color mapping
    if use_log_scale:
        power_min = power[power > 0].min() if np.any(power > 0) else 1e-10
        power_clipped = np.where(power > 0, power, power_min)
        color_values = np.log10(power_clipped)
        color_label = "Power Density (log10)"
    else:
        color_values = power
        color_label = "Power Density"
    
    # Create hover text
    hover_text = [
        f"<b>Position</b><br>" +
        f"X: {x_val:.4g}<br>" +
        f"Y: {y_val:.4g}<br>" +
        f"Z: {z_val:.4g}<br>" +
        f"<b>Power Density</b><br>" +
        f"Value: {p_val:.4g}"
        for x_val, y_val, z_val, p_val in zip(x, y, z, power)
    ]
    
    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=point_size,
            color=color_values,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title=color_label),
            opacity=opacity,
            line=dict(width=0)
        ),
        text=hover_text,
        hoverinfo="text",
        hovertemplate="%{text}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"3D Power Density Scatter (N={len(x)} points)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data"
        ),
        width=1000,
        height=700,
        hovermode="closest"
    )
    
    return fig


def create_surface_plot(data, x_scale, y_scale, z_scale, colorscale, use_log_scale, 
                       interpolation_method, grid_resolution, opacity):
    """Create 3D surface color map"""
    x = data[:, 0] * x_scale
    y = data[:, 1] * y_scale
    z = data[:, 2] * z_scale
    power = data[:, 3]
    
    # Create regular grid
    xi = np.linspace(x.min(), x.max(), grid_resolution)
    yi = np.linspace(y.min(), y.max(), grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate power values onto grid
    zi_grid = griddata((x, y), z, (xi_grid, yi_grid), method=interpolation_method)
    power_grid = griddata((x, y), power, (xi_grid, yi_grid), method=interpolation_method)
    
    # Color mapping
    if use_log_scale:
        power_min = power[power > 0].min() if np.any(power > 0) else 1e-10
        power_clipped = np.where(power_grid > 0, power_grid, power_min)
        color_values = np.log10(power_clipped)
        color_label = "Power Density (log10)"
    else:
        color_values = power_grid
        color_label = "Power Density"
    
    fig = go.Figure(data=go.Surface(
        x=xi_grid,
        y=yi_grid,
        z=zi_grid,
        surfacecolor=color_values,
        colorscale=colorscale,
        colorbar=dict(title=color_label),
        opacity=opacity,
        hovertemplate="X: %{x:.4g}<br>Y: %{y:.4g}<br>Z: %{z:.4g}<br>Power: %{surfacecolor:.4g}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"3D Power Density Surface (gridded from {len(x)} points)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data"
        ),
        width=1000,
        height=700,
        hovermode="closest"
    )
    
    return fig


# ============ LOAD AND PLOT ============

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📈 Visualization")

with col2:
    if st.button("🔄 Refresh Plot", width='stretch'):
        st.cache_data.clear()

# Load data
data = load_data(data_file, skip_rows)

if data is not None:
    # Display data statistics
    with st.expander("📊 Data Statistics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Points loaded", f"{data.shape[0]:,}")
        
        with col2:
            st.metric("X range", f"[{data[:, 0].min():.3g}, {data[:, 0].max():.3g}]")
        
        with col3:
            st.metric("Y range", f"[{data[:, 1].min():.3g}, {data[:, 1].max():.3g}]")
        
        with col4:
            st.metric("Z range", f"[{data[:, 2].min():.3g}, {data[:, 2].max():.3g}]")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Power range", f"[{data[:, 3].min():.3g}, {data[:, 3].max():.3g}]")
        
        with col2:
            st.metric("Colorscale", colorscale)
    
    # Create plot based on selection
    try:
        if plot_type == "3D Scatter":
            fig = create_scatter_plot(data, x_scale, y_scale, z_scale, colorscale, use_log_scale, point_size, opacity)
        else:  # 3D Surface Color Map
            fig = create_surface_plot(data, x_scale, y_scale, z_scale, colorscale, use_log_scale,
                                     interpolation_method, grid_resolution, opacity)
        
        # Display plot
        st.plotly_chart(fig, width='stretch')
        
        # Export options
        st.subheader("💾 Export")
        col1, col2 = st.columns(2)
        
        with col1:
            html_filename = st.text_input("HTML filename:", value="power_density_3d.html")
            if st.button("📥 Download as HTML", width='stretch'):
                fig.write_html(html_filename)
                st.success(f"✓ Saved to {html_filename}")
        
        with col2:
            png_filename = st.text_input("PNG filename:", value="power_density_3d.png")
            if st.button("📥 Download as PNG", width='stretch'):
                try:
                    fig.write_image(png_filename)
                    st.success(f"✓ Saved to {png_filename}")
                except Exception as e:
                    st.error(f"PNG export requires kaleido: pip install kaleido\n{e}")
    
    except Exception as e:
        st.error(f"❌ Error creating plot: {e}")
else:
    st.info("⏳ Waiting for valid data file...")

# ============ FOOTER ============
st.markdown("---")
st.markdown("""
**Controls:**
- 🎮 **Left-drag**: Rotate | **Scroll**: Zoom | **Right-drag**: Pan
- 📊 Use sidebar sliders to adjust axis scales in real-time
- 🔄 Switch between scatter and surface plots instantly
- 💾 Export as interactive HTML or static PNG
""")