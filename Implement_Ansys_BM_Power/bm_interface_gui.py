# Filename: bm_interface_gui.py
# Version: 1.2
# Description: GUI interface to parse APDL heat surface data and compute power deposition

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os
from parseAPDL2_0 import parse_ansys_apdl_optimized, get_heat_flux_element_normal_and_centroid
from find_BM5 import read_geometry, find_tangent_and_offset, define_plane_axes

# Utility function to compute grazing angle and power density
def compute_power_density(centroid, tangent_point, normal_vector, offset):
    direction_vector = np.array(centroid) - np.array(tangent_point)
    distance = np.linalg.norm(direction_vector)
    if distance == 0:
        return 0, 0, 0  # Avoid divide-by-zero

    unit_vector = direction_vector / distance
    normal_vector = np.array(normal_vector)
    dot_product = np.dot(-unit_vector, normal_vector)

    # Assume direction_vector is defined and distance is its norm.
    unit_vector = direction_vector / distance  # Normalize the incident vector

    # Ensure the normal_vector is a numpy array and assumed to be unit.
    normal_vector = np.array(normal_vector)

    # Compute the dot product between the reversed incident vector and the normal.
    # This gives cos(theta), where theta is the incidence angle.
    dot_product = np.dot(-unit_vector, normal_vector)

    # Clamp the dot product to the valid range to avoid numerical errors.
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute the incidence angle (theta) in radians.
    theta = np.arccos(dot_product)

    # Grazing angle is the complement: gamma = (pi/2 - theta).
    grazing_angle = np.pi / 2 - theta

    # Take the absolute value to ensure the angle is positive.
    grazing_angle = abs(grazing_angle)

    # Optionally, convert to degrees:
    grazing_angle_degrees = np.degrees(grazing_angle)

    #print("Grazing angle (radians):", grazing_angle)
    print("Grazing angle (degrees):", grazing_angle_degrees)


    # Calculate peak power density
    distance_m = distance / 1000.0
    peak_power_density = 418 / (distance_m ** 2) * np.sin(grazing_angle)

    # Power density formula
    denom = 2 * (0.608 * 0.17) ** 2
    exponent = -((offset / distance_m) ** 2) / denom
    power_density = peak_power_density * np.exp(exponent)

    return distance, power_density, grazing_angle

# GUI Application
class BMInterfaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bending Magnet Power Interface")

        self.file_path = None
        self.nodes = None
        self.hf_elements = None
        self.results = []

        # GUI layout using grid
        frame = tk.Frame(root, padx=20, pady=20)
        frame.pack()

        # File select row
        tk.Label(frame, text="Select APDL file:").grid(row=0, column=0, sticky='e')
        self.select_button = tk.Button(frame, text="select", command=self.load_apdl_file)
        self.select_button.grid(row=0, column=1)
        tk.Label(frame, text="(*.dat)").grid(row=0, column=2, padx=(10, 0))

        # Progress bar
        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate", length=200)
        self.progress.grid(row=0, column=3, padx=(20, 0))

        # Info labels
        self.node_label = tk.Label(frame, text="Total node read:     ")
        self.node_label.grid(row=1, column=0, columnspan=3, sticky='w', pady=(10, 0))

        self.elem_label = tk.Label(frame, text="Total heat flux element read:     ")
        self.elem_label.grid(row=2, column=0, columnspan=3, sticky='w')

        self.power_label = tk.Label(frame, text="Total power deposited on the model:     kW")
        self.power_label.grid(row=3, column=0, columnspan=3, sticky='w')

        # Save file row
        tk.Label(frame, text="Output .inp file for external data").grid(row=4, column=0, sticky='e', pady=(10, 0))
        self.save_button = tk.Button(frame, text="Save .inp File", command=self.save_inp_file, state=tk.DISABLED)
        self.save_button.grid(row=4, column=1, pady=(10, 0))
        tk.Label(frame, text="(*.inp)").grid(row=4, column=2, sticky='w', pady=(10, 0))

        # Exit button and version
        self.exit_button = tk.Button(frame, text="Exit", command=root.quit)
        self.exit_button.grid(row=5, column=1, pady=20)

        tk.Label(frame, text="Version 1.2 by Albert Sheng").grid(row=6, column=3, sticky='e')

    def load_apdl_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("DAT files", "*.dat")])
        if not self.file_path:
            return

        self.progress.start()
        self.nodes, self.hf_elements = parse_ansys_apdl_optimized(self.file_path)
        self.node_label.config(text=f"Total node read: {len(self.nodes)}")
        self.elem_label.config(text=f"Total heat flux element read: {len(self.hf_elements)}")

        self.process_geometry()
        self.progress.stop()

    def process_geometry(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            geo_path = os.path.join(script_dir, "BM_coordinate2.txt")
            s1, s2, c, radius = read_geometry(geo_path)
            s1_arr, s2_arr, c_arr = map(np.array, [s1, s2, c])
            n, x_dir, y_dir = define_plane_axes(s1_arr, s2_arr, c_arr)

            self.results = []
            for i, elem in enumerate(self.hf_elements):
                info = get_heat_flux_element_normal_and_centroid(self.nodes, self.hf_elements, i)
                centroid = info["centroid"]
                normal = info["normal"]
                if centroid and normal:
                    point = np.array(centroid)
                    distance, tp_3d, offset = find_tangent_and_offset(point, c, x_dir, y_dir, radius, n)
                    if tp_3d is not None:
                        dist, pdensity, grazing = compute_power_density(point, tp_3d, normal, offset)
                        self.results.append((centroid, dist, pdensity, grazing))

            total_power = sum(r[2] for r in self.results)
            self.power_label.config(text=f"Total power deposited on the model: {total_power:.1f} kW")
            self.save_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_inp_file(self):
        if not self.file_path:
            messagebox.showerror("Error", "No APDL file loaded.")
            return

        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        folder = os.path.dirname(self.file_path)
        save_path = os.path.join(folder, base_name + ".inp")
        log_path = os.path.join(folder, base_name + ".log")

        with open(save_path, 'w') as f_inp, open(log_path, 'w') as f_log:
            f_log.write("cx, cy, cz, tx, ty, tz, distance_mm, grazing_angle_deg, power_density\n")
            for centroid, dist, pdensity, grazing in self.results:
                f_inp.write(f"{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}, {pdensity:.6f}\n")   
                grazing_deg = np.degrees(grazing)
                tangent = centroid - dist * np.array([np.cos(grazing), np.sin(grazing), 0])  # approximate
                f_log.write(f"{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}, \
{tangent[0]:.6f}, {tangent[1]:.6f}, {tangent[2]:.6f}, {dist:.3f}, {grazing_deg:.3f}, {pdensity:.6f}\n")

        messagebox.showinfo("Saved", f"Output saved to {save_path}\nLog saved to {log_path}")


if __name__ == '__main__':
    root = tk.Tk()
    app = BMInterfaceApp(root)
    root.mainloop()
