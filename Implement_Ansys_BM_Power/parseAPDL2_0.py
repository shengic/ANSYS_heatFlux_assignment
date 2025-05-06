#!/usr/bin/env python3
"""
Version: 2.0
Timestamp: 2025-03-23 14:00:00

Description:
  This script provides a Tkinter file dialog for selecting an ANSYS APDL .dat file
  and parses:
    - Node coordinates
    - Heat flux surface element connectivity

  Key Functions:
    - parse_ansys_apdl_optimized(file_path):
        Reads the file, extracts node coordinates, and heat flux element connectivity.
        (Only the first heat flux section is parsed.)
    - get_heat_flux_element_first_five_nodes(nodes, hf_elements, index):
        Returns the first five node IDs (and their coordinates) for a given heat flux element.
    - get_heat_flux_element_normal_and_centroid(nodes, hf_elements, index):
        Computes the unit normal vector (based on nodes i, j, k) and the centroid
        (averaging all 8 nodes' coordinates) of the specified heat flux element.

Version History:
  Version 1.4:
    - Introduced node parsing with fallback to manual splitting if np.genfromtxt fails.
    - Added a function to retrieve the first five node IDs for a given element.
    - Printed progress while reading nodes and heat flux lines.
  Version 2.0:
    - Added get_heat_flux_element_normal_and_centroid function to compute:
        * Unit normal vector from i, j, k
        * Centroid by averaging all 8 node coordinates
    - Integrated everything into a single script.
    - Modified the heat flux parsing to stop after the first heat flux section is read.
"""

import numpy as np
import tkinter as tk
from tkinter import filedialog

VERSION = "2.0"

def select_file_dialog():
    """
    Opens a file dialog to select a .dat file.
    
    Returns:
      file_path (str): The full path of the selected .dat file.
    """
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()
    # Open file dialog for .dat files only
    file_path = filedialog.askopenfilename(
        title="Select ANSYS APDL .dat File",
        filetypes=[("DAT files", "*.dat")]
    )
    root.destroy()
    return file_path

def parse_ansys_apdl_optimized(file_path):
    """
    Optimized parser for an ANSYS APDL file extracting node coordinates and
    heat flux surface element connectivity.
    
    Assumptions:
      - The nodes section follows the marker:
          /com,*********** Nodes for the whole assembly ***********
        and each valid node line is in the format: node_number x y z
        where tokens are whitespace-delimited.
      - The heat flux section to be parsed starts after:
          /com,*********** Create "Heat Flux" ***********
        and includes header lines (starting with "et,", "eblock," or enclosed in parentheses)
        followed by data lines. Each data line is whitespace separated where:
          - Token 0: element ID
          - Tokens 5-12: eight nodal connectivity numbers.
        Once a line with only “-1” is encountered, the parser stops reading further lines.
    
    Returns:
      nodes_dict: dict mapping node_number (int) to (x, y, z) tuple (floats)
      hf_elements: list of dicts, each with:
                    'element_id': int,
                    'nodes': list of eight node numbers (ints)
    """
    nodes_lines = []
    heat_flux_lines = []
    section = None
    nodes_count = 0
    hf_count = 0

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for section markers.
            if '/com,*********** Nodes for the whole assembly ***********' in line:
                section = 'nodes'
                print("\nStarting to read nodes...")
                continue
            elif '/com,*********** Create "Heat Flux" ***********' in line:
                section = 'heat_flux'
                print("\nStarting to read heat flux data...")
                continue

            if section == 'nodes':
                # Skip lines starting with '/'.
                if not line.startswith('/'):
                    tokens = line.split()
                    if len(tokens) == 4:
                        try:
                            int(tokens[0])      # Validate node number
                            float(tokens[1])    # Validate x coordinate
                            float(tokens[2])    # Validate y coordinate
                            float(tokens[3])    # Validate z coordinate
                            nodes_lines.append(line)
                            nodes_count += 1
                            print(f"\rReading nodes: {nodes_count}", end='', flush=True)
                        except ValueError:
                            continue
            elif section == 'heat_flux':
                # Check if the line signals the end of the heat flux section.
                if line.strip() == "-1":
                    print("\nEnd of heat flux section reached.")
                    # Stop reading further lines.
                    break
                # Skip header lines starting with "et,2", "eblock," or those enclosed in parentheses.
                if (line.lower().startswith("et,2") or 
                    line.lower().startswith("eblock,") or 
                    (line.startswith('(') and line.endswith(')'))):
                    continue
                heat_flux_lines.append(line)
                hf_count += 1
                print(f"\rReading heat flux lines: {hf_count}", end='', flush=True)
        # Print a newline after the progress output.
        print()
    
    print(f"Total valid node lines collected: {len(nodes_lines)}")
    
    # --- Process nodes section using NumPy ---
    try:
        nodes_data = np.genfromtxt(nodes_lines, delimiter=None)
    except Exception as e:
        print(f"\nError using np.genfromtxt: {e}")
        print("Falling back to manual parsing...")
        try:
            nodes_data = np.array([list(map(float, line.split())) for line in nodes_lines])
        except Exception as inner_e:
            print(f"Fallback parsing failed: {inner_e}")
            nodes_data = np.empty((0, 4))  # Last resort: empty array

    if nodes_data.size == 0:
        print("No valid node data found.")
        nodes_dict = {}
    else:
        if nodes_data.ndim == 1:
            nodes_data = nodes_data.reshape((1, -1))
        node_ids = nodes_data[:, 0].astype(np.int64)
        coords = nodes_data[:, 1:]
        nodes_dict = {node_id: tuple(coord) for node_id, coord in zip(node_ids, coords)}

    # --- Process heat flux section ---
    hf_elements = []
    for line in heat_flux_lines:
        tokens = line.split()
        # Expecting at least 13 tokens: 5 header values + 8 nodal connectivity numbers.
        if len(tokens) >= 13:
            try:
                element_id = int(tokens[0])
                connectivity = [int(token) for token in tokens[5:13]]
                hf_elements.append({'element_id': element_id, 'nodes': connectivity})
            except ValueError:
                continue

    return nodes_dict, hf_elements

def get_heat_flux_element_first_five_nodes(nodes, hf_elements, index):
    """
    For a given heat flux element (0-indexed), returns a list of tuples
    containing the first five node numbers and their corresponding xyz coordinates.
    
    Parameters:
      nodes (dict): Mapping from node number to (x, y, z) coordinates.
      hf_elements (list): List of heat flux element dictionaries.
      index (int): The 0-indexed heat flux element.
    
    Returns:
      List of tuples: Each tuple is (node_number, (x, y, z)).
                      If a node's coordinate is not found, returns "Coordinate not found".
    
    Raises:
      IndexError: If the index is out of range.
    """
    if index < 0 or index >= len(hf_elements):
        raise IndexError("Heat flux element index out of range.")
    
    element = hf_elements[index]
    first_five_nodes = element['nodes'][:5]
    result = []
    for node in first_five_nodes:
        coordinate = nodes.get(node, "Coordinate not found")
        result.append((node, coordinate))
    return result

def get_heat_flux_element_normal_and_centroid(nodes, hf_elements, index):
    """
    Computes:
      1) The unit normal vector to the surface formed by the first three nodes
         (i, j, k) of the heat flux element using vectors IK x IJ.
      2) The centroid (point of interest) by averaging the coordinates of all 8 connectivity nodes.

    Parameters:
      nodes (dict): Mapping from node number to (x, y, z) coordinates.
      hf_elements (list): List of heat flux element dictionaries.
      index (int): The 0-indexed heat flux element.
    
    Returns:
      dict: { "normal": (nx, ny, nz) or None, "centroid": (x, y, z) or None }
    
    Raises:
      IndexError: If the index is out of range.
    """
    if index < 0 or index >= len(hf_elements):
        raise IndexError(f"Heat flux element index {index} is out of range.")

    element = hf_elements[index]
    node_ids = element['nodes']
    result = {"normal": None, "centroid": None}

    # Compute centroid from all nodes.
    if len(node_ids) >= 1:
        sum_x = sum_y = sum_z = 0.0
        valid_count = 0
        for nid in node_ids:
            coord = nodes.get(nid, None)
            if coord is None:
                continue
            sum_x += coord[0]
            sum_y += coord[1]
            sum_z += coord[2]
            valid_count += 1
        if valid_count > 0:
            result["centroid"] = (sum_x / valid_count, sum_y / valid_count, sum_z / valid_count)

    # Compute normal using the first three nodes (i, j, k).
    if len(node_ids) >= 3:
        i_id, j_id, k_id = node_ids[0], node_ids[1], node_ids[2]
        i_coord = nodes.get(i_id, None)
        j_coord = nodes.get(j_id, None)
        k_coord = nodes.get(k_id, None)
        if i_coord and j_coord and k_coord:
            i_coord = np.array(i_coord, dtype=float)
            j_coord = np.array(j_coord, dtype=float)
            k_coord = np.array(k_coord, dtype=float)
            ik_vec = k_coord - i_coord
            ij_vec = j_coord - i_coord
            normal = np.cross(ik_vec, ij_vec)
            norm_mag = np.linalg.norm(normal)
            if norm_mag >= 1e-12:
                result["normal"] = tuple(normal / norm_mag)

    return result

def main():
    print(f"ANSYS APDL Parser (Version {VERSION})")
    file_path = select_file_dialog()
    if not file_path:
        print("No file selected. Exiting.")
        return

    nodes, hf_elements = parse_ansys_apdl_optimized(file_path)

    print(f"\nTotal nodes read: {len(nodes)}")
    print(f"Total heat flux elements read: {len(hf_elements)}\n")
    
    # Example: Display connectivity and coordinates for the first 5 heat flux elements.
    for elem in hf_elements[:5]:
        element_id = elem['element_id']
        connectivity = elem['nodes']
        coords = [nodes.get(node, "Coordinate not found") for node in connectivity]
        print(f"Element ID: {element_id}")
        print(f"  Connectivity nodes: {connectivity}")
        print(f"  Coordinates: {coords}\n")
    
    # Demonstration of get_heat_flux_element_normal_and_centroid:
    if hf_elements:
        index = 0  # Demonstrate for the first heat flux element
        info = get_heat_flux_element_normal_and_centroid(nodes, hf_elements, index)
        normal = info["normal"]
        centroid = info["centroid"]
        print("===============================================\n")
        print(f"Element #{index} Normal Vector: {normal}")
        print(f"Element #{index} Centroid: {centroid}\n")

    # Example usage of get_heat_flux_element_first_five_nodes:
    if hf_elements:
        try:
            index = 0  # Change as needed
            first_five = get_heat_flux_element_first_five_nodes(nodes, hf_elements, index)
            print(f"First 5 nodes for the {index}-th heat flux element:")
            for node, coord in first_five:
                print(f"  Node {node}: {coord}")
        except IndexError as e:
            print(e)

if __name__ == '__main__':
    main()
