#v1.00
import tkinter as tk
from tkinter import ttk
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import tkinter.filedialog
import csv
import pandas as pd
from tkinter import messagebox


# List to store coils, each coil will be a dictionary
coils = []

# Global dictionary to store the V responses for each channel
channel_responses = {}



# Biot-Savart Law function
def biot_savart(px, py, pz, coil_x, coil_y, coil_z, current, subdivisions=100):  # increase subdivisions to enhance bio-savart precision
   
    Hx, Hy, Hz = 0.0, 0.0, 0.0
    epsilon = 1e-15  # Small constant to prevent division by zero

    for i in range(len(coil_x) - 1):
        # Divide each segment into smaller segments
        segment_x = np.linspace(coil_x[i], coil_x[i + 1], subdivisions, endpoint=False)
        segment_y = np.linspace(coil_y[i], coil_y[i + 1], subdivisions, endpoint=False)
        segment_z = np.linspace(coil_z[i], coil_z[i + 1], subdivisions, endpoint=False)

        for j in range(subdivisions - 1):
            # Differential length vector (dl) for the small segment
            dl_x = segment_x[j + 1] - segment_x[j]
            dl_y = segment_y[j + 1] - segment_y[j]
            dl_z = segment_z[j + 1] - segment_z[j]

            # Position vector (r) from the segment to the point
            r_x = px - segment_x[j]
            r_y = py - segment_y[j]
            r_z = pz - segment_z[j]
            r = np.sqrt(r_x ** 2 + r_y ** 2 + r_z ** 2) + epsilon

            # Cross product dl x r
            cross_x = dl_y * r_z - dl_z * r_y
            cross_y = dl_z * r_x - dl_x * r_z
            cross_z = dl_x * r_y - dl_y * r_x

            # Contribution of this small segment to the magnetic field
            dHx = current / (4 * np.pi * r ** 3) * cross_x
            dHy = current / (4 * np.pi * r ** 3) * cross_y
            dHz = current / (4 * np.pi * r ** 3) * cross_z

            # Summing up the contributions from each small segment
            Hx += dHx
            Hy += dHy
            Hz += dHz

    # The last segment from the end point of the coil to the start point
    # to ensure the coil is closed if it is supposed to be
    if np.array_equal([coil_x[0], coil_y[0], coil_z[0]], [coil_x[-1], coil_y[-1], coil_z[-1]]) is False:
        # Compute the final segment's contribution if the coil is not closed
        dl_x = coil_x[0] - coil_x[-1]
        dl_y = coil_y[0] - coil_y[-1]
        dl_z = coil_z[0] - coil_z[-1]
        r_x = px - coil_x[-1]
        r_y = py - coil_y[-1]
        r_z = pz - coil_z[-1]
        r = np.sqrt(r_x ** 2 + r_y ** 2 + r_z ** 2) + epsilon
        cross_x = dl_y * r_z - dl_z * r_y
        cross_y = dl_z * r_x - dl_x * r_z
        cross_z = dl_x * r_y - dl_y * r_x
        Hx += current / (4 * np.pi * r ** 3) * cross_x
        Hy += current / (4 * np.pi * r ** 3) * cross_y
        Hz += current / (4 * np.pi * r ** 3) * cross_z

    return Hx, Hy, Hz

# Function to generate a coil based on input coordinates
def generate_coil():
    # Get coil name and number of corners
    name = coil_name_entry.get()
    corners = int(num_corners_entry.get())
    
    # Get the coordinates from the text area
    coordinates = coordinates_text.get("1.0", tk.END).strip().split("\n")
    coil_points = np.array([list(map(float, line.split())) for line in coordinates])
    coil_points = np.vstack((coil_points, coil_points[0]))  # Close the coil
    
    # Add coil to list
    coils.append({'name': name, 'points': coil_points})
    refresh_coil_listboxes()  # Update the coil selection listboxes
    
    # Update the 3D plot with the new coil
    ax.clear()
    for coil in coils:
        ax.plot(coil['points'][:, 0], coil['points'][:, 1], coil['points'][:, 2], label=coil['name'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Metallic Coils in 3D Space')
    set_axes_equal(ax)
    ax.legend()
    canvas.draw()
    print(f"Coil generated: {name}")

def clear_coil_data():
    # Clear input fields
    coil_name_entry.delete(0, tk.END)
    num_corners_entry.delete(0, tk.END)
    coordinates_text.delete('1.0', tk.END)

    # Clear the 3D plot
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Metallic Coils in 3D Space')
    canvas.draw()

    # Reset the coils list
    coils.clear()

    # Refresh the listboxes in the coil selection tab
    refresh_coil_listboxes()

    
# Function to refresh Listboxes in the coil selection tab
def refresh_coil_listboxes():
    transmit_coil_listbox.delete(0, tk.END)
    receive_coil_listbox.delete(0, tk.END)
    for coil in coils:
        transmit_coil_listbox.insert(tk.END, coil['name'])
        receive_coil_listbox.insert(tk.END, coil['name'])
        


def generate_channels():
    selected_transmit_coil = transmit_coil_listbox.get(tk.ANCHOR)
    selected_receive_coils = [receive_coil_listbox.get(idx) for idx in receive_coil_listbox.curselection()]

    if not selected_transmit_coil or not selected_receive_coils:
        print("Please select a transmit coil and at least one receive coil.")
        return

    # Add new channels without clearing the existing ones
    for rcv_coil in selected_receive_coils:
        channel = f"TX: {selected_transmit_coil}, RX: {rcv_coil}"
        # Check if the channel already exists to avoid duplicates
        if channel not in channels_listbox.get(0, tk.END):
            channels_listbox.insert(tk.END, channel)

def clear_channels():
    channels_listbox.delete(0, tk.END)
    
# Function to perform forward modeling on selected coils    
def forward_modeling():
    global channel_responses, distances
    channel_responses = {}
    distances = []
    # Assuming the current is the same for all coils
    current = float(current_entry.get())

    # Retrieve the start and end points from the user input and split into X, Y, Z components
    start_x, start_y, start_z = map(float, start_point_entry.get().split())
    end_x, end_y, end_z = map(float, end_point_entry.get().split())

    # Number of points in the trajectory
    num_points = int(num_points_entry.get())

    # Retrieve the M matrix values from the user input
    M = np.array([[float(matrix_entries[0][0].get()), float(matrix_entries[0][1].get()), float(matrix_entries[0][2].get())],
                  [float(matrix_entries[1][0].get()), float(matrix_entries[1][1].get()), float(matrix_entries[1][2].get())],
                  [float(matrix_entries[2][0].get()), float(matrix_entries[2][1].get()), float(matrix_entries[2][2].get())]])

    # Generate the trajectory as a straight line between the start and end points
    x_positions = np.linspace(start_x, end_x, num_points)
    y_positions = np.linspace(start_y, end_y, num_points)
    z_positions = np.linspace(start_z, end_z, num_points)

    # Calculate the linear distance for each point on the trajectory
    distances = np.zeros(num_points)
    for i in range(1, num_points):
        distances[i] = distances[i-1] + np.sqrt((x_positions[i] - x_positions[i-1])**2 +
                                                (y_positions[i] - y_positions[i-1])**2 +
                                                (z_positions[i] - z_positions[i-1])**2)

    # Clear the existing plot
    forward_modeling_ax.clear()

    # Iterate through each channel and plot V responses
    for idx in range(channels_listbox.size()):
        channel = channels_listbox.get(idx)
        tx_name, rx_name = channel.replace('TX: ', '').replace('RX: ', '').split(', ')
        
        transmit_coil = next((coil for coil in coils if coil['name'] == tx_name), None)
        receive_coil = next((coil for coil in coils if coil['name'] == rx_name), None)

        if transmit_coil is None or receive_coil is None:
            print("Error in channel selection.")
            continue

        V_responses = []
        for x, y, z in zip(x_positions, y_positions, z_positions):
            H_TX = np.array(biot_savart(x, y, z, *transmit_coil['points'].T, current))
            H_RX = np.array(biot_savart(x, y, z, *receive_coil['points'].T, current))

            V_response = M[0, 0] * (H_TX[0] * H_RX[0]) + \
                         M[0, 1] * (H_TX[1] * H_RX[0] + H_TX[0] * H_RX[1]) + \
                         M[0, 2] * (H_TX[2] * H_RX[0] + H_TX[0] * H_RX[2]) + \
                         M[1, 1] * (H_TX[1] * H_RX[1]) + \
                         M[1, 2] * (H_TX[2] * H_RX[1] + H_TX[1] * H_RX[2]) + \
                         M[2, 2] * (H_TX[2] * H_RX[2])

            V_responses.append(V_response)
            # Store the responses in the global dictionary
            # In forward_modeling, when calculating V_responses, include coordinates:
            channel_responses[channel] = [(x, y, z, V_response) for x, y, z, V_response in zip(x_positions, y_positions, z_positions, V_responses)]

            
            
        # Plot the V responses for this channel
        forward_modeling_ax.plot(distances, V_responses, '-o', label=f'{channel}')
        
    # Plot coils and trajectory in the 3D plot
    forward_modeling_3d_ax.clear()
    for coil in coils:
        forward_modeling_3d_ax.plot(coil['points'][:, 0], coil['points'][:, 1], coil['points'][:, 2], label=coil['name'])
    
    # Plot the trajectory line
    forward_modeling_3d_ax.plot([start_x, end_x], [start_y, end_y], [start_z, end_z], 'r--', label='Trajectory')
    forward_modeling_3d_ax.text(start_x, start_y, start_z, 'Start', color='green')
    forward_modeling_3d_ax.text(end_x, end_y, end_z, 'End', color='blue')

    # Set labels and title for the 3D plot
    forward_modeling_3d_ax.set_xlabel('X')
    forward_modeling_3d_ax.set_ylabel('Y')
    forward_modeling_3d_ax.set_zlabel('Z')
    forward_modeling_3d_ax.set_title('Coils and Trajectory')
    forward_modeling_3d_ax.legend()
    set_axes_equal(forward_modeling_3d_ax)
    forward_modeling_3d_canvas.draw()    

    # Update plot settings
    forward_modeling_ax.set_xlabel('Distance Along Trajectory (m)')
    forward_modeling_ax.set_ylabel('V Response')
    forward_modeling_ax.set_title('V Responses Along Trajectory')
    forward_modeling_ax.legend()
    forward_modeling_canvas.draw()
    
def set_axes_equal(ax):
    
   # Make axes of 3D plot have equal scale

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def save_coil_data():
    # Get the coordinates from the text area
    coordinates = coordinates_text.get("1.0", tk.END).strip()

    # Open a Save As dialog to let the user specify the file name and location
    filename = tk.filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Save coil data as..."
    )

    if filename:  # Check if a filename was selected
        # Save to CSV file
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for line in coordinates.split('\n'):
                writer.writerow(line.split())

        print(f"Coil data saved to {filename}")


def load_coil_data():
    # Ask the user for a file to open
    filename = tk.filedialog.askopenfilename(
        title="Select coil data file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )

    if filename:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            coordinates = '\n'.join(' '.join(row) for row in reader)

        # Update the UI with the loaded data
        coil_name_entry.delete(0, tk.END)
        coil_name_entry.insert(0, filename.split('_')[0])  # Assumes coil name is part of the filename
        coordinates_text.delete('1.0', tk.END)
        coordinates_text.insert(tk.END, coordinates)

        print(f"Loaded coil data from {filename}")    
        
def save_all_coils_data():
    filename = tk.filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Save all coils data as..."
    )
    
    if filename:  # Check if a filename was selected
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            for coil in coils:
                writer.writerow([coil['name']])  # Write coil name
                for point in coil['points']:
                    writer.writerow(point)  # Write coil points
                writer.writerow([])  # Add a blank row after each coil for separation

        print(f"All coil data saved to {filename}")

def load_all_coils_data():
    filename = tk.filedialog.askopenfilename(
        title="Select all coils data file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )

    if filename:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            coils.clear()  # Clear existing coils
            coil_data = []
            coil_name = None
            for row in reader:
                if not row:
                    if coil_data and coil_name:
                        # Convert collected points and add to coils list
                        coil_points = np.array(coil_data, dtype=float)
                        coils.append({'name': coil_name, 'points': coil_points})
                        coil_data = []
                        coil_name = None
                elif coil_name is None:
                    coil_name = row[0]  # First non-empty row is the coil name
                else:
                    coil_data.append(row)  # Collect coil point data

            if coil_data and coil_name:
                # Add the last coil if file does not end with a blank row
                coil_points = np.array(coil_data, dtype=float)
                coils.append({'name': coil_name, 'points': coil_points})

        refresh_coil_listboxes()  # Refresh UI with new coil data
        coil_name_entry.delete(0, tk.END)
        num_corners_entry.delete(0, tk.END)
        coordinates_text.delete('1.0', tk.END)
        ax.clear()
        for coil in coils:
            ax.plot(coil['points'][:, 0], coil['points'][:, 1], coil['points'][:, 2], label=coil['name'])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Metallic Coils in 3D Space')
        set_axes_equal(ax)
        ax.legend()
        canvas.draw()   
        print(f"Loaded all coil data from {filename}")
    

# Create the main window for the application
root = tk.Tk()
root.title("Electromagnetic Sensing Simulator")

# Create a tabbed interface within the main window
notebook = ttk.Notebook(root)
notebook.pack(padx=10, pady=10)

# Coil generation tab
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Generate Coils")

# Add widgets for coil properties
tk.Label(tab1, text="Coil Name:").grid(row=0, column=0)
coil_name_entry = tk.Entry(tab1)
coil_name_entry.grid(row=0, column=1)
coil_name_entry.insert(0, "coil2")

tk.Label(tab1, text="Number of Corners:").grid(row=1, column=0)
num_corners_entry = tk.Entry(tab1)
num_corners_entry.grid(row=1, column=1)
num_corners_entry.insert(0, "8")

tk.Label(tab1, text="Coordinates (X Y Z):").grid(row=2, column=0)
coordinates_text = tk.Text(tab1, height=5, width=30)
coordinates_text.grid(row=2, column=1)
#default_coordinates = "0 0 1\n1 0 1\n1 1 1\n0 1 1"
default_coordinates = "0 0 -1\n1 0 -1\n1 0.4 -1\n0 0.6 -1\n0 1 -1\n1 1 -1\n1 0.6 -1\n0 0.4 -1"
coordinates_text.insert(tk.END, default_coordinates)

# Button to create the coils
generate_button = tk.Button(tab1, text="Generate Coil", command=generate_coil)
generate_button.grid(row=3, column=0, columnspan=2)

# Create a Matplotlib figure for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=tab1)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=4, column=0, columnspan=2)

# Button to clear the coil data
clear_button = tk.Button(tab1, text="Clear Coil Data", command=clear_coil_data)
clear_button.grid(row=3, column=1, columnspan=2)

# Add buttons for saving and loading coil data
save_button = tk.Button(tab1, text="Save Coil Data", command=save_coil_data)
save_button.grid(row=1, column=20, columnspan=2)

load_button = tk.Button(tab1, text="Load Coil Data", command=load_coil_data)
load_button.grid(row=2, column=20, columnspan=2)

# Button to save all coil data
save_all_coils_button = tk.Button(tab1, text="Save All Coil Data", command=save_all_coils_data)
save_all_coils_button.grid(row=1, column=30, columnspan=2)

# Button to load all coil data
load_all_coils_button = tk.Button(tab1, text="Load All Coil Data", command=load_all_coils_data)
load_all_coils_button.grid(row=2, column=30, columnspan=2)

# COIL SELECTION TAB
coil_selection_tab = ttk.Frame(notebook)
notebook.add(coil_selection_tab, text="Coil Selection")
# Disclaimer Label
discl="     To define a channel: \n     First select Transmit Coil, then select Receive Coil \n     then click Generate Channels"
disclaimer_label = tk.Label(coil_selection_tab, text=discl, justify=tk.LEFT)
disclaimer_label.grid(row=2, column=3, columnspan=2, sticky="w", pady=(5, 5))

# Transmit Coils Listbox
tk.Label(coil_selection_tab, text="Transmit Coils").grid(row=0, column=0)
transmit_coil_listbox = tk.Listbox(coil_selection_tab, selectmode=tk.SINGLE)
transmit_coil_listbox.grid(row=1, column=0)

# Receive Coils Listbox
tk.Label(coil_selection_tab, text="Receive Coils").grid(row=0, column=1)
receive_coil_listbox = tk.Listbox(coil_selection_tab, selectmode=tk.SINGLE)
receive_coil_listbox.grid(row=1, column=1)

# Channels Listbox
tk.Label(coil_selection_tab, text="Channels").grid(row=2, column=0, columnspan=2)
channels_listbox = tk.Listbox(coil_selection_tab, selectmode=tk.SINGLE)
channels_listbox.grid(row=3, column=0, columnspan=2)

# Button to generate channels
generate_channels_button = tk.Button(coil_selection_tab, text="Generate Channels", command=generate_channels)
generate_channels_button.grid(row=4, column=0, columnspan=2)

# Button to clear channels
clear_channels_button = tk.Button(coil_selection_tab, text="Clear Channels", command=clear_channels)
clear_channels_button.grid(row=5, column=0, columnspan=2)

# Forward modeling tab
forward_tab = ttk.Frame(notebook)
notebook.add(forward_tab, text="Forward Modeling")

# Add label and entry field for the current
tk.Label(forward_tab, text="Current (Amps):").grid(row=0, column=0)
current_entry = tk.Entry(forward_tab)
current_entry.grid(row=0, column=1)
current_entry.insert(0, "1")  # Set the default value for current as specified

# Modify labels and entry fields for trajectory parameters
tk.Label(forward_tab, text="Start Point (X Y Z):").grid(row=1, column=0)
start_point_entry = tk.Entry(forward_tab)
start_point_entry.grid(row=1, column=1)
start_point_entry.insert(0, "0 -1.2 -0.8")  # Set the default start point as specified

tk.Label(forward_tab, text="End Point (X Y Z):").grid(row=2, column=0)
end_point_entry = tk.Entry(forward_tab)
end_point_entry.grid(row=2, column=1)
end_point_entry.insert(0, "1 2.2 0.8")    # Set the default end point as specified

tk.Label(forward_tab, text="Number of Points:").grid(row=3, column=0)
num_points_entry = tk.Entry(forward_tab)
num_points_entry.grid(row=3, column=1)
num_points_entry.insert(0, "500")     # Set the default number of points as specified

# Add matrix entry labels and fields for the M matrix
tk.Label(forward_tab, text="M Matrix:").grid(row=4, column=0, columnspan=4)
matrix_entries = [[tk.Entry(forward_tab, width=5) for _ in range(3)] for _ in range(3)]
default_M_values = [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]]  # Default M matrix values

for i in range(3):
    for j in range(3):
        matrix_entries[i][j].grid(row=i+5, column=j)
        matrix_entries[i][j].insert(0, default_M_values[i][j])

# Button to execute forward modeling
forward_modeling_button = tk.Button(forward_tab, text="Execute Forward Modeling", command=forward_modeling)
forward_modeling_button.grid(row=8, column=0, columnspan=4)

# Create a figure and axes for the V response plot
forward_modeling_fig = plt.figure(figsize=(5, 4))
forward_modeling_ax = forward_modeling_fig.add_subplot(111)
forward_modeling_canvas = FigureCanvasTkAgg(forward_modeling_fig, master=forward_tab)
forward_modeling_widget = forward_modeling_canvas.get_tk_widget()
forward_modeling_widget.grid(row=9, column=0, columnspan=4)

# Create a figure and axes for the 3D plot of coils and trajectory in the forward modeling tab
forward_modeling_3d_fig = plt.figure(figsize=(5, 4))
forward_modeling_3d_ax = forward_modeling_3d_fig.add_subplot(111, projection='3d')
forward_modeling_3d_canvas = FigureCanvasTkAgg(forward_modeling_3d_fig, master=forward_tab)
forward_modeling_3d_widget = forward_modeling_3d_canvas.get_tk_widget()
forward_modeling_3d_widget.grid(row=9, column=10, columnspan=4)

def save_response_data():
    filename = tk.filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Save response data as..."
    )

    if filename:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Y", "Z", "Channel", "Response"])
            
            for channel, data in channel_responses.items():
                for x, y, z, response in data:
                    writer.writerow([x, y, z, channel, response])

            print(f"Response data saved to {filename}")
    else:
        print("Save operation cancelled")

#  button in Tab 3 for saving the response data
save_response_data_button = tk.Button(forward_tab, text="Save Response Data", command=save_response_data)
save_response_data_button.grid(row=10, column=0, columnspan=4)


#________________________________________________________________________________________________________
# INVERSE MODELLING TAB

# Function to display coils
def display_coils():
    # Clear any existing plots
    inverse_modeling_3d_ax.clear()

    # Plot each coil
    for coil in coils:
        inverse_modeling_3d_ax.plot(coil['points'][:, 0], coil['points'][:, 1], coil['points'][:, 2], label=coil['name'])

    # Set labels and title for the 3D plot
    inverse_modeling_3d_ax.set_xlabel('X')
    inverse_modeling_3d_ax.set_ylabel('Y')
    inverse_modeling_3d_ax.set_zlabel('Z')
    inverse_modeling_3d_ax.set_title('Coils in 3D Space')
    set_axes_equal(inverse_modeling_3d_ax)
    inverse_modeling_3d_ax.legend()
    inverse_modeling_3d_canvas.draw()

def load_v_responses():
    filename = tk.filedialog.askopenfilename(
        title="Select V response data file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    if not filename:
        print("No file selected")
        return None
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)
    
    # Select only the columns 'X', 'Y', 'Z', and 'Response'
    v_responses = df[['X', 'Y', 'Z', 'Response']]
    return v_responses

def solve_for_m_matrix(v_responses, current, regularization_param):
    # Initialize the A matrix and the b vector
    A = []
    b = v_responses['Response'].values
    
    # Calculate the A matrix based on the interaction terms
    for index, row in v_responses.iterrows():
        H = np.zeros((3, 2))  # 2 coils: transmit and receive
        for idx, coil in enumerate(coils):
            H[:, idx] = biot_savart(row['X'], row['Y'], row['Z'], *coil['points'].T, current)
        
        # Calculate the interaction terms for the A matrix
        interaction_terms = [
            H[0, 0] * H[0, 1],  # Htxx * Hrxx
            H[0, 0] * H[1, 1] + H[1, 0] * H[0, 1],  # Htxx * Hrxy + Htxy * Hrxx
            H[0, 0] * H[2, 1] + H[2, 0] * H[0, 1],  # Htxx * Hrxz + Htxz * Hrxx
            H[1, 0] * H[1, 1],  # Htxy * Hrxy
            H[1, 0] * H[2, 1] + H[2, 0] * H[1, 1],  # Htxy * Hrxz + Htxz * Hrxy
            H[2, 0] * H[2, 1],  # Htxz * Hrxz
        ]
        A.append(interaction_terms)
    
    A = np.array(A)
    
    # Solve for M matrix using least squares
    M_vector, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Print the norm of the residuals
    norm_of_residuals = np.linalg.norm(residuals)
    print(f"Norm of residuals (without regularization): {norm_of_residuals}")
    
    # Solve for M matrix using least squares with regularization
    regularization_matrix = regularization_param * np.identity(A.shape[1])
    A_augmented = np.vstack([A, regularization_matrix])
    b_augmented = np.concatenate([b, np.zeros(A.shape[1])])

    M_vector_reg, residuals_reg, rank_reg, s_reg = np.linalg.lstsq(A_augmented, b_augmented, rcond=None)
    norm_of_residuals_reg = np.linalg.norm(residuals_reg)
    print(f"Norm of residuals (with regularization): {norm_of_residuals_reg}")
    
    # Reconstruct the M matrix from the vector
    M_matrix = np.array([
        [M_vector[0], M_vector[1], M_vector[2]],
        [M_vector[1], M_vector[3], M_vector[4]],
        [M_vector[2], M_vector[4], M_vector[5]]
    ])
    M_matrix_reg = np.array([
        [M_vector_reg[0], M_vector_reg[1], M_vector_reg[2]],
        [M_vector_reg[1], M_vector_reg[3], M_vector_reg[4]],
        [M_vector_reg[2], M_vector_reg[4], M_vector_reg[5]]
    ])

    return M_matrix, M_matrix_reg


def import_and_compute_m_matrix():
    # Use the function to load the V responses
    v_responses = load_v_responses()
    if v_responses is None:
        messagebox.showerror("Error", "No file was loaded.")
        return
    
    # Prompt for current value
    current = tk.simpledialog.askfloat("Input", "Please enter the current value (Amps):",
                                    minvalue=0.0, maxvalue=100.0)
    if current is None:
        messagebox.showerror("Error", "No current value was provided.")
        return
    
    # Fetch regularization parameter from entry
    try:
        regularization_param = float(reg_param_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Invalid regularization parameter.")
        return

    # Compute the M matrix
    M_matrix, M_matrixreg = solve_for_m_matrix(v_responses, current, regularization_param)
    
    # Display the M matrix
    display_m_matrix(M_matrix, M_matrixreg)
    
    # Plot response data
    plot_v_response(v_responses)

def display_m_matrix(M_matrix, M_matrixreg):
    # Clear any existing data in the display area
    m_matrix_display.delete('1.0', tk.END)
    m_matrix_displayregularised.delete('1.0', tk.END)
    # Convert the matrix to a string and insert it into the Text widget
    M_matrix_str = '\n'.join(['\t'.join([f"{item:.4f}" for item in row]) for row in M_matrix])
    m_matrix_display.insert(tk.END, M_matrix_str)
    
    # Convert the regularised matrix to a string and insert it into the Text widget
    M_matrixreg_str = '\n'.join(['\t'.join([f"{item:.4f}" for item in row]) for row in M_matrixreg])
    m_matrix_displayregularised.insert(tk.END, M_matrixreg_str)


# Function to plot V response
def plot_v_response(v_responses):
    # Clear any existing plots
    v_response_ax.clear()
    
    # Plot the V responses
    v_response_ax.plot(range(len(v_responses)), v_responses['Response'], '-o', label='V Response')
    
    # Set labels and title for the plot
    v_response_ax.set_xlabel('Data Point Index')
    v_response_ax.set_ylabel('V Response')
    v_response_ax.set_title('V Responses Along Trajectory')
    v_response_ax.legend()
    v_response_canvas.draw()

    
    
# GUI for last (inverse) tab   
inverse_tab = ttk.Frame(notebook)
notebook.add(inverse_tab, text="Inverse Modeling")
    
# Button to display coils
display_coils_button = tk.Button(inverse_tab, text="Update Coils", command=display_coils)
display_coils_button.pack(side=tk.LEFT)

# Entry for regularization parameter
reg_param_label = tk.Label(inverse_tab, text="Regularization Param:")
reg_param_label.pack()
reg_param_entry = tk.Entry(inverse_tab)
reg_param_entry.pack()
reg_param_entry.insert(0, "0.00007")  # Default value

# Create a figure and axes for the 3D plot of coils in the inverse modeling tab
inverse_modeling_3d_fig = plt.figure(figsize=(4, 3))
inverse_modeling_3d_ax = inverse_modeling_3d_fig.add_subplot(111, projection='3d')
inverse_modeling_3d_canvas = FigureCanvasTkAgg(inverse_modeling_3d_fig, master=inverse_tab)
inverse_modeling_3d_widget = inverse_modeling_3d_canvas.get_tk_widget()
inverse_modeling_3d_widget.pack()

# Create a figure and axes for the V response plot in the inverse modeling tab
v_response_fig = plt.figure(figsize=(4, 3))
v_response_ax = v_response_fig.add_subplot(111)
v_response_canvas = FigureCanvasTkAgg(v_response_fig, master=inverse_tab)
v_response_widget = v_response_canvas.get_tk_widget()

# Button to import V responses and compute M matrix
import_button = tk.Button(inverse_tab, text="Import V Responses and Compute M Matrix",command=import_and_compute_m_matrix)
import_button.pack()

# Text widget to display the M matrix
m_matrix_display = tk.Text(inverse_tab, height=5, width=35)
m_matrix_display.pack()

# Text widget to display the M matrix regularised
regularisedheader = tk.Text(inverse_tab, height=1, width=35)
regularisedheader.insert("1.0", "After Tikhinov regularisation")
regularisedheader.pack()
m_matrix_displayregularised = tk.Text(inverse_tab, height=5, width=35)
m_matrix_displayregularised.pack()

# Layout of the plots 
plots_frame = tk.Frame(inverse_tab)
plots_frame.pack()
inverse_modeling_3d_widget.pack(side=tk.LEFT)
v_response_widget.pack(side=tk.RIGHT)


#________________________________________________________________________________________________________
# VISUALISER TAB

# Function to import coil geometry for visualization
def import_coil_geometry():
    filename = tk.filedialog.askopenfilename(
        title="Select coil geometry file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    if not filename:
        print("No file selected")
        return None
    
    # Load coil geometry from the CSV file
    global coil_points  # Use a global variable to store the coil's points
    coil_points = np.loadtxt(filename, delimiter=',')
    
    print(f"Loaded coil geometry from {filename}")
    messagebox.showinfo("Info", "Coil geometry loaded successfully.")

# Modified display_magnetic_field_heatmaps_log_scale function to work with one coil
def display_magnetic_field_heatmaps_log_scale_for_one_coil():
    global coil_points  # Assume coil_points is defined globally
    
    # Ensure the coil geometry is loaded
    if 'coil_points' not in globals() or coil_points.size == 0:
        messagebox.showerror("Error", "Please import coil geometry first.")
        return

    grid_size = 50  # Grid resolution for the heatmap
    extent_range = 3  # Spatial extent range for the calculations
    space_range = np.linspace(-extent_range, extent_range, grid_size)
    
    # Initialize arrays for magnetic field magnitudes in all planes
    B_magnitude_xy = np.zeros((grid_size, grid_size))
    B_magnitude_yz = np.zeros((grid_size, grid_size))
    B_magnitude_xz = np.zeros((grid_size, grid_size))
    
    # Retrieve the current value from a user input (make sure this variable is properly defined or retrieved from your GUI)
    current = float(current_entry.get())  # Assuming current_entry is a Tkinter Entry widget for input

    # Calculate the magnetic field components for the coil in all planes
    for i, x in enumerate(space_range):
        for j, y in enumerate(space_range):
            # Magnetic field for the XY plane
            Bx, By, Bz = biot_savart(x, y, 0, coil_points[:, 0], coil_points[:, 1], coil_points[:, 2], current)
            B_magnitude_xy[i, j] = np.sqrt(Bx**2 + By**2 + Bz**2)
            
            # Magnetic field for the YZ plane
            Bx, By, Bz = biot_savart(0, y, x, coil_points[:, 0], coil_points[:, 1], coil_points[:, 2], current)
            B_magnitude_yz[i, j] = np.sqrt(Bx**2 + By**2 + Bz**2)
            
            # Magnetic field for the XZ plane
            Bx, By, Bz = biot_savart(x, 0, y, coil_points[:, 0], coil_points[:, 1], coil_points[:, 2], current)
            B_magnitude_xz[i, j] = np.sqrt(Bx**2 + By**2 + Bz**2)

    # Create a figure for the heatmaps
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot heatmaps for each plane
    titles = ['X vs Y Plane', 'Y vs Z Plane', 'Z vs X Plane']
    for ax, B_magnitude, title in zip(axs, [B_magnitude_xy, B_magnitude_yz, B_magnitude_xz], titles):
        img = ax.imshow(B_magnitude, cmap='inferno', origin='lower', extent=[-extent_range, extent_range, -extent_range, extent_range], norm=mcolors.LogNorm())
        ax.set_title(title)

    # Add a colorbar to the figure
    fig.colorbar(img, ax=axs.ravel().tolist(), orientation='vertical', label='Magnetic Field Intensity Magnitude (log scale)')

    # Embed the figure into the Tkinter GUI
    canvas = FigureCanvasTkAgg(fig, master=visualizer_tab) 
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# Add a new tab for the visualizer
visualizer_tab = ttk.Frame(notebook)
notebook.add(visualizer_tab, text="Visualiser")

# Labeled entry for current input
tk.Label(visualizer_tab, text="Current (Amps):").pack(pady=(10, 0))
current_entry_visualiser = tk.Entry(visualizer_tab)
current_entry_visualiser.pack(pady=(0, 10))
current_entry_visualiser.insert(0, "1")  # Default value

# Button to import coil geometry
import_coil_button = tk.Button(visualizer_tab, text="Import Coil Geometry", command=import_coil_geometry)
import_coil_button.pack(pady=10)

# Button to display magnetic field heatmaps
display_heatmaps_button = tk.Button(visualizer_tab, text="Display Magnetic Field Heatmaps", command=display_magnetic_field_heatmaps_log_scale_for_one_coil)
display_heatmaps_button.pack(pady=10)



root.mainloop() 
