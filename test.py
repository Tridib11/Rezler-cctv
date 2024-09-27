import cv2
import tkinter as tk
from tkinter import Label, Entry, Button, StringVar, Toplevel, Listbox, messagebox
from PIL import Image, ImageTk
import paho.mqtt.client as mqtt
import os
import time
import socket
import threading
import pyaudio
import wave
import numpy as np

# MQTT Configuration
MQTT_BROKER = 'localhost'
MQTT_PORT = 1883
MQTT_TOPIC = 'camera/control'

# Global variables
rtsp_url = ""
recording = False
scanning = False
out = None
audio_stream = None
frames = []

# Audio configuration
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
AMPLIFICATION_FACTOR = 2.0  # Increase this to make the audio louder

# Load YOLOv4 model for object detection
weights_path = 'yolov4.weights'
config_path = 'yolov4.cfg'
names_path = 'coco.names'

# Load YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class labels
with open(names_path, 'r') as f:
    CLASSES = f.read().strip().split('\n')

# Define random colors for each class label
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def start_stream():
    global recording, scanning, out, audio_stream, frames
    # Initialize the video capture object
    cap = cv2.VideoCapture(rtsp_url)

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if scanning:
                frame = perform_object_detection(frame)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            video_label.frame = frame  # Store the frame for capturing

            if recording and out is not None:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Write frame to video file

        video_label.after(10, update_frame)

    def perform_object_detection(frame):
        # Get frame dimensions
        (H, W) = frame.shape[:2]
        
        # Convert the frame to blob for object detection
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        
        # Get the output layer names
        ln = net.getUnconnectedOutLayersNames()
        
        # Forward pass
        layer_outputs = net.forward(ln)
        
        boxes = []
        confidences = []
        class_ids = []

        # Loop over each detection
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter out weak detections
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    # Calculate the top-left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        
        # Ensure at least one detection exists
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                # Draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"{CLASSES[class_ids[i]]}: {int(confidences[i] * 100)}%"
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def capture_image():
        # Capture the current frame from the stream
        if hasattr(video_label, 'frame'):
            frame = video_label.frame
            # Create directory if it doesn't exist
            save_dir = "captured_images"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Generate a unique filename based on the current time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"captured_image_{timestamp}.png")
            
            # Save the image
            cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"Image captured and saved as '{filename}'.")

    def start_recording():
        global recording, out, audio_stream, frames
        recording = True
        # Create a directory for recordings
        save_dir = "recorded_videos"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate a unique filename based on the current time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(save_dir, f"recorded_video_{timestamp}.mp4")
        audio_filename = os.path.join(save_dir, f"recorded_audio_{timestamp}.wav")
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        # Initialize audio stream
        p = pyaudio.PyAudio()
        audio_stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = []

        print(f"Recording started, saving video to '{video_filename}' and audio to '{audio_filename}'.")
        stop_record_button.place(relx=0.75, rely=0.9, anchor="center")  # Show the stop button
        exit_button.place_forget()  # Hide the exit button

        def record_audio():
            while recording:
                data = audio_stream.read(CHUNK)
                # Convert to numpy array for amplification
                audio_data = np.frombuffer(data, dtype=np.int16)
                # Amplify audio
                amplified_data = np.clip(audio_data * AMPLIFICATION_FACTOR, -32768, 32767).astype(np.int16)
                frames.append(amplified_data.tobytes())

        # Start the audio recording in a separate thread
        threading.Thread(target=record_audio).start()

    def stop_recording():
        global recording, out, audio_stream, frames
        recording = False
        if out is not None:
            out.release()
            out = None
            print("Video recording stopped.")
        if audio_stream is not None:
            audio_stream.stop_stream()
            audio_stream.close()
            p = pyaudio.PyAudio()
            audio_filename = f"recorded_videos/recorded_audio_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            wf = wave.open(audio_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(AUDIO_FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            print(f"Audio recording stopped and saved as '{audio_filename}'.")
        stop_record_button.place_forget()  # Hide the stop button
        exit_button.place(relx=0.75, rely=0.9, anchor="center")  # Show the exit button

    def start_scanning():
        global scanning
        scanning = True
        start_scan_button.place_forget()  # Hide the start button
        stop_scan_button.place(relx=0.25, rely=0.85, anchor="center")  # Show the stop button

    def stop_scanning():
        global scanning
        scanning = False
        stop_scan_button.place_forget()  # Hide the stop button
        start_scan_button.place(relx=0.25, rely=0.85, anchor="center")  # Show the start button

    def exit_application():
        # Exit the application
        cap.release()
        client.loop_stop()
        client.disconnect()
        if recording:
            stop_recording()  # Ensure the recording is stopped properly
        stream_window.destroy()
        root.quit()  # Exit the main application

    # Create the main window for streaming
    stream_window = Toplevel(root)
    stream_window.title("CCTV Stream Viewer with Control")
    stream_window.geometry("800x600")  # Set initial size of the window

    # Create a label to display the video frames
    global video_label
    video_label = Label(stream_window)
    video_label.pack(fill="both", expand=True)

    # Create a Capture button that floats on the video stream
    capture_button = Button(stream_window, text="Capture Image", font=("Arial", 14), command=capture_image)
    capture_button.place(relx=0.5, rely=0.85, anchor="center")  # Adjusted position for padding

    # Create a Record Video button
    record_button = Button(stream_window, text="Start Recording", font=("Arial", 14), command=start_recording)
    record_button.place(relx=0.75, rely=0.85, anchor="center")  # Adjusted position for padding

    # Create a Stop Recording button (initially hidden)
    stop_record_button = Button(stream_window, text="Stop Recording", font=("Arial", 14), command=stop_recording)

    # Create an Exit button to close the application
    exit_button = Button(stream_window, text="Exit", font=("Arial", 14), command=exit_application)
    exit_button.place(relx=0.75, rely=0.9, anchor="center")  # Adjusted position for padding

    # Create buttons for scanning
    start_scan_button = Button(stream_window, text="Start Scanning", font=("Arial", 14), command=start_scanning)
    start_scan_button.place(relx=0.25, rely=0.85, anchor="center")  # Show the start scanning button

    stop_scan_button = Button(stream_window, text="Stop Scanning", font=("Arial", 14), command=stop_scanning)

    # Start the video frame update loop
    update_frame()

    # Connect to MQTT broker
    client = mqtt.Client()

    def on_connect(client, userdata, flags, rc):
        print("Connected to MQTT Broker with result code " + str(rc))

    def control_camera(direction):
        client.publish(MQTT_TOPIC, direction)
        print(f"Sent command: {direction}")

    def key_press(event):
        if event.keysym == 'Up':
            control_camera('up')
        elif event.keysym == 'Down':
            control_camera('down')
        elif event.keysym == 'Left':
            control_camera('left')
        elif event.keysym == 'Right':
            control_camera('right')

    # Bind arrow keys to the control function
    stream_window.bind('<Up>', key_press)
    stream_window.bind('<Down>', key_press)
    stream_window.bind('<Left>', key_press)
    stream_window.bind('<Right>', key_press)

    # Connect to MQTT broker
    client.on_connect = on_connect
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    # Handle window close event to release resources
    stream_window.protocol("WM_DELETE_WINDOW", exit_application)

def submit_credentials():
    global rtsp_url
    username = username_var.get()
    password = password_var.get()
    host = host_var.get()
    port = port_var.get()
    rtsp_url = f"rtsp://{username}:{password}@{host}:{port}/cam/realmonitor?channel=1&subtype=0&unicast=true"

    # rtsp://192.168.1.20:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif
    # ForcePersistent = true
    root.withdraw()  # Hide the login window
    start_stream()   # Start streaming

def scan_network_for_cameras():
    def scan_ip(ip, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)  # Increased timeout for better detection on slow networks
            result = sock.connect_ex((ip, port))
            if result == 0:
                device_name = get_device_name(ip, port)
                found_cameras.append(f"{device_name} ({ip}:{port})" if device_name else f"{ip}:{port}")
                update_camera_list()
            sock.close()
        except Exception as e:
            print(f"Error scanning IP {ip}: {e}")

    def get_device_name(ip, port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)  # Increase timeout for device name fetching
                s.connect((ip, port))
                s.sendall(b"GET / HTTP/1.1\r\nHost: {}\r\n\r\n".format(ip).encode())
                response = s.recv(1024).decode()
                if "Server:" in response:
                    server_line = [line for line in response.splitlines() if "Server:" in line]
                    if server_line:
                        return server_line[0].split("Server: ")[-1].strip()
        except Exception as e:
            print(f"Failed to get device name for {ip}:{port} - {e}")
        return None

    def update_camera_list():
        camera_listbox.delete(0, tk.END)  # Clear existing list
        for camera in found_cameras:
            camera_listbox.insert(tk.END, camera)
        if not found_cameras:
            camera_listbox.insert(tk.END, "No cameras found.")

    found_cameras = []
    
    # Scan specific range of IPs and ports
    for i in range(1, 255):  # Adjust range if your subnet is different
        ip = f"192.168.1.{i}"
        # Scanning common RTSP ports (e.g., 554, 8554)
        for port in [554, 8554]:
            threading.Thread(target=scan_ip, args=(ip, port)).start()

def show_camera_selection_dialog():
    # Create a new window for camera selection
    scan_window = Toplevel(root)
    scan_window.title("Available Cameras")
    scan_window.geometry("400x400")

    Label(scan_window, text="Scanning for available cameras...", font=("Arial", 14)).pack(pady=10)
    
    global camera_listbox
    camera_listbox = Listbox(scan_window, font=("Arial", 12), height=15, width=35)
    camera_listbox.pack(pady=10)

    def select_camera():
        try:
            selected = camera_listbox.get(camera_listbox.curselection())
            ip_port = selected.split('(')[-1].strip(')')
            ip, port = ip_port.split(':')
            host_var.set(ip)  # Set the selected IP as the host
            port_var.set(port)  # Set the selected port
            scan_window.destroy()   # Close the scan window
            root.deiconify()        # Show the login window
        except tk.TclError:
            messagebox.showwarning("Selection Error", "Please select a camera from the list.")

    select_button = Button(scan_window, text="Select Camera", font=("Arial", 14), command=select_camera)
    select_button.pack(pady=10)

    scan_network_for_cameras()

# Create the main window for input
root = tk.Tk()
root.title("CCTV Login")
root.geometry("500x500")  # Set larger initial size of the window
root.withdraw()  # Hide the main window initially

# Variables to store user inputs
username_var = StringVar()
password_var = StringVar()
host_var = StringVar()
port_var = StringVar()

# Create and place labels and entry fields
Label(root, text="Username:", font=("Arial", 18)).pack(pady=10)
username_entry = Entry(root, textvariable=username_var, font=("Arial", 18), width=25)
username_entry.pack(pady=5)

Label(root, text="Password:", font=("Arial", 18)).pack(pady=10)
password_entry = Entry(root, textvariable=password_var, font=("Arial", 18), show="*", width=25)
password_entry.pack(pady=5)

Label(root, text="Host:", font=("Arial", 18)).pack(pady=10)
host_entry = Entry(root, textvariable=host_var, font=("Arial", 18), width=25)
host_entry.pack(pady=5)

Label(root, text="Port:", font=("Arial", 18)).pack(pady=10)
port_entry = Entry(root, textvariable=port_var, font=("Arial", 18), width=25)
port_entry.pack(pady=5)

# Submit button with increased size
submit_button = Button(root, text="Start Stream", font=("Arial", 20, 'bold'), command=submit_credentials, width=20, height=2)
submit_button.pack(pady=30)

# Show camera selection dialog at the start
show_camera_selection_dialog()

# Run the Tkinter main loop
root.mainloop()
