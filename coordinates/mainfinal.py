import cv2
import numpy as np
import pandas as pd

# === Settings ===
max_points = 24
num_devices = 20
image_path = r"C:/Users/yhm/Desktop/5g_new/coordinates/iitgn3.jpg"
view_width, view_height = 1280, 720

# === Load image ===
img = cv2.imread(image_path)
if img is None:
    print("Error loading image.")
    exit()

min_scale = max(view_width / img.shape[1], view_height / img.shape[0])
max_scale = 5.0
scale = 1.0
offset_x, offset_y = 0, 0
drag_start = None

# === For all devices ===
device_points = [[] for _ in range(num_devices)]
current_device = 0

# === Colors for each device ===
colors = [(int(np.random.randint(0, 256)), int(np.random.randint(0, 256)), int(np.random.randint(0, 256)))
          for _ in range(num_devices)]

cv2.namedWindow("Map Viewer", cv2.WINDOW_NORMAL)

def draw_canvas():
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    max_offset_x = max(new_w - view_width, 0)
    max_offset_y = max(new_h - view_height, 0)
    ox = np.clip(offset_x, 0, max_offset_x)
    oy = np.clip(offset_y, 0, max_offset_y)

    end_x = min(ox + view_width, new_w)
    end_y = min(oy + view_height, new_h)

    view = resized[oy:end_y, ox:end_x].copy()

    # Draw points for all devices
    for dev_id, dev_points in enumerate(device_points):
        for idx, (px, py) in enumerate(dev_points):
            sx = int(px * scale) - ox
            sy = int(py * scale) - oy
            if 0 <= sx < view.shape[1] and 0 <= sy < view.shape[0]:
                cv2.circle(view, (sx, sy), 5, colors[dev_id], -1)
                if dev_id == current_device:
                    cv2.putText(view, str(idx + 1), (sx + 5, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    return view

def mouse_callback(event, x, y, flags, param):
    global drag_start, offset_x, offset_y, scale

    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            px = int((x + offset_x) / scale)
            py = int((y + offset_y) / scale)
            if len(device_points[current_device]) < max_points:
                device_points[current_device].append((px, py))
                print(f"Device {current_device + 1} - Point {len(device_points[current_device])}: ({px}, {py})")
        else:
            drag_start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drag_start:
        dx = x - drag_start[0]
        dy = y - drag_start[1]
        offset_x = np.clip(offset_x - dx, 0, max(int(img.shape[1] * scale) - view_width, 0))
        offset_y = np.clip(offset_y - dy, 0, max(int(img.shape[0] * scale) - view_height, 0))
        drag_start = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drag_start = None

    elif event == cv2.EVENT_MOUSEWHEEL:
        zoom_dir = 1.1 if flags > 0 else 0.9
        new_scale = np.clip(scale * zoom_dir, min_scale, max_scale)

        mx = x + offset_x
        my = y + offset_y
        ratio = new_scale / scale
        offset_x = int(mx * ratio - x)
        offset_y = int(my * ratio - y)
        offset_x = np.clip(offset_x, 0, max(int(img.shape[1] * new_scale) - view_width, 0))
        offset_y = np.clip(offset_y, 0, max(int(img.shape[0] * new_scale) - view_height, 0))
        scale = new_scale

cv2.setMouseCallback("Map Viewer", mouse_callback)

print("=== Instructions ===")
print("Ctrl + Click to mark points")
print("Drag mouse to move image")
print("Scroll to zoom in/out")
print("'s' to save and go to next device")
print("ESC to exit")

while current_device < num_devices:
    view = draw_canvas()
    title_text = f"Map Viewer - Device {current_device + 1} ({len(device_points[current_device])}/{max_points})"
    cv2.setWindowTitle("Map Viewer", title_text)
    cv2.imshow("Map Viewer", view)
    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        print("Cancelled.")
        break

    elif key == ord('s') or len(device_points[current_device]) >= max_points:
        if len(device_points[current_device]) == max_points:
            print(f"Device {current_device + 1} mapping completed.")
            current_device += 1
        else:
            print(f"Device {current_device + 1} doesn't have 24 points. Complete before saving.")

cv2.destroyAllWindows()

# === Save Final CSV ===
records = []
for device_id, points in enumerate(device_points, start=1):
    for timestamp, (x, y) in enumerate(points, start=1):
        records.append({
            "device_id": device_id,
            "timestamp": timestamp,
            "x": x,
            "y": y,
            "status": 1
        })

df = pd.DataFrame(records)
df.sort_values(by=["timestamp", "device_id"], inplace=True)
# output_file = "multi_device_path.csv"
output_file = "scenario4.csv"
df.to_csv(output_file, index=False)
print(f"\nSaved data for {num_devices} devices to '{output_file}'")

