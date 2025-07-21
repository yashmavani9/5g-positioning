import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from collections import defaultdict
from math import sqrt


# Log and csv file
csv_log_path = r'C:/Users/yhm/Desktop/5g_new/alert/code/alert_log.csv'
human_log_path = r'C:/Users/yhm/Desktop/5g_new/alert/code/alert_log.log'

# Parameters
N = 4       # Minimum cluster size
T = 3       # Time window (seconds)
D = 50      # Movement threshold
EPS = 50    # DBSCAN epsilon (is in pixels)
MIN_SAMPLES = N
UE_TRAIL_LENGTH = 5 # Number of previous positions to keep for each UE for trail graphics

# Load dataset
df = pd.read_csv(r'C:/Users/yhm/Desktop/5g_new/alert/scenario2-hostage.csv')
# df = pd.read_csv(r'C:/Users/yhm/Desktop/5g_new/alert/scenario1-normal.csv')
# df = pd.read_csv(r'C:/Users/yhm/Desktop/5g_new/alert/scenario3-bus.csv')
df = df[df['status'] == 1]
timestamps = sorted(df['timestamp'].unique())

# Load map image
map_img = cv2.imread(r'C:/Users/yhm/Desktop/5g_new/alert/iitgn3.jpg')
orig_img = map_img.copy()
img_height, img_width = map_img.shape[:2]

# Store recent positions for each device/UE
ue_trail_history = defaultdict(list)

# Cluster history
cluster_history = defaultdict(list)
font = cv2.FONT_HERSHEY_SIMPLEX

# 5× Visual Scaling Parameters
FONT_SCALE = 5
FONT_THICKNESS = 10
CIRCLE_RADIUS = 60
CIRCLE_THICKNESS = -1
ALERT_RADIUS = 150
ALERT_BORDER = 20
ALERT_TEXT_OFFSET = 200

# Display size
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1000, 700

# Main loop
for t_index, current_time in enumerate(timestamps):
    frame = orig_img.copy()
    # df = df[df['status'] == 1]  --> Already filtered above
    # current_df = df[df['timestamp'] == current_time]
    current_df = df[df['timestamp'] == current_time].copy()
    coords = current_df[['x', 'y']].to_numpy()
    coords = coords[~np.isnan(coords).any(axis=1)]  # Remove NaN values

    # Update each UE trail history
    for _, row in current_df.iterrows():
        ue_id = row['device_id'] if 'device_id' in row else row.name 
        ue_trail_history[ue_id].append((int(row['x']), int(row['y'])))

        # Keep only last N entries
        if len(ue_trail_history[ue_id]) > UE_TRAIL_LENGTH:
            ue_trail_history[ue_id] = ue_trail_history[ue_id][-UE_TRAIL_LENGTH:]


    # Handling the case with no sufficient people
    if len(coords) < MIN_SAMPLES:
        for x, y in coords:
            cv2.circle(frame, (int(x), int(y)), CIRCLE_RADIUS, (150, 150, 150), CIRCLE_THICKNESS)
        cv2.putText(frame, f'Time: {current_time}s | No Clusters', (50, 150),
                    font, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
    else:
        # More than N people detected, perform clustering(DBSCAN)
        clustering = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(coords)
        # DBSCAN will assigns cluster label to each point (-1 for noise)
        labels = clustering.labels_
        current_df['cluster'] = labels #Adding cluster labels to the DataFrame
        alert_triggered = False

        for label in set(labels):
            cluster_points = current_df[current_df['cluster'] == label]
            x_vals = cluster_points['x'].to_numpy()
            y_vals = cluster_points['y'].to_numpy()
            color = (0, 255, 0) if label != -1 else (180, 180, 180) # Green for clusters, gray(= -1) for noise
            # color = (1,1,185) if label != -1 else (129, 227, 246)  # Yellow for clusters, Green(=-1) for noise

            for i in range(len(x_vals)):
                cv2.circle(frame, (int(x_vals[i]), int(y_vals[i])), CIRCLE_RADIUS, color, CIRCLE_THICKNESS)

            if label != -1:
                # if cluster with enough point and not noise 
                # Calculate centroid 
                centroid_x = np.mean(x_vals)
                centroid_y = np.mean(y_vals)
                cluster_history[label].append((current_time, centroid_x, centroid_y))

                # Keep only last T seconds
                cluster_history[label] = [
                    entry for entry in cluster_history[label]
                    if current_time - entry[0] <= T
                ]

                # Check if we have enough N at current time and two timestamps in history
                if len(cluster_points) >= N and len(cluster_history[label]) > 1:
                    start_x, start_y = cluster_history[label][0][1:]
                    end_x, end_y = cluster_history[label][-1][1:]
                    movement = sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2) #How much the group has drifted
                    
                    # Trigger alert if movement is less than D and duration is longer than T
                    if movement <= D and (cluster_history[label][-1][0] - cluster_history[label][0][0]) >= T:
                        alert_triggered = True
                        cv2.circle(frame, (int(centroid_x), int(centroid_y)), ALERT_RADIUS, (0, 0, 255), ALERT_BORDER)
                        cv2.putText(frame, "ALERT!",
                                    (int(centroid_x) - ALERT_TEXT_OFFSET, int(centroid_y) - ALERT_TEXT_OFFSET),
                                    font, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
                        
                        # Adding the alert data into .csv and .log
                        # Log to CSV file
                        with open(csv_log_path, 'a') as f_csv:
                            f_csv.write(f"{current_time},{len(cluster_points)},{int(centroid_x)},{int(centroid_y)}\n")

                        # Log to human-readable .log file
                        with open(human_log_path, 'a') as f_log:
                            f_log.write(
                                        f"[ALERT] Time: {current_time}s | "
                                        f"People: {len(cluster_points)} | "
                                        f"Centroid: ({int(centroid_x)}, {int(centroid_y)})\n"
                                        )

        if alert_triggered:
            cv2.putText(frame, "ALERT: Hostage-like Cluster Detected!",
                        (50, 250), font, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)
            
    
    # Draw trails for each UE
    # for trail in ue_trail_history.values():
    #     if len(trail) >= 2: # At least two points for a line
    #         for i in range(len(trail) - 1):
    #             cv2.line(frame, trail[i], trail[i + 1], (90,218,90), 10)  

    # Time display (bottom-left)
    cv2.putText(frame, f'Time: {current_time}s',
                (50, 150), font, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)

    # Resize image to fit screen
    frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("Hostage Detection", frame_resized)

    key = cv2.waitKey(500)
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()

