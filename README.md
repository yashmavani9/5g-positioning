<h1>5G Positioning for Early Hostage Detection</h1>

This project explores the use of 5G positioning techniques to detect potential hostage scenarios by analyzing UE movement patterns. The system focuses on identifying clusters of devices that remain stationary over time, which may indicate a hostage-like situation. Time Difference of Arrival (TDOA) was chosen for its accuracy and its ability to estimate UE locations without requiring synchronization. As existing datasets and simulation tools had limitations, we created a Python-based generator to simulate UE coordinates. A real-time alert system was built using DBSCAN and OpenCV to detect and visualize such clusters on a campus map.

