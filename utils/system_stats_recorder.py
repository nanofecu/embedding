import psutil
import pandas as pd
import time
from datetime import datetime

# Set the data recording interval in seconds (e.g., every 60 seconds)
INTERVAL = 60

# Initialize data storage
data = {
    "timestamp": [],
    "cpu_usage": [],
    "memory_usage": [],
    "disk_read_speed": [],
    "disk_write_speed": []
}

# Retrieve initial disk IO counters to compare subsequent readings
initial_disk_io = psutil.disk_io_counters()

try:
    while True:
        # Record the current time
        current_time = datetime.now()

        # Record overall CPU usage percentage
        cpu_usage = psutil.cpu_percent()

        # Record memory usage percentage
        memory_usage = psutil.virtual_memory().percent

        # Calculate disk read and write speeds by comparing current and initial IO counters
        current_disk_io = psutil.disk_io_counters()
        disk_read_speed = current_disk_io.read_bytes - initial_disk_io.read_bytes
        disk_write_speed = current_disk_io.write_bytes - initial_disk_io.write_bytes

        # Update the initial disk IO counters for the next loop iteration
        initial_disk_io = current_disk_io

        # Store the collected data
        data['timestamp'].append(current_time)
        data['cpu_usage'].append(cpu_usage)
        data['memory_usage'].append(memory_usage)
        data['disk_read_speed'].append(disk_read_speed)
        data['disk_write_speed'].append(disk_write_speed)

        # Output the latest CPU usage to console for verification
        print(f"Data captured: {cpu_usage}% CPU Usage")

        # Sleep for the interval duration before collecting the next set of data
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    # Save the collected data to a CSV file when the user interrupts the program (e.g., by pressing Ctrl+C)
    df = pd.DataFrame(data)
    df.to_csv('system_monitoring_data.csv', index=False)
    print("Data saved to 'system_monitoring_data.csv'. Exiting...")
