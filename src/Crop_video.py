import cv2
import os


def convert_to_seconds(timestamp):
    """
    Convert a timestamp string (HH:MM:SS) to total seconds.

    :param timestamp: String in format "HH:MM:SS"
    :return: Total seconds as an integer
    """
    h, m, s = map(int, timestamp.split(":"))
    return h * 3600 + m * 60 + s


def crop_video(input_video, output_video, start_time, end_time):
    """
    Crop an MP4 video file between specified time frames using OpenCV.

    :param input_video: Path to the input video file.
    :param output_video: Path to save the cropped video.
    :param start_time: Start time as a string in "HH:MM:SS".
    :param end_time: End time as a string in "HH:MM:SS".
    """
    # Check if the input file exists
    if not os.path.exists(input_video):
        print(f"❌ Error: File '{input_video}' not found.")
        return

    # Convert timestamps to seconds
    start_sec = convert_to_seconds(start_time)
    end_sec = convert_to_seconds(end_time)

    # Open the input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("❌ Error: Unable to open video file.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the starting and ending frame numbers
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    if start_frame >= total_frames:
        print("❌ Error: Start time is beyond video length.")
        cap.release()
        return
    if end_frame > total_frames:
        end_frame = total_frames

    # Set up the VideoWriter (using mp4v codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"Processing frames from {start_frame} to {end_frame}...")
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= current_frame < end_frame:
            out.write(frame)
        current_frame += 1
        if current_frame >= end_frame:
            break

    cap.release()
    out.release()
    print(f"✅ Cropped video saved as: {output_video}")


# Example usage
input_video = r"C:\Users\97254\PycharmProjects\pythonProject\pool_02.mp4"
output_video = r"C:\Users\97254\PycharmProjects\pythonProject\cropped_output.mp4"
start_time = "00:01:01"
end_time = "00:01:45"

crop_video(input_video, output_video, start_time, end_time)
