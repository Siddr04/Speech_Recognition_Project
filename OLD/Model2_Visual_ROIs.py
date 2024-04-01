import cv2

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video file
video_capture = cv2.VideoCapture('referral-marketing-how-to-approach-referral-partners_MLuMohuj.mp4')

# Get frames per second (fps) of the video
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object to save the output video
out_video = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (int(video_capture.get(3)), int(video_capture.get(4))))

# Initialize an audio recorder
audio = cv2.VideoCapture('referral-marketing-how-to-approach-referral-partners_MLuMohuj.mp4')

# Get the audio codec
fourcc = int(audio.get(cv2.CAP_PROP_FOURCC))

# # Get the number of audio channels
# channels = audio.get(cv2.CAP_PROP_CHANNEL_COUNT)

# Get the sample rate
sample_rate = int(audio.get(cv2.CAP_PROP_FRAME_WIDTH))

# Loop through each frame of the video
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Optionally, you can save the visual ROIs as images
        # cv2.imwrite('face_{}.jpg'.format(frame_number), frame[y:y+h, x:x+w])

    # Write the frame with detected faces to the output video
    out_video.write(frame)
    
    # Display the frame
    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and video writer objects
video_capture.release()
out_video.release()
cv2.destroyAllWindows()
