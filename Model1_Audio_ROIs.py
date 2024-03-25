from moviepy.editor import VideoFileClip
from pyAudioAnalysis import audioBasicIO, audioSegmentation

# Step 1: Extract audio from the video using moviepy
def extract_audio(video_file, audio_file):
    video_clip = VideoFileClip(video_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_file)

# Step 2: Perform audio analysis using pyAudioAnalysis
def analyze_audio(audio_file):
    # Read audio file
    [Fs, x] = audioBasicIO.read_audio_file(audio_file)
    
    # Perform voice activity detection
    segments = audioSegmentation.silence_removal(x, Fs, 0.020, 0.020, smooth_window=1.0, weight=0.3, plot=False)
    
    # Output the segments with speech activity
    for seg in segments:
        print("Speech segment:", seg)

# Video file path
video_file = 'your_video_file.mp4'
# Audio file path
audio_file = 'output_audio.wav'

# Step 1: Extract audio from the video
extract_audio(video_file, audio_file)

# Step 2: Analyze the extracted audio using pyAudioAnalysis
analyze_audio(audio_file)
