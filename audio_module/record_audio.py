import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100
seconds = 5

print("Recording...")

audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)

sd.wait()

write("driver_response.wav", fs, audio)

print("Recording saved")