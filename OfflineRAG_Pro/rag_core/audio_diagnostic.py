# audio_diagnostic.py
# Run this to diagnose audio recording issues

import sys
import numpy as np

print("=" * 60)
print("🎤 AUDIO DIAGNOSTIC TOOL")
print("=" * 60)

# Test 1: Check audio libraries
print("\n1️⃣ Checking audio libraries...")
try:
    import sounddevice as sd

    print("   ✅ sounddevice installed")
except ImportError:
    print("   ❌ sounddevice NOT installed")
    print("   📦 Install: pip install sounddevice")
    sys.exit(1)

try:
    import soundfile as sf

    print("   ✅ soundfile installed")
except ImportError:
    print("   ❌ soundfile NOT installed")
    print("   📦 Install: pip install soundfile")
    sys.exit(1)

try:
    import whisper

    print("   ✅ whisper installed")
except ImportError:
    print("   ❌ whisper NOT installed")
    print("   📦 Install: pip install openai-whisper")
    sys.exit(1)

# Test 2: List audio devices
print("\n2️⃣ Available audio devices:")
try:
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        default_input = "🎤 DEFAULT INPUT" if i == sd.default.device[0] else ""
        default_output = "🔊 DEFAULT OUTPUT" if i == sd.default.device[1] else ""

        print(f"\n   [{i}] {dev['name']}")
        print(f"       Input channels: {dev['max_input_channels']}")
        print(f"       Output channels: {dev['max_output_channels']}")
        print(f"       Sample rate: {dev['default_samplerate']} Hz")
        if default_input or default_output:
            print(f"       {default_input} {default_output}")
except Exception as e:
    print(f"   ❌ Error listing devices: {e}")

# Test 3: Test microphone
print("\n3️⃣ Testing microphone...")
try:
    sample_rate = 16000
    duration = 3

    print(f"   Recording {duration} seconds...")
    print("   🗣️  Please speak something!")

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    audio = recording.flatten()

    # Analyze recording
    max_amplitude = np.max(np.abs(audio))
    mean_energy = np.mean(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))

    print(f"\n   📊 Recording Analysis:")
    print(f"      Max amplitude: {max_amplitude:.4f}")
    print(f"      Mean energy: {mean_energy:.4f}")
    print(f"      RMS: {rms:.4f}")

    if max_amplitude < 0.001:
        print("\n   ⚠️  WARNING: Audio signal too weak!")
        print("      Possible issues:")
        print("      • Microphone not connected")
        print("      • Wrong input device selected")
        print("      • Microphone muted in system settings")
        print("      • Need to grant microphone permissions")
    elif max_amplitude > 0.01:
        print("\n   ✅ Microphone working properly!")
    else:
        print("\n   ⚠️  Audio detected but weak - check microphone volume")

    # Save test recording
    test_file = "test_recording.wav"
    sf.write(test_file, audio, sample_rate)
    print(f"\n   💾 Saved test recording to: {test_file}")

except Exception as e:
    print(f"   ❌ Microphone test failed: {e}")

# Test 4: Test Whisper transcription
print("\n4️⃣ Testing Whisper transcription...")
try:
    if 'audio' in locals() and max_amplitude > 0.001:
        print("   Loading Whisper model (this may take a moment)...")
        model = whisper.load_model("tiny")  # Use tiny for speed

        print("   Transcribing...")
        result = model.transcribe(audio, fp16=False, language="en")

        text = result.get("text", "").strip()
        print(f"\n   📝 Transcription: '{text}'")

        if not text:
            print("   ⚠️  No transcription - audio may be too quiet or unclear")
        else:
            print("   ✅ Whisper working!")
    else:
        print("   ⏭️  Skipped (no valid audio)")
except Exception as e:
    print(f"   ❌ Whisper test failed: {e}")

# Test 5: Check system permissions
print("\n5️⃣ System recommendations:")
print("   Windows:")
print("   • Settings → Privacy → Microphone → Allow apps")
print("   • Check microphone volume in Sound settings")
print("   ")
print("   Linux:")
print("   • Run: pactl list sources (to see audio sources)")
print("   • Install: sudo apt-get install libportaudio2")
print("   ")
print("   Mac:")
print("   • System Preferences → Security & Privacy → Microphone")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

# Final recommendations
print("\n💡 Quick fixes:")
print("1. Try selecting different input device in system sound settings")
print("2. Increase microphone volume/gain")
print("3. Test with a different microphone")
print("4. Run Streamlit with administrator privileges")
print("5. Check if other apps can use the microphone")

print("\nTo use a specific device in Jarvis, modify voice_io.py:")
print("   sd.default.device = [DEVICE_ID, None]  # Use device ID from list above")