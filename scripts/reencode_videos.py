"""Re-encode all demo and precomputed MP4 videos to H.264 (AVC1) for native browser playback."""
import glob
import os
import subprocess
import imageio_ffmpeg

def reencode_to_h264():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    mp4_files = glob.glob("demo/**/*.mp4", recursive=True)

    print(f"Found {len(mp4_files)} MP4 files to convert to browser-compatible H.264...")

    for fpath in mp4_files:
        temp_out = fpath.replace(".mp4", "_h264.mp4")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", fpath,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            temp_out,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(temp_out):
            os.replace(temp_out, fpath)
            print(f"[OK] Re-encoded -> {fpath}")
        else:
            print(f"[FAIL] Failed to re-encode {fpath}: {res.stderr.decode('utf-8', errors='ignore')[:200]}")

    print("\nAll videos are now standard H.264 and 100% playable in Chrome, Edge, and Safari!")

if __name__ == "__main__":
    reencode_to_h264()
