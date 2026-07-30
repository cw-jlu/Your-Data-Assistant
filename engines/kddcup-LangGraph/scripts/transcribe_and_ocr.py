import sys
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\src")))

def run():
    print("Python version:", sys.version)
    
    # 尝试检查 ASR 库
    try:
        from data_agent_baseline.run.video_preprocessor import transcribe_video_audio
        print("transcribe_video_audio imported successfully.")
        
        video_path = Path(r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\data\input\task_6\context\video\briefing.mp4")
        print("Transcribing...")
        # 我们可以只用 cpu 运行 faster_whisper
        res = transcribe_video_audio(video_path, device="cpu")
        print("ASR Result:")
        print(res.get("text"))
        print("ASR Segments:")
        for seg in res.get("segments", []):
            print(f"[{seg['start_sec']} - {seg['end_sec']}]: {seg['text']}")
    except Exception as e:
        print("Error during ASR:", e)

    # 检查是否有 OCR 相关的包
    ocr_libs = ["pytesseract", "easyocr", "paddleocr", "ddddocr", "tesserocr"]
    print("\nChecking OCR libraries:")
    for lib in ocr_libs:
        try:
            __import__(lib)
            print(f"  {lib}: Available")
        except ImportError:
            print(f"  {lib}: Not available")

if __name__ == "__main__":
    run()
