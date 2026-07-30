import sys
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\src")))

def dump_asr():
    from data_agent_baseline.run.video_preprocessor import transcribe_video_audio
    
    video_path = Path(r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\data\input\task_6\context\video\briefing.mp4")
    print("Transcribing...")
    res = transcribe_video_audio(video_path, device="cpu")
    
    out_path = r"c:\Users\ulna\Desktop\kddcup\kddcup2026-data-agents-starter-kit\scripts\asr_output.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Full Text:\n")
        f.write(res.get("text", "") + "\n\n")
        f.write("Segments:\n")
        for seg in res.get("segments", []):
            f.write(f"[{seg['start_sec']} - {seg['end_sec']}]: {seg['text']}\n")
            
    print("ASR output written to asr_output.txt successfully.")

if __name__ == "__main__":
    dump_asr()
