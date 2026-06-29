#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from urllib import request

import edge_tts
import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad


SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 512
SAMPLE_WIDTH_BYTES = 2


@dataclass
class Settings:
    mic_device: str
    speaker_device: str
    ollama_model: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    tts_voice: str
    silence_ms: int
    max_utterance_seconds: float
    ollama_url: str
    system_prompt: str
    max_history_turns: int


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required command: {name}")


def pcm16_to_float32(chunk: bytes) -> np.ndarray:
    return np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0


class MicStream:
    def __init__(self, device: str):
        self.device = device
        self.proc: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "MicStream":
        require_command("arecord")
        self.proc = subprocess.Popen(
            [
                "arecord",
                "-q",
                "-D",
                self.device,
                "-f",
                "S16_LE",
                "-r",
                str(SAMPLE_RATE),
                "-c",
                "1",
                "-t",
                "raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.1)
        if self.proc.poll() is not None:
            stderr = ""
            if self.proc.stderr:
                stderr = self.proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"microphone stream failed to open. {stderr}".strip())
        return self

    def __exit__(self, *_args: object) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def read_chunk(self) -> bytes:
        if not self.proc or not self.proc.stdout:
            raise RuntimeError("microphone stream is not open")
        chunk_size = CHUNK_SAMPLES * SAMPLE_WIDTH_BYTES
        chunk = self.proc.stdout.read(chunk_size)
        if len(chunk) != chunk_size:
            stderr = ""
            if self.proc.stderr:
                stderr = self.proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"microphone stream ended early. {stderr}".strip())
        return chunk


class SpeechSegmenter:
    def __init__(self, silence_ms: int, max_seconds: float):
        self.model = load_silero_vad()
        self.vad = VADIterator(
            self.model,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=silence_ms,
        )
        self.max_chunks = int(max_seconds * SAMPLE_RATE / CHUNK_SAMPLES)

    def listen_once(self, mic: MicStream) -> np.ndarray:
        frames: list[bytes] = []
        in_speech = False
        started_at = time.monotonic()

        while True:
            chunk = mic.read_chunk()
            audio = pcm16_to_float32(chunk)
            event = self.vad(torch.from_numpy(audio), return_seconds=False)

            if event and "start" in event:
                frames = [chunk]
                in_speech = True
                started_at = time.monotonic()
                continue

            if in_speech:
                frames.append(chunk)

            if event and "end" in event and in_speech:
                self.vad.reset_states()
                return pcm16_to_float32(b"".join(frames))

            if in_speech and len(frames) >= self.max_chunks:
                self.vad.reset_states()
                return pcm16_to_float32(b"".join(frames))

            if not in_speech and time.monotonic() - started_at > 0.5:
                started_at = time.monotonic()


class Assistant:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.messages = [{"role": "system", "content": settings.system_prompt}]
        self.asr = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.asr.transcribe(
            audio,
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def ask_ollama(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        self.trim_history()
        payload = {
            "model": self.settings.ollama_model,
            "messages": self.messages,
            "stream": False,
            "options": {"temperature": 0.4},
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.settings.ollama_url.rstrip('/')}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        reply = body.get("message", {}).get("content", "").strip()
        if reply:
            self.messages.append({"role": "assistant", "content": reply})
            self.trim_history()
        return reply

    def trim_history(self) -> None:
        keep = self.settings.max_history_turns * 2
        if len(self.messages) > keep + 1:
            self.messages = [self.messages[0], *self.messages[-keep:]]

    async def speak(self, text: str) -> None:
        require_command("ffmpeg")
        require_command("aplay")
        with tempfile.TemporaryDirectory() as tmpdir:
            mp3_path = os.path.join(tmpdir, "reply.mp3")
            await edge_tts.Communicate(text, self.settings.tts_voice).save(mp3_path)
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    mp3_path,
                    "-f",
                    "wav",
                    "-",
                ],
                stdout=subprocess.PIPE,
            )
            aplay = subprocess.Popen(
                ["aplay", "-q", "-D", self.settings.speaker_device],
                stdin=ffmpeg.stdout,
            )
            if ffmpeg.stdout:
                ffmpeg.stdout.close()
            aplay.wait()
            ffmpeg.wait()


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="Local voice assistant")
    parser.add_argument("--mic-device", default=os.getenv("MIC_DEVICE", "plughw:2,0"))
    parser.add_argument("--speaker-device", default=os.getenv("SPEAKER_DEVICE", "plughw:0,0"))
    parser.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "exaone3.5:2.4b"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--whisper-model", default=os.getenv("WHISPER_MODEL", "small"))
    parser.add_argument("--whisper-device", default=os.getenv("WHISPER_DEVICE", "cpu"))
    parser.add_argument("--whisper-compute-type", default=os.getenv("WHISPER_COMPUTE_TYPE", "int8"))
    parser.add_argument("--tts-voice", default=os.getenv("TTS_VOICE", "ko-KR-SunHiNeural"))
    parser.add_argument("--silence-ms", type=int, default=int(os.getenv("VAD_SILENCE_MS", "700")))
    parser.add_argument(
        "--system-prompt",
        default=os.getenv(
            "SYSTEM_PROMPT",
            (
                "당신은 한국어 음성 대화를 위한 자연스러운 AI 비서입니다. "
                "사용자의 이전 발화를 기억하고 이어서 대화하세요. "
                "답변은 말로 듣기 좋게 짧고 직접적으로 하세요. "
                "음성 인식 결과가 어색하면 단정하지 말고 자연스럽게 확인 질문을 하세요. "
                "이모지, 마크다운, 긴 목록은 쓰지 마세요."
            ),
        ),
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=int(os.getenv("MAX_HISTORY_TURNS", "8")),
    )
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=float(os.getenv("MAX_UTTERANCE_SECONDS", "20")),
    )
    args = parser.parse_args()
    return Settings(**vars(args))


async def main() -> None:
    settings = parse_args()
    segmenter = SpeechSegmenter(settings.silence_ms, settings.max_utterance_seconds)
    assistant = Assistant(settings)

    print("Listening. Press Ctrl+C to stop.")
    print(f"Mic: {settings.mic_device} | Speaker: {settings.speaker_device}")
    print(f"ASR: Whisper {settings.whisper_model} | LLM: Ollama {settings.ollama_model}")

    with MicStream(settings.mic_device) as mic:
        while True:
            audio = segmenter.listen_once(mic)
            text = assistant.transcribe(audio)
            if not text:
                continue
            print(f"\nYou: {text}")
            reply = assistant.ask_ollama(text)
            print(f"Assistant: {reply}")
            if reply:
                await assistant.speak(reply)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
