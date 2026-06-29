#!/usr/bin/env python3
import argparse
import asyncio
from datetime import datetime
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any
from urllib import request

import edge_tts
from faster_whisper import WhisperModel

from voice_agent import MicStream, SpeechSegmenter


API_URL = "http://relay.emtake.com/api/query"
API_REFERENCE = """내용 예시
PowerShell:
curl.exe -X POST "http://relay.emtake.com/api/query" -H "Content-Type: application/json" -d '{\\"Type\\":\\"LLMREPORT\\", \\"Account\\":\\"test4@test.com\\", \\"CMD\\":\\"SleepData\\", \\"val\\":\\"1\\", \\"date\\":\\"2026-05-05\\"}'

Type:
- LLMREPORT: baby
- LLMREPORTS: senior

Optional:
- val: 마지막 날로부터 +val 값을 출력
- date: 특정 날짜의 데이터 출력
- val과 date를 둘 다 보내면 val이 우선이고 date는 무시됨

Available examples:
- test1@test.com / SleepData: 수면 시작, 끝, 수면 분, 도중 깬 횟수
- test3@test.com / Temp: 전날 최저/최고 온도, 0도 기준
- test5@test.com / Breath: 전날 최저/최고 호흡 값
- test7@test.com / dB: 전날 최고 실내 소음
- test9@test.com / IndoorTemp: 전날 최저/최고 실내 온도
- test11@test.com / ALL: Temp, Breath, dB, IndoorTemp, Humidity, Bright, SleepData 전체 호출 값
"""

@dataclass
class SensorQuery:
    sensor: str
    case: str
    condition: str
    query: str
    is_template: bool = False


GENERAL_QUERIES = [
    SensorQuery(
        sensor="general",
        case="current_condition",
        condition="no_abnormal_data",
        query="현재 몸 상태는 어떠신가요? 불편한 곳이 있나요?",
    ),
    SensorQuery(
        sensor="general",
        case="sleep_quality",
        condition="no_abnormal_data",
        query="지난 수면은 편안하셨나요? 중간에 깨거나 뒤척임이 있었나요?",
    ),
    SensorQuery(
        sensor="general",
        case="room_comfort",
        condition="no_abnormal_data",
        query="방 온도와 습도는 편안하게 느껴지시나요?",
    ),
    SensorQuery(
        sensor="general",
        case="breathing_comfort",
        condition="no_abnormal_data",
        query="숨쉬기나 가슴 답답함은 괜찮으신가요?",
    ),
    SensorQuery(
        sensor="general",
        case="support_needed",
        condition="no_abnormal_data",
        query="지금 도움이 필요하거나 보호자에게 전하고 싶은 말이 있나요?",
    ),
]


@dataclass
class Settings:
    api_url: str
    report_type: str
    account: str
    cmd: str
    val: str | None
    date: str | None
    queries_file: str
    dialog_file: str
    mic_device: str
    speaker_device: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    tts_voice: str
    silence_ms: int
    max_utterance_seconds: float
    max_queries: int
    playback_retries: int
    playback_retry_delay: float
    mic_retries: int
    mic_retry_delay: float
    device_settle_seconds: float
    interval_seconds: float
    once: bool
    dry_run: bool
    print_reference: bool
    print_raw: bool
    print_summary: bool
    body_temp_baseline: float


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required command: {name}")


def load_queries(path: str) -> list[SensorQuery]:
    queries: list[SensorQuery] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number}: expected JSON object")

            question = row.get("query")
            is_template = False
            if question is None:
                question = row.get("question_template")
                is_template = question is not None

            missing = [
                key
                for key in ("sensor", "case", "condition")
                if row.get(key) is None
            ]
            if question is None:
                missing.append("query or question_template")
            if missing:
                raise SystemExit(f"{path}:{line_number}: missing required field(s): {', '.join(missing)}")

            queries.append(
                SensorQuery(
                    sensor=str(row["sensor"]),
                    case=str(row["case"]),
                    condition=str(row["condition"]),
                    query=str(question),
                    is_template=is_template,
                )
            )
    return queries


def decode_json_body(raw: bytes) -> dict[str, Any]:
    body: Any = json.loads(raw.decode("utf-8"))
    while isinstance(body, str):
        body = json.loads(body)
    if not isinstance(body, dict):
        raise ValueError(f"expected JSON object, got {type(body).__name__}")
    return body


def fetch_sensor_data(settings: Settings) -> dict[str, Any]:
    payload = {
        "Type": settings.report_type,
        "Account": settings.account,
        "CMD": settings.cmd,
    }
    if settings.val:
        payload["val"] = settings.val
    elif settings.date:
        payload["date"] = settings.date

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        settings.api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as response:
        return decode_json_body(response.read())


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def minmax(section: Any) -> tuple[float | None, float | None]:
    if not isinstance(section, dict):
        return None, None
    return number(section.get("Min")), number(section.get("Max"))


def normalize_body_temperature(value: float | None, baseline: float) -> float | None:
    if value is None:
        return None
    # The API example labels Temp as "0 degree based"; small values are treated
    # as offset from a normal body-temperature baseline.
    if -10.0 <= value <= 10.0:
        return baseline + value
    return value


def format_value(value: Any, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value}{suffix}"


def format_minmax(data: dict[str, Any], key: str, suffix: str = "") -> str:
    min_value, max_value = minmax(data.get(key))
    return f"Min {format_value(min_value, suffix)}, Max {format_value(max_value, suffix)}"


def format_raw_sensor_data(data: dict[str, Any]) -> str:
    return "\n".join(
        [
            "===== Relay Raw Data =====",
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        ]
    )


def normalize_report_data(data: dict[str, Any]) -> dict[str, Any]:
    sensor_keys = {"Temp", "Breath", "dB", "IndoorTemp", "Humidity", "Bright", "SleepData"}
    if any(key in data for key in sensor_keys):
        return data

    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        if any(sensor_key in value for sensor_key in sensor_keys):
            merged = dict(data)
            merged.update(value)
            merged.setdefault("UID", key)
            return merged

    return data


def summarize_sensor_data(data: dict[str, Any], settings: Settings) -> str:
    lines = [
        "===== Relay Data Summary =====",
        f"Request: Type={settings.report_type}, Account={settings.account}, CMD={settings.cmd}, val={settings.val or '-'}, date={settings.date or '-'}",
    ]

    for key, label in (("name", "Name"), ("gender", "Gender"), ("birth_date", "Birth date"), ("last_update_at", "Last update")):
        if data.get(key) is not None:
            lines.append(f"{label}: {data[key]}")

    if isinstance(data.get("Temp"), dict):
        temp_min, temp_max = minmax(data.get("Temp"))
        body_min = normalize_body_temperature(temp_min, settings.body_temp_baseline)
        body_max = normalize_body_temperature(temp_max, settings.body_temp_baseline)
        lines.append(
            "Temp: "
            f"raw Min {format_value(temp_min)}, raw Max {format_value(temp_max)}; "
            f"body estimate Min {format_value(body_min, 'C')}, Max {format_value(body_max, 'C')}"
        )
    if isinstance(data.get("Breath"), dict):
        lines.append(f"Breath: {format_minmax(data, 'Breath')}")
    if isinstance(data.get("dB"), dict):
        _db_min, db_max = minmax(data.get("dB"))
        lines.append(f"dB: Max {format_value(db_max, 'dB')}")
    if isinstance(data.get("IndoorTemp"), dict):
        lines.append(f"IndoorTemp: {format_minmax(data, 'IndoorTemp', 'C')}")
    if isinstance(data.get("Humidity"), dict):
        lines.append(f"Humidity: {format_minmax(data, 'Humidity', '%')}")
    if isinstance(data.get("Bright"), dict):
        lines.append(f"Bright: {format_minmax(data, 'Bright')}")

    sleep_data = data.get("SleepData")
    if isinstance(sleep_data, dict):
        sessions = sleep_data.get("sessions")
        if isinstance(sessions, list):
            lines.append(f"Sleep sessions: {len(sessions)}")
            for index, session in enumerate(sessions, start=1):
                if isinstance(session, dict):
                    lines.append(
                        f"  {index}. {session.get('start', 'N/A')} -> {session.get('end', 'N/A')}, "
                        f"{session.get('duration_min', 'N/A')} min, wake_up {session.get('wake_up', 'N/A')}"
                    )
        for key in ("day_wakeup", "day_gs", "day_pr", "week_gs", "week_pr", "month_gs", "month_pr", "status"):
            if sleep_data.get(key) is not None:
                lines.append(f"Sleep {key}: {sleep_data[key]}")

    return "\n".join(lines)


def detect_abnormal_cases(data: dict[str, Any], queries: list[SensorQuery], baseline: float) -> list[SensorQuery]:
    temp_min, temp_max = minmax(data.get("Temp"))
    body_min = normalize_body_temperature(temp_min, baseline)
    body_max = normalize_body_temperature(temp_max, baseline)

    breath_min, breath_max = minmax(data.get("Breath"))
    _db_min, db_max = minmax(data.get("dB"))
    room_min, room_max = minmax(data.get("IndoorTemp"))
    humidity_min, humidity_max = minmax(data.get("Humidity"))

    matched_cases: set[str] = set()

    if body_max is not None and body_max > 39.0:
        matched_cases.add("very_high_body_temperature")
    elif body_max is not None and body_max > 37.8:
        matched_cases.add("high_body_temperature")
    if body_min is not None and body_min < 35.5:
        matched_cases.add("low_body_temperature")

    if db_max is not None and db_max > 85:
        matched_cases.add("very_loud_noise")
        matched_cases.add("high_noise")
    elif db_max is not None and db_max > 70:
        matched_cases.add("loud_noise")
        matched_cases.add("high_noise")

    if breath_max is not None and breath_max > 24:
        matched_cases.add("fast_breathing")
    if breath_min is not None and breath_min < 10:
        matched_cases.add("slow_breathing")
    if breath_max is not None and breath_max <= 0:
        matched_cases.add("no_breathing_detected")
        matched_cases.add("no_breathing_signal")

    if room_max is not None and room_max > 35.0:
        matched_cases.add("room_very_hot")
    elif room_max is not None and room_max > 30.0:
        matched_cases.add("room_too_hot")
    if room_min is not None and room_min < 16.0:
        matched_cases.add("room_too_cold")
    if room_min is not None and room_max is not None and room_max - room_min > 5.0:
        matched_cases.add("sudden_room_temperature_change")
    if (room_min is not None and room_min < 18.0) or (room_max is not None and room_max > 26.0):
        matched_cases.add("bad_sleep_temperature")

    if humidity_min is not None and humidity_min < 20:
        matched_cases.add("very_dry")
        matched_cases.add("room_too_dry")
    elif humidity_min is not None and humidity_min < 30:
        matched_cases.add("too_dry")
        matched_cases.add("room_too_dry")
    if humidity_max is not None and humidity_max > 80:
        matched_cases.add("very_humid")
        matched_cases.add("room_too_humid")
    elif humidity_max is not None and humidity_max > 70:
        matched_cases.add("too_humid")
        matched_cases.add("room_too_humid")
    if (humidity_min is not None and humidity_min < 40) or (humidity_max is not None and humidity_max > 60):
        matched_cases.add("bad_sleep_humidity")

    return [query for query in queries if query.case in matched_cases]


def abnormal_value_text(data: dict[str, Any], match: SensorQuery, baseline: float) -> str:
    temp_min, temp_max = minmax(data.get("Temp"))
    body_min = normalize_body_temperature(temp_min, baseline)
    body_max = normalize_body_temperature(temp_max, baseline)
    breath_min, breath_max = minmax(data.get("Breath"))
    _db_min, db_max = minmax(data.get("dB"))
    room_min, room_max = minmax(data.get("IndoorTemp"))
    humidity_min, humidity_max = minmax(data.get("Humidity"))

    value_by_case = {
        "high_body_temperature": (
            f"열화상 원본 최고값은 {format_value(temp_max)}이고, 추정 체온은 {format_value(body_max, '도')}입니다."
        ),
        "very_high_body_temperature": (
            f"열화상 원본 최고값은 {format_value(temp_max)}이고, 추정 체온은 {format_value(body_max, '도')}입니다."
        ),
        "low_body_temperature": (
            f"열화상 원본 최저값은 {format_value(temp_min)}이고, 추정 체온은 {format_value(body_min, '도')}입니다."
        ),
        "loud_noise": f"실내 소음 최고값은 {format_value(db_max, '데시벨')}입니다.",
        "very_loud_noise": f"실내 소음 최고값은 {format_value(db_max, '데시벨')}입니다.",
        "high_noise": f"실내 소음 최고값은 {format_value(db_max, '데시벨')}입니다.",
        "fast_breathing": f"호흡 최고값은 {format_value(breath_max)}입니다.",
        "slow_breathing": f"호흡 최저값은 {format_value(breath_min)}입니다.",
        "no_breathing_detected": f"호흡 최고값은 {format_value(breath_max)}입니다.",
        "no_breathing_signal": f"호흡 최고값은 {format_value(breath_max)}입니다.",
        "room_too_hot": f"실내 온도 최고값은 {format_value(room_max, '도')}입니다.",
        "room_very_hot": f"실내 온도 최고값은 {format_value(room_max, '도')}입니다.",
        "room_too_cold": f"실내 온도 최저값은 {format_value(room_min, '도')}입니다.",
        "sudden_room_temperature_change": (
            f"실내 온도 최저값은 {format_value(room_min, '도')}, 최고값은 {format_value(room_max, '도')}입니다."
        ),
        "bad_sleep_temperature": (
            f"실내 온도 최저값은 {format_value(room_min, '도')}, 최고값은 {format_value(room_max, '도')}입니다."
        ),
        "too_dry": f"실내 습도 최저값은 {format_value(humidity_min, '퍼센트')}입니다.",
        "very_dry": f"실내 습도 최저값은 {format_value(humidity_min, '퍼센트')}입니다.",
        "room_too_dry": f"실내 습도 최저값은 {format_value(humidity_min, '퍼센트')}입니다.",
        "too_humid": f"실내 습도 최고값은 {format_value(humidity_max, '퍼센트')}입니다.",
        "very_humid": f"실내 습도 최고값은 {format_value(humidity_max, '퍼센트')}입니다.",
        "room_too_humid": f"실내 습도 최고값은 {format_value(humidity_max, '퍼센트')}입니다.",
        "bad_sleep_humidity": (
            f"실내 습도 최저값은 {format_value(humidity_min, '퍼센트')}, 최고값은 {format_value(humidity_max, '퍼센트')}입니다."
        ),
    }
    return value_by_case.get(match.case, "")


class SensorTemplateValues(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "N/A"


def format_sensor_query_template(template: str, data: dict[str, Any], match: SensorQuery, baseline: float) -> str:
    temp_min, temp_max = minmax(data.get("Temp"))
    body_min = normalize_body_temperature(temp_min, baseline)
    body_max = normalize_body_temperature(temp_max, baseline)
    breath_min, breath_max = minmax(data.get("Breath"))
    _db_min, db_max = minmax(data.get("dB"))
    room_min, room_max = minmax(data.get("IndoorTemp"))
    humidity_min, humidity_max = minmax(data.get("Humidity"))

    name = data.get("name")
    if not isinstance(name, str) or not name:
        name = "대상자"

    body_temperature = body_min if match.case == "low_body_temperature" else body_max
    room_temperature = room_min if match.case == "room_too_cold" else room_max
    humidity = humidity_min if match.case in {"room_too_dry", "too_dry", "very_dry"} else humidity_max
    breathing_rate = breath_min if match.case == "slow_breathing" else breath_max

    values = SensorTemplateValues(
        name=name,
        body_temperature=format_value(body_temperature),
        thermal_raw_min=format_value(temp_min),
        thermal_raw_max=format_value(temp_max),
        noise_db=format_value(db_max),
        humidity=format_value(humidity),
        humidity_min=format_value(humidity_min),
        humidity_max=format_value(humidity_max),
        room_temperature=format_value(room_temperature),
        temperature_min=format_value(room_min),
        temperature_max=format_value(room_max),
        breathing_rate=format_value(breathing_rate),
        sleep_humidity=format_value(humidity),
        sleep_temperature=format_value(room_temperature),
    )
    return template.format_map(values)


def build_interview_text(data: dict[str, Any], settings: Settings, match: SensorQuery, prefix: str) -> str:
    if match.is_template:
        return format_sensor_query_template(match.query, data, match, settings.body_temp_baseline)

    value_text = abnormal_value_text(data, match, settings.body_temp_baseline)
    if value_text:
        return f"{prefix}{value_text} {match.query}"
    return f"{prefix}{match.query}"


async def speak(
    text: str,
    voice: str,
    speaker_device: str,
    playback_retries: int,
    playback_retry_delay: float,
) -> None:
    require_command("ffmpeg")
    require_command("aplay")
    with tempfile.TemporaryDirectory() as tmpdir:
        mp3_path = os.path.join(tmpdir, "alert.mp3")
        wav_path = os.path.join(tmpdir, "alert.wav")
        await edge_tts.Communicate(text, voice).save(mp3_path)

        ffmpeg = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                mp3_path,
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                wav_path,
            ],
            capture_output=True,
            text=True,
        )
        if ffmpeg.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {ffmpeg.stderr.strip()}")

        last_error = ""
        for attempt in range(playback_retries + 1):
            aplay = subprocess.run(
                ["aplay", "-q", "-D", speaker_device, wav_path],
                capture_output=True,
                text=True,
            )
            if aplay.returncode == 0:
                return

            last_error = (aplay.stderr or aplay.stdout or "").strip()
            if "Device or resource busy" not in last_error or attempt >= playback_retries:
                break

            wait_seconds = playback_retry_delay * (attempt + 1)
            print(f"Speaker busy; retrying playback in {wait_seconds:.1f}s...")
            await asyncio.sleep(wait_seconds)

        raise RuntimeError(f"aplay failed on {speaker_device}: {last_error}")


def transcribe_answer(asr: WhisperModel, audio: object) -> str:
    segments, _info = asr.transcribe(
        audio,
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


async def listen_for_answer(
    segmenter: SpeechSegmenter,
    mic_device: str,
    mic_retries: int,
    mic_retry_delay: float,
) -> object:
    last_error = ""
    for attempt in range(mic_retries + 1):
        try:
            with MicStream(mic_device) as mic:
                return segmenter.listen_once(mic)
        except RuntimeError as exc:
            last_error = str(exc)
            if "Device or resource busy" not in last_error or attempt >= mic_retries:
                break

            wait_seconds = mic_retry_delay * (attempt + 1)
            print(f"Microphone busy; retrying capture in {wait_seconds:.1f}s...")
            await asyncio.sleep(wait_seconds)

    raise RuntimeError(f"microphone capture failed on {mic_device}: {last_error}")


def append_dialog(settings: Settings, match: SensorQuery, question: str, answer: str) -> None:
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "account": settings.account,
        "sensor": match.sensor,
        "case": match.case,
        "condition": match.condition,
        "question": question,
        "answer": answer,
    }
    with open(settings.dialog_file, "a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


async def run_once(
    settings: Settings,
    queries: list[SensorQuery],
    segmenter: SpeechSegmenter | None,
    asr: WhisperModel | None,
) -> None:
    raw_data = fetch_sensor_data(settings)
    if settings.print_raw:
        print(format_raw_sensor_data(raw_data))
    data = normalize_report_data(raw_data)
    if settings.print_summary:
        print(summarize_sensor_data(data, settings))

    name = data.get("name")
    prefix = f"{name}님, " if isinstance(name, str) and name else ""
    matches = detect_abnormal_cases(data, queries, settings.body_temp_baseline)
    if matches:
        selected = matches[: settings.max_queries]
        if len(matches) > len(selected):
            print(f"Detected {len(matches)} abnormal cases; asking first {len(selected)}.")
        elif len(selected) < settings.max_queries:
            fill_count = settings.max_queries - len(selected)
            print(f"Detected {len(matches)} abnormal cases; adding {fill_count} general questions.")
            selected = [*selected, *GENERAL_QUERIES[:fill_count]]
    else:
        print(f"No abnormal sensor cases detected. Asking {settings.max_queries} general interview questions.")
        selected = GENERAL_QUERIES[: settings.max_queries]

    for match in selected:
        text = build_interview_text(data, settings, match, prefix)
        print(f"[{match.sensor}/{match.case}] {text}")
        if settings.dry_run:
            print("Dry run: skipped speaker output and microphone capture.")
            continue

        if segmenter is None or asr is None:
            raise RuntimeError("microphone capture is not initialized")

        await speak(
            text,
            settings.tts_voice,
            settings.speaker_device,
            settings.playback_retries,
            settings.playback_retry_delay,
        )
        print("Listening for answer...")
        audio = await listen_for_answer(
            segmenter,
            settings.mic_device,
            settings.mic_retries,
            settings.mic_retry_delay,
        )
        answer = transcribe_answer(asr, audio)
        print(f"Answer: {answer}")
        append_dialog(settings, match, text, answer)
        if settings.device_settle_seconds > 0:
            await asyncio.sleep(settings.device_settle_seconds)


async def main() -> None:
    settings = parse_args()
    queries = load_queries(settings.queries_file)
    segmenter = None
    asr = None

    if settings.print_reference:
        print(API_REFERENCE)

    if not settings.dry_run:
        segmenter = SpeechSegmenter(settings.silence_ms, settings.max_utterance_seconds)
        asr = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )

    while True:
        try:
            await run_once(settings, queries, segmenter, asr)
        except Exception as exc:
            print(f"Sensor check failed: {exc}")
        if settings.once:
            return
        await asyncio.sleep(settings.interval_seconds)


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="Sensor abnormal-case TTS announcer")
    parser.add_argument("--api-url", default=os.getenv("SENSOR_API_URL", API_URL))
    parser.add_argument("--type", dest="report_type", default=os.getenv("REPORT_TYPE", "LLMREPORT"))
    parser.add_argument("--account", default=os.getenv("ACCOUNT", "test5@test.com"))
    parser.add_argument("--cmd", default=os.getenv("CMD", "ALL"))
    parser.add_argument("--val", default=os.getenv("VAL"))
    parser.add_argument("--date", default=os.getenv("DATE"))
    parser.add_argument("--queries-file", default=os.getenv("QUERIES_FILE", "queries.txt"))
    parser.add_argument("--dialog-file", default=os.getenv("DIALOG_FILE", "dialog.txt"))
    parser.add_argument("--mic-device", default=os.getenv("MIC_DEVICE", "plughw:2,0"))
    parser.add_argument("--speaker-device", default=os.getenv("SPEAKER_DEVICE", "plughw:0,0"))
    parser.add_argument("--whisper-model", default=os.getenv("WHISPER_MODEL", "small"))
    parser.add_argument("--whisper-device", default=os.getenv("WHISPER_DEVICE", "cpu"))
    parser.add_argument("--whisper-compute-type", default=os.getenv("WHISPER_COMPUTE_TYPE", "int8"))
    parser.add_argument("--tts-voice", default=os.getenv("TTS_VOICE", "ko-KR-SunHiNeural"))
    parser.add_argument("--silence-ms", type=int, default=int(os.getenv("VAD_SILENCE_MS", "700")))
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=float(os.getenv("MAX_UTTERANCE_SECONDS", "20")),
    )
    parser.add_argument("--max-queries", type=int, default=int(os.getenv("MAX_QUERIES", "5")))
    parser.add_argument("--playback-retries", type=int, default=int(os.getenv("PLAYBACK_RETRIES", "3")))
    parser.add_argument(
        "--playback-retry-delay",
        type=float,
        default=float(os.getenv("PLAYBACK_RETRY_DELAY", "0.8")),
    )
    parser.add_argument(
        "--device-settle-seconds",
        type=float,
        default=float(os.getenv("DEVICE_SETTLE_SECONDS", "0.5")),
    )
    parser.add_argument("--mic-retries", type=int, default=int(os.getenv("MIC_RETRIES", "5")))
    parser.add_argument(
        "--mic-retry-delay",
        type=float,
        default=float(os.getenv("MIC_RETRY_DELAY", "0.8")),
    )
    parser.add_argument("--interval-seconds", type=float, default=float(os.getenv("INTERVAL_SECONDS", "60")))
    parser.add_argument("--body-temp-baseline", type=float, default=float(os.getenv("BODY_TEMP_BASELINE", "36.5")))
    parser.add_argument("--once", action="store_true", default=os.getenv("ONCE", "1") != "0")
    parser.add_argument("--loop", dest="once", action="store_false")
    parser.add_argument("--dry-run", action="store_true", default=os.getenv("DRY_RUN", "0") == "1")
    parser.add_argument("--no-reference", dest="print_reference", action="store_false")
    parser.add_argument("--no-raw", dest="print_raw", action="store_false")
    parser.add_argument("--no-summary", dest="print_summary", action="store_false")
    parser.set_defaults(
        print_reference=os.getenv("PRINT_REFERENCE", "1") != "0",
        print_raw=os.getenv("PRINT_RAW", "1") != "0",
        print_summary=os.getenv("PRINT_SUMMARY", "1") != "0",
    )
    args = parser.parse_args()
    return Settings(**vars(args))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
