# Local Voice Agent

Voice loop using:

- Silero VAD for speech detection
- Whisper `small` for STT
- Ollama `exaone3.5:2.4b` for generation
- Edge TTS for speech output
- Ollama chat mode with rolling dialog history
- ALSA USB mic/speaker devices

Default devices:

- Mic: `plughw:2,0` (`card 2: C10 [MATA STUDIO C10]`)
- Speaker: `plughw:0,0` (`card 0: UACDemoV10 [UACDemoV1.0]`)

## Install

```bash
sudo apt install -y alsa-utils ffmpeg
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
ollama pull exaone3.5:2.4b
```

## Run

```bash
. .venv/bin/activate
python voice_agent.py
```

Override devices or models if needed:

```bash
python voice_agent.py \
  --mic-device plughw:2,0 \
  --speaker-device plughw:0,0 \
  --ollama-model exaone3.5:2.4b \
  --whisper-model small \
  --tts-voice ko-KR-SunHiNeural \
  --max-history-turns 8
```

The assistant uses Ollama `/api/chat`, not one-shot `/api/generate`, so previous turns stay in context. Override the dialog behavior with `--system-prompt` or `SYSTEM_PROMPT`.

For a GPU Whisper backend, try:

```bash
python voice_agent.py --whisper-device cuda --whisper-compute-type float16
```

## Check ALSA Devices

```bash
arecord -l
aplay -l
```

If card numbers change after reboot, update `--mic-device` and `--speaker-device`.

## Sensor Abnormal TTS

`main.py` prints the relay API reference at startup, fetches the report data, prints the raw relay JSON, prints a readable summary, checks `queries.txt` against the sensor values, speaks up to 5 interview questions through Edge TTS, listens for the person's answer after each question, transcribes it with Whisper, and appends the dialog to `dialog.txt`.

If abnormal data exists, the interview uses abnormal-case questions from `queries.txt` and mentions the triggering sensor value before the question. If no abnormal data exists, it asks 5 general condition questions instead.

Equivalent PowerShell query:

```powershell
curl.exe -X POST "http://relay.emtake.com/api/query" -H "Content-Type: application/json" -d '{\"Type\":\"LLMREPORT\", \"Account\":\"test11@test.com\", \"CMD\":\"ALL\", \"val\":\"1\"}'
```

Dry-run without speaker output:

```bash
. .venv/bin/activate
python main.py --dry-run --account test11@test.com --cmd ALL --val 1
```

Run with speaker output:

```bash
python main.py \
  --account test11@test.com \
  --cmd ALL \
  --val 1 \
  --mic-device plughw:2,0 \
  --speaker-device plughw:0,0 \
  --max-queries 5 \
  --playback-retries 3
```

Defaults:

- `--type LLMREPORT`
- `--account test11@test.com`
- `--cmd ALL`
- `--queries-file queries.txt`
- `--dialog-file dialog.txt`
- `--mic-device plughw:2,0`
- `--speaker-device plughw:0,0`
- `--max-queries 5`
- `--playback-retries 3`
- `--playback-retry-delay 0.8`
- `--device-settle-seconds 0.5`

For senior reports, use `--type LLMREPORTS`.

Each line in `dialog.txt` is JSON containing timestamp, account, sensor, case, question, and transcribed answer.

Use `--no-reference` to skip the startup API reference, `--no-raw` to skip the raw relay JSON, or `--no-summary` to skip the fetched-data summary.

If ALSA reports `Device or resource busy`, `main.py` retries speaker playback before listening. Increase `--playback-retries` or `--device-settle-seconds` if the USB device releases slowly.

By default, `main.py` exits after one interview batch. Use `--loop` only when you want repeated polling with `--interval-seconds`.
