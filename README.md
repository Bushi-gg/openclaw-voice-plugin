# OpenClaw Voice Plugin (SAGE Fork)

Custom fork of the OpenClaw voice-call plugin with Groq Whisper STT integration.

## Features

- **Twilio Voice Calls** - Inbound/outbound phone calls
- **Groq Whisper STT** - Fast, cheap speech-to-text via Groq API
- **ElevenLabs TTS** - High-quality text-to-speech
- **VAD (Voice Activity Detection)** - Natural speech boundary detection
- **Tailscale Funnel** - Secure webhook exposure

## Configuration

```json
{
  "plugins": {
    "entries": {
      "voice-call": {
        "enabled": true,
        "config": {
          "provider": "twilio",
          "stt": {
            "provider": "groq",
            "model": "whisper-large-v3-turbo"
          },
          "tts": {
            "provider": "elevenlabs"
          },
          "twilio": {
            "accountSid": "AC...",
            "authToken": "..."
          }
        }
      }
    }
  }
}
```

## Groq STT Provider

The Groq STT provider (`src/providers/stt-groq.ts`) implements:

- **mu-law decoding** - Converts Twilio's mu-law audio to PCM
- **VAD** - RMS energy-based voice activity detection
- **Silence detection** - Configurable silence threshold (default 800ms)
- **Batch transcription** - Sends accumulated audio on silence

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `model` | `whisper-large-v3-turbo` | Groq Whisper model |
| `silenceDurationMs` | `800` | Silence duration to trigger transcription |
| `vadThreshold` | `0.01` | RMS energy threshold for speech detection |

## Upstream

Forked from: `@openclaw/voice-call` v2026.2.1

## License

MIT
