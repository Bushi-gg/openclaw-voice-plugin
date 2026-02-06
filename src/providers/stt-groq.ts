/**
 * Groq Whisper STT Provider
 *
 * Uses Groq's OpenAI-compatible REST API for speech-to-text transcription.
 * Buffers mu-law audio from Twilio media streams, performs client-side VAD,
 * and sends accumulated audio to Groq's Whisper endpoint on silence detection.
 */

import type { RealtimeSTTSession } from "./stt-openai-realtime.js";

// ---------------------------------------------------------------------------
// mu-law decoding lookup table (ITU-T G.711)
// Converts 8-bit mu-law samples to 16-bit linear PCM.
// ---------------------------------------------------------------------------

const MULAW_DECODE_TABLE = new Int16Array(256);
(function buildMulawTable() {
  for (let i = 0; i < 256; i++) {
    let mulaw = ~i & 0xff;
    const sign = mulaw & 0x80 ? -1 : 1;
    const exponent = (mulaw >> 4) & 0x07;
    const mantissa = mulaw & 0x0f;
    let magnitude = ((mantissa << 1) | 0x21) << (exponent + 2);
    magnitude -= 0x84;
    MULAW_DECODE_TABLE[i] = sign * magnitude;
  }
})();

function decodeMulaw(mulawBuf: Buffer): Int16Array {
  const pcm = new Int16Array(mulawBuf.length);
  for (let i = 0; i < mulawBuf.length; i++) {
    pcm[i] = MULAW_DECODE_TABLE[mulawBuf[i]];
  }
  return pcm;
}

// ---------------------------------------------------------------------------
// WAV encoding (8 kHz, 16-bit mono PCM)
// ---------------------------------------------------------------------------

function encodeWav(pcmSamples: Int16Array, sampleRate = 8000): Buffer {
  const dataLen = pcmSamples.length * 2;
  const buf = Buffer.alloc(44 + dataLen);

  // RIFF header
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + dataLen, 4);
  buf.write("WAVE", 8);

  // fmt chunk
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16); // chunk size
  buf.writeUInt16LE(1, 20); // PCM format
  buf.writeUInt16LE(1, 22); // mono
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28); // byte rate
  buf.writeUInt16LE(2, 32); // block align
  buf.writeUInt16LE(16, 34); // bits per sample

  // data chunk
  buf.write("data", 36);
  buf.writeUInt32LE(dataLen, 40);
  for (let i = 0; i < pcmSamples.length; i++) {
    buf.writeInt16LE(pcmSamples[i], 44 + i * 2);
  }

  return buf;
}

// ---------------------------------------------------------------------------
// RMS energy calculation for VAD
// ---------------------------------------------------------------------------

function calculateRms(pcm: Int16Array): number {
  if (pcm.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < pcm.length; i++) {
    sum += pcm[i] * pcm[i];
  }
  return Math.sqrt(sum / pcm.length) / 32768; // normalize to 0–1
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export interface GroqSTTConfig {
  /** Groq API key */
  apiKey: string;
  /** Whisper model (default: whisper-large-v3-turbo) */
  model?: string;
  /** Base URL (default: https://api.groq.com/openai/v1) */
  baseUrl?: string;
  /** Silence duration in ms to trigger transcription (default: 800) */
  silenceDurationMs?: number;
  /** RMS energy threshold for speech detection (default: 0.01) */
  vadThreshold?: number;
}

const DEFAULT_MODEL = "whisper-large-v3-turbo";
const DEFAULT_BASE_URL = "https://api.groq.com/openai/v1";
const DEFAULT_SILENCE_MS = 800;
const DEFAULT_VAD_THRESHOLD = 0.01;

// Minimum audio duration to send for transcription (200ms at 8kHz)
const MIN_SPEECH_SAMPLES = 8000 * 0.2;

// ---------------------------------------------------------------------------
// Provider (factory)
// ---------------------------------------------------------------------------

export class GroqSTTProvider {
  readonly name = "groq";
  private apiKey: string;
  private model: string;
  private baseUrl: string;
  private silenceDurationMs: number;
  private vadThreshold: number;

  constructor(config: GroqSTTConfig) {
    if (!config.apiKey) {
      throw new Error("Groq API key required for Groq Whisper STT");
    }
    this.apiKey = config.apiKey;
    this.model = config.model || DEFAULT_MODEL;
    this.baseUrl = config.baseUrl || DEFAULT_BASE_URL;
    this.silenceDurationMs = config.silenceDurationMs || DEFAULT_SILENCE_MS;
    this.vadThreshold = config.vadThreshold || DEFAULT_VAD_THRESHOLD;
  }

  createSession(): RealtimeSTTSession {
    return new GroqSTTSession(
      this.apiKey,
      this.model,
      this.baseUrl,
      this.silenceDurationMs,
      this.vadThreshold,
    );
  }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

class GroqSTTSession implements RealtimeSTTSession {
  private connected = false;
  private closed = false;

  // Audio buffer
  private audioChunks: Buffer[] = [];
  private totalSamples = 0;

  // VAD state
  private isSpeaking = false;
  private silenceStartMs: number | null = null;
  private silenceTimer: ReturnType<typeof setTimeout> | null = null;

  // Callbacks
  private onTranscriptCallback: ((transcript: string) => void) | null = null;
  private onPartialCallback: ((partial: string) => void) | null = null;
  private onSpeechStartCallback: (() => void) | null = null;

  // Transcription in flight
  private transcribing = false;

  constructor(
    private readonly apiKey: string,
    private readonly model: string,
    private readonly baseUrl: string,
    private readonly silenceDurationMs: number,
    private readonly vadThreshold: number,
  ) {}

  async connect(): Promise<void> {
    this.closed = false;
    this.connected = true;
    console.log("[GroqSTT] Session connected (batch mode)");
  }

  sendAudio(muLawData: Buffer): void {
    if (!this.connected || this.closed) return;

    // Decode to PCM for energy analysis
    const pcm = decodeMulaw(muLawData);
    const energy = calculateRms(pcm);

    if (energy > this.vadThreshold) {
      // Speech detected
      if (!this.isSpeaking) {
        this.isSpeaking = true;
        this.silenceStartMs = null;
        this.clearSilenceTimer();
        console.log("[GroqSTT] Speech started");
        this.onSpeechStartCallback?.();
      }

      // Buffer the raw mu-law chunk
      this.audioChunks.push(Buffer.from(muLawData));
      this.totalSamples += muLawData.length;

      // Reset silence tracking
      this.silenceStartMs = null;
      this.clearSilenceTimer();
    } else if (this.isSpeaking) {
      // Still buffer audio during silence (captures trailing speech)
      this.audioChunks.push(Buffer.from(muLawData));
      this.totalSamples += muLawData.length;

      if (this.silenceStartMs === null) {
        this.silenceStartMs = Date.now();
        // Set a timer for silence detection
        this.silenceTimer = setTimeout(() => {
          this.onSilenceDetected();
        }, this.silenceDurationMs);
      }
    }
  }

  private clearSilenceTimer(): void {
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }
  }

  private onSilenceDetected(): void {
    if (!this.isSpeaking || this.transcribing) return;

    this.isSpeaking = false;
    this.silenceStartMs = null;
    this.clearSilenceTimer();

    if (this.totalSamples < MIN_SPEECH_SAMPLES) {
      // Too short — discard (likely noise)
      console.log("[GroqSTT] Audio too short, discarding");
      this.audioChunks = [];
      this.totalSamples = 0;
      return;
    }

    // Grab current buffer and reset
    const chunks = this.audioChunks;
    this.audioChunks = [];
    this.totalSamples = 0;

    // Notify partial with placeholder while transcribing
    this.onPartialCallback?.("...");

    // Transcribe asynchronously
    this.transcribing = true;
    this.transcribe(chunks)
      .then((text) => {
        if (text && !this.closed) {
          console.log(`[GroqSTT] Transcript: ${text}`);
          this.onTranscriptCallback?.(text);
        }
      })
      .catch((err) => {
        console.error("[GroqSTT] Transcription error:", err);
      })
      .finally(() => {
        this.transcribing = false;
      });
  }

  private async transcribe(chunks: Buffer[]): Promise<string | null> {
    // Concatenate mu-law chunks
    const mulaw = Buffer.concat(chunks);

    // Decode to PCM and encode as WAV
    const pcm = decodeMulaw(mulaw);
    const wav = encodeWav(pcm);

    const url = `${this.baseUrl.replace(/\/+$/, "")}/audio/transcriptions`;

    const form = new FormData();
    const blob = new Blob([wav], { type: "audio/wav" });
    form.append("file", blob, "audio.wav");
    form.append("model", this.model);
    form.append("response_format", "json");

    const res = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: form,
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`Groq STT failed (HTTP ${res.status}): ${detail}`);
    }

    const payload = (await res.json()) as { text?: string };
    return payload.text?.trim() || null;
  }

  onPartial(callback: (partial: string) => void): void {
    this.onPartialCallback = callback;
  }

  onTranscript(callback: (transcript: string) => void): void {
    this.onTranscriptCallback = callback;
  }

  onSpeechStart(callback: () => void): void {
    this.onSpeechStartCallback = callback;
  }

  async waitForTranscript(timeoutMs = 30000): Promise<string> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.onTranscriptCallback = null;
        reject(new Error("Transcript timeout"));
      }, timeoutMs);

      this.onTranscriptCallback = (transcript) => {
        clearTimeout(timeout);
        this.onTranscriptCallback = null;
        resolve(transcript);
      };
    });
  }

  close(): void {
    this.closed = true;
    this.connected = false;
    this.clearSilenceTimer();
    this.audioChunks = [];
    this.totalSamples = 0;
    console.log("[GroqSTT] Session closed");
  }

  isConnected(): boolean {
    return this.connected;
  }
}
