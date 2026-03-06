#!/usr/bin/env node

// cli.js: Stream a WAV file to Inworld Realtime API and record the response.

import fs from "fs";
import fetch from "node-fetch";
import minimist from "minimist";
import pkg from "@roamhq/wrtc";
const { RTCPeerConnection, nonstandard } = pkg;
import "dotenv/config";
import wav from "wav";

(async () => {
  const apiKey = process.env.INWORLD_API_KEY;
  const proxy = process.env.INWORLD_PROXY || "https://api.inworld.ai";

  // Parse command-line arguments
  const argv = minimist(process.argv.slice(2), {
    string: ["input", "output", "model"],
    alias: { i: "input", o: "output", m: "model" },
  });

  const inputRaw = argv.input;
  const inputPath = Array.isArray(inputRaw) ? inputRaw[inputRaw.length - 1] : inputRaw;

  const outputRaw = argv.output;
  const outputPath = Array.isArray(outputRaw) ? outputRaw[outputRaw.length - 1] : outputRaw || "combined.wav";

  const modelId = argv.model || "openai/gpt-4o-mini";

  if (!inputPath) {
    console.error(
      "Usage: cli.js --input <path/to.wav> [--output <out.wav>] [--model <model-id>]"
    );
    process.exit(1);
  }

  if (!apiKey) {
    console.error("Missing INWORLD_API_KEY in .env");
    process.exit(1);
  }

  // Read and decode WAV
  const reader = new wav.Reader();
  const inStream = fs.createReadStream(inputPath);
  let format;
  const pcmChunks = [];

  reader.on("format", (fmt) => {
    format = fmt;
  });
  reader.on("data", (data) => pcmChunks.push(data));

  await new Promise((resolve) => {
    reader.on("end", resolve);
    inStream.pipe(reader);
  });

  if (!format) {
    console.error("Failed to parse WAV format from input file.");
    process.exit(1);
  }

  // Flatten samples to Int16Array (downmix to mono)
  const { sampleRate: origSampleRate, bitDepth, channels } = format;
  const bytesPerSample = bitDepth / 8;
  const buffer = Buffer.concat(pcmChunks);
  const totalFrames = buffer.length / (bytesPerSample * channels);
  const samples = new Int16Array(totalFrames);
  for (let i = 0; i < totalFrames; i++) {
    let acc = 0;
    for (let c = 0; c < channels; c++) {
      const offset = (i * channels + c) * bytesPerSample;
      const sample = bitDepth === 32
        ? Math.round(Math.max(-1, Math.min(1, buffer.readFloatLE(offset))) * 32767)
        : buffer.readInt16LE(offset);
      acc += sample;
    }
    samples[i] = Math.round(acc / channels);
  }

  // Resample to 48000 Hz
  const targetRate = 48000;
  let resampled;
  if (origSampleRate !== targetRate) {
    const ratio = targetRate / origSampleRate;
    const newLen = Math.floor(samples.length * ratio);
    resampled = new Int16Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const idx = i / ratio;
      const i0 = Math.floor(idx);
      const i1 = Math.min(i0 + 1, samples.length - 1);
      const frac = idx - i0;
      resampled[i] = Math.round(samples[i0] * (1 - frac) + samples[i1] * frac);
    }
  } else {
    resampled = samples;
  }

  // Build PCM buffer
  const inputPcm = Buffer.alloc(resampled.length * 2);
  for (let i = 0; i < resampled.length; i++) {
    inputPcm.writeInt16LE(resampled[i], i * 2);
  }

  const sampleRate = targetRate;
  const frameSize = sampleRate / 100; // 480 samples per 10ms frame
  const frameBytes = frameSize * 2;

  // Fetch ICE servers
  const iceRes = await fetch(`${proxy}/v1/realtime/ice-servers`, {
    method: "GET",
    headers: { Authorization: `Bearer ${apiKey}` },
  });
  if (!iceRes.ok) {
    console.error("Failed to fetch ICE servers:", iceRes.status, await iceRes.text());
    process.exit(1);
  }
  const iceData = await iceRes.json();
  const iceServers = iceData.iceServers || iceData.ice_servers || [];

  // WebRTC setup with ICE servers
  const pc = new RTCPeerConnection({ iceServers });
  const source = new nonstandard.RTCAudioSource();
  const track = source.createTrack();
  pc.addTrack(track);
  const gptBuffers = []; // { samples: Int16Array, time: bigint }
  let done = false;

  pc.ontrack = ({ track: incomingTrack }) => {
    const sink = new nonstandard.RTCAudioSink(incomingTrack);
    sink.ondata = ({ samples }) => {
      gptBuffers.push({ samples: new Int16Array(samples), time: process.hrtime.bigint() });
    };
  };

  // Data channel: Inworld uses "oai-events" (hyphen)
  const dc = pc.createDataChannel("oai-events");

  dc.onopen = () => {
    const sessionUpdate = {
      type: "session.update",
      session: {
        modelId,
        instructions: "You are a helpful assistant.",
        output_modalities: ["audio", "text"],
        audio: {
          input: {
            turn_detection: {
              type: "semantic_vad",
              eagerness: "high",
              create_response: true,
              interrupt_response: true,
            },
          },
          output: {
            model: "inworld-tts-1.5-mini",
            voice: "Clive",
          },
        },
      },
    };
    dc.send(JSON.stringify(sessionUpdate));
  };

  dc.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "response.done") done = true;
    if (m.type === "error") console.error("[error]", JSON.stringify(m.error));
  };

  // SDP exchange
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const sigRes = await fetch(`${proxy}/v1/realtime/calls`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/sdp",
    },
    body: offer.sdp,
  });
  if (!sigRes.ok) {
    console.error("SDP exchange failed:", sigRes.status, await sigRes.text());
    process.exit(1);
  }
  const ans = await sigRes.text();
  await pc.setRemoteDescription({ type: "answer", sdp: ans });

  // Wait for data channel to open before streaming
  if (dc.readyState !== "open") {
    await new Promise((resolve) => {
      const origOnOpen = dc.onopen;
      dc.onopen = (e) => {
        if (origOnOpen) origOnOpen(e);
        resolve();
      };
    });
  }

  // Stream input audio at real-time pace
  const startTime = process.hrtime.bigint();
  for (let off = 0; off < inputPcm.length; off += frameBytes) {
    let chunk = inputPcm.slice(off, off + frameBytes);
    if (chunk.length < frameBytes) {
      const pad = Buffer.alloc(frameBytes - chunk.length);
      chunk = Buffer.concat([chunk, pad]);
    }
    const temp = new Int16Array(chunk.buffer, chunk.byteOffset, frameSize);
    const frame = Int16Array.from(temp);
    source.onData({ samples: frame, sampleRate, bitsPerSample: 16, channelCount: 1 });
    await new Promise((r) => setTimeout(r, 10));
  }

  // Wait a few seconds for any in-flight response to finish
  const gracePeriod = 5000; // 5s after input ends
  const waitStart = Date.now();
  while (!done && Date.now() - waitStart < gracePeriod) {
    // Keep feeding silence so the WebRTC track stays alive
    const silentFrame = new Int16Array(frameSize);
    source.onData({ samples: silentFrame, sampleRate, bitsPerSample: 16, channelCount: 1 });
    await new Promise((r) => setTimeout(r, 10));
  }

  // Concatenate all response audio chunks
  let totalGptSamples = 0;
  gptBuffers.forEach(({ samples }) => {
    totalGptSamples += samples.length;
  });

  const gptAudio = new Int16Array(totalGptSamples);
  let gptOffset = 0;
  gptBuffers.forEach(({ samples }) => {
    gptAudio.set(samples, gptOffset);
    gptOffset += samples.length;
  });

  // Truncate to input duration — we only care what the model said during the conversation
  const inputSamples = resampled.length;
  const outputAudio = gptAudio.length > inputSamples
    ? gptAudio.slice(0, inputSamples)
    : gptAudio;

  const gptPath = outputPath.replace(".wav", "_gpt_response.wav");
  const gptWriter = new wav.Writer({ sampleRate, channels: 1, bitDepth: 16 });
  const gptStream = fs.createWriteStream(gptPath);
  gptWriter.pipe(gptStream);
  gptWriter.write(Buffer.from(outputAudio.buffer, outputAudio.byteOffset, outputAudio.byteLength));
  gptWriter.end();

  await new Promise((resolve) => {
    gptStream.on("finish", resolve);
  });
  console.log(`Response saved to: ${gptPath}`);

  dc.close();
  pc.close();
  process.exit(0);
})();
