//! Audio utilities for `inference.rs`.
//!
//! This crate mirrors the `inference-vision` crate and focuses on audio specific
//! functionality such as reading audio data, resampling and computing
//! mel spectrogram features.

pub mod fft;
pub mod mel;
pub mod nemo;
pub mod whisper;

use anyhow::Result;
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CODEC_TYPE_NULL, DecoderOptions},
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

// 30 minutes of 48 kHz stereo; compressed input can otherwise expand to tens of GB
const MAX_DECODED_AUDIO_SAMPLES: usize = 48_000 * 2 * 30 * 60;

/// Raw audio input consisting of PCM samples and a sample rate.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioInput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioInput {
    /// Read a wav file from disk.
    pub fn read_wav(wav_path: &str) -> Result<Self> {
        let mut reader = hound::WavReader::open(wav_path)?;
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<std::result::Result<_, _>>()?,
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                // Match libsndfile/soundfile normalization for PCM16 by
                // dividing by the full signed range, not by `i16::MAX`.
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<std::result::Result<_, _>>()?,
        };
        Ok(Self {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        })
    }

    /// Decode audio bytes using `symphonia`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_limited(bytes, MAX_DECODED_AUDIO_SAMPLES)
    }

    fn from_bytes_limited(bytes: &[u8], max_samples: usize) -> Result<Self> {
        let cursor = std::io::Cursor::new(bytes.to_vec());
        let mut decoder = Decoder::open(Box::new(cursor), bytes.len() as u64)?;
        // the stated length up front: grown by doubling, a long recording's buffer held up to twice its samples
        let mut samples = Vec::with_capacity(decoder.stated_samples().min(max_samples));
        let mut packet = Vec::new();
        while decoder.next(&mut packet)? {
            if samples.len() + packet.len() > max_samples {
                anyhow::bail!("audio exceeds the {max_samples} decoded sample limit");
            }
            samples.extend_from_slice(&packet);
        }
        samples.shrink_to_fit();
        Ok(Self {
            samples,
            sample_rate: decoder.sample_rate,
            channels: decoder.channels,
        })
    }

    /// Decodes the audio file at `path` to mono at `sample_rate` as it reads, so a long recording at another rate or
    /// in stereo never sits whole in memory as decoded; at most `max_samples` samples out.
    pub fn from_file_mono(
        path: &std::path::Path,
        sample_rate: u32,
        max_samples: usize,
    ) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        let mut decoder = Decoder::open(Box::new(file), len)?;
        let mut resampler = mel::StreamResampler::new(decoder.sample_rate, sample_rate)?;
        let ratio = f64::from(sample_rate) / f64::from(decoder.sample_rate);
        let stated = (decoder.stated_frames() as f64 * ratio) as usize;
        let mut samples = Vec::with_capacity(stated.min(max_samples));
        let (mut packet, mut mono) = (Vec::new(), Vec::new());
        while decoder.next(&mut packet)? {
            let channels = usize::from(decoder.channels.max(1));
            mono.clear();
            mono.extend(
                packet
                    .chunks_exact(channels)
                    .map(|frame| frame.iter().sum::<f32>() / channels as f32),
            );
            let out = resampler.push(&mono)?;
            if samples.len() + out.len() > max_samples {
                anyhow::bail!("audio exceeds the {max_samples} decoded sample limit");
            }
            samples.extend_from_slice(&out);
        }
        samples.extend(resampler.finish()?);
        samples.truncate(max_samples);
        samples.shrink_to_fit();
        Ok(Self {
            samples,
            sample_rate,
            channels: 1,
        })
    }

    /// Convert multi channel audio to mono by averaging channels.
    /// [`Self::to_mono`], borrowed when the samples are mono already.
    pub fn mono(&self) -> std::borrow::Cow<'_, [f32]> {
        if self.channels <= 1 {
            return std::borrow::Cow::Borrowed(&self.samples);
        }
        std::borrow::Cow::Owned(self.to_mono())
    }

    pub fn to_mono(&self) -> Vec<f32> {
        if self.channels <= 1 {
            return self.samples.clone();
        }
        let mut mono = vec![0.0; self.samples.len() / self.channels as usize];
        for (i, sample) in self.samples.iter().enumerate() {
            mono[i / self.channels as usize] += *sample;
        }
        for s in &mut mono {
            *s /= self.channels as f32;
        }
        mono
    }

    /// Normalize audio to prevent clipping
    pub fn normalize(&mut self) -> &mut Self {
        let max_amplitude = self.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if max_amplitude > 0.0 && max_amplitude != 1.0 {
            let scale = 1.0 / max_amplitude;
            for sample in &mut self.samples {
                *sample *= scale;
            }
        }
        self
    }

    /// Apply fade in/out to reduce audio artifacts
    pub fn apply_fade(&mut self, fade_in_samples: usize, fade_out_samples: usize) -> &mut Self {
        let len = self.samples.len();
        // Fade in
        for i in 0..fade_in_samples.min(len) {
            let factor = i as f32 / fade_in_samples as f32;
            self.samples[i] *= factor;
        }
        // Fade out
        for i in 0..fade_out_samples.min(len) {
            let factor = (fade_out_samples - i) as f32 / fade_out_samples as f32;
            self.samples[len - 1 - i] *= factor;
        }
        self
    }

    /// Remove DC offset (audio centered around 0)
    pub fn remove_dc_offset(&mut self) -> &mut Self {
        if self.samples.is_empty() {
            return self;
        }
        let mean = self.samples.iter().sum::<f32>() / self.samples.len() as f32;
        for sample in &mut self.samples {
            *sample -= mean;
        }
        self
    }
}

// a container's first audio track, decoded a packet at a time
struct Decoder {
    format: Box<dyn symphonia::core::formats::FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    track_id: u32,
    sample_rate: u32,
    // the last packet's, which Matroska may leave to the codec's own header (AAC's does)
    channels: u16,
    frames: Option<u64>,
}

impl Decoder {
    // `len`: the source's bytes, bounding a packet where the container states no size
    fn open(source: Box<dyn symphonia::core::io::MediaSource>, len: u64) -> Result<Self> {
        let mss = MediaSourceStream::new(source, Default::default());
        let probed = symphonia::default::get_probe().format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )?;
        let format = probed.format;
        // a video's default track is its picture, so the first track with a sample rate is the audio
        let track = format
            .tracks()
            .iter()
            .find(|t| {
                t.codec_params.codec != CODEC_TYPE_NULL && t.codec_params.sample_rate.is_some()
            })
            .ok_or_else(|| anyhow::anyhow!("no audio track in a codec this server decodes"))?;
        let track_id = track.id;
        let mut codec_params = track.codec_params.clone();
        // Matroska leaves PCM's packet size unstated; no packet holds more frames than the whole input
        if codec_params.max_frames_per_packet.is_none() {
            let channels = codec_params.channels.map_or(1, |c| c.count());
            let sample_bytes = codec_params
                .bits_per_coded_sample
                .map_or(1, |bits| bits.div_ceil(8));
            codec_params.max_frames_per_packet =
                Some(len / (channels as u64 * u64::from(sample_bytes)) + 1);
        }
        let sample_rate = codec_params
            .sample_rate
            .ok_or_else(|| anyhow::anyhow!("unknown sample rate"))?;
        #[allow(clippy::cast_possible_truncation)]
        let channels = codec_params.channels.map(|c| c.count() as u16).unwrap_or(1);
        let decoder =
            symphonia::default::get_codecs().make(&codec_params, &DecoderOptions::default())?;
        Ok(Self {
            format,
            decoder,
            track_id,
            sample_rate,
            channels,
            frames: codec_params.n_frames,
        })
    }

    // the frames the container states, 0 when it states none
    fn stated_frames(&self) -> usize {
        self.frames.unwrap_or(0) as usize
    }

    fn stated_samples(&self) -> usize {
        self.stated_frames() * usize::from(self.channels)
    }

    // the next packet's interleaved samples into `out`; false at the end
    fn next(&mut self, out: &mut Vec<f32>) -> Result<bool> {
        loop {
            match self.format.next_packet() {
                Ok(packet) if packet.track_id() != self.track_id => {}
                Ok(packet) => {
                    let decoded = self.decoder.decode(&packet)?;
                    self.channels = decoded.spec().channels.count().try_into()?;
                    let mut buf =
                        SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                    buf.copy_interleaved_ref(decoded);
                    out.clear();
                    out.extend_from_slice(buf.samples());
                    return Ok(true);
                }
                Err(symphonia::core::errors::Error::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(false);
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AudioInput;
    use hound::{SampleFormat, WavSpec, WavWriter};
    use std::io::Cursor;

    // a 48 kHz stereo file decoded to 16 kHz mono as it reads: as decoding it whole, downmixing and resampling, and
    // capped where asked
    #[test]
    fn a_file_decodes_to_mono_at_the_asked_rate() {
        const RATE: u32 = 48_000;
        const FRAMES: usize = 3 * RATE as usize + 77;
        let dir = std::env::temp_dir().join(format!("inference-audio-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stereo.wav");
        let spec = WavSpec {
            channels: 2,
            sample_rate: RATE,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&path, spec).unwrap();
        for i in 0..FRAMES {
            let t = i as f32 / RATE as f32;
            writer
                .write_sample(((t * 440.0 * std::f32::consts::TAU).sin() * 9000.0) as i16)
                .unwrap();
            writer
                .write_sample(((t * 660.0 * std::f32::consts::TAU).sin() * 9000.0) as i16)
                .unwrap();
        }
        writer.finalize().unwrap();
        let streamed = AudioInput::from_file_mono(&path, 16_000, usize::MAX).unwrap();
        assert_eq!((streamed.sample_rate, streamed.channels), (16_000, 1));
        assert_eq!(
            streamed.samples.len(),
            (FRAMES as f64 / 3.0).round() as usize
        );
        let whole = AudioInput::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
        let mono = whole.to_mono();
        let want = crate::mel::resample(&mono, RATE, 16_000).unwrap();
        let worst = streamed
            .samples
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-3, "worst {worst}");
        assert!(AudioInput::from_file_mono(&path, 16_000, 1000).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn read_wav_roundtrip() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create("/tmp/test.wav", spec).unwrap();
        for _ in 0..160 {
            writer.write_sample::<i16>(0).unwrap();
        }
        writer.finalize().unwrap();
        let input = AudioInput::read_wav("/tmp/test.wav").unwrap();
        assert_eq!(input.samples.len(), 160);
        assert_eq!(input.sample_rate, 16000);
        std::fs::remove_file("/tmp/test.wav").unwrap();
    }

    #[test]
    fn read_wav_matches_pcm16_full_scale_normalization() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create("/tmp/test_full_scale.wav", spec).unwrap();
        writer.write_sample::<i16>(i16::MIN).unwrap();
        writer.write_sample::<i16>(i16::MAX).unwrap();
        writer.finalize().unwrap();

        let input = AudioInput::read_wav("/tmp/test_full_scale.wav").unwrap();
        assert_eq!(input.samples, vec![-1.0, 32767.0 / 32768.0]);

        std::fs::remove_file("/tmp/test_full_scale.wav").unwrap();
    }

    #[test]
    fn from_bytes() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 8000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut buffer: Vec<u8> = Vec::new();
        {
            let mut writer = WavWriter::new(Cursor::new(&mut buffer), spec).unwrap();
            for _ in 0..80 {
                writer.write_sample::<i16>(0).unwrap();
            }
            writer.finalize().unwrap();
        }
        let input = AudioInput::from_bytes(&buffer).unwrap();
        assert_eq!(input.samples.len(), 80);
        assert_eq!(input.sample_rate, 8000);
        assert!(AudioInput::from_bytes_limited(&buffer, 79).is_err());
        assert!(AudioInput::from_bytes_limited(&buffer, 80).is_ok());
    }

    // one EBML element, its size always written in the 8-byte form
    fn element(id: u32, body: &[u8]) -> Vec<u8> {
        let id = id.to_be_bytes();
        let start = id.iter().position(|&b| b != 0).unwrap();
        let mut out = id[start..].to_vec();
        out.push(0x01);
        out.extend_from_slice(&(body.len() as u64).to_be_bytes()[1..]);
        out.extend_from_slice(body);
        out
    }

    fn uint(id: u32, value: u64) -> Vec<u8> {
        element(id, &value.to_be_bytes())
    }

    // a simple block of the cluster: track number as a 1-byte vint, timestamp 0, keyframe
    fn block(track: u8, data: &[u8]) -> Vec<u8> {
        element(0xA3, &[&[0x80 | track, 0, 0, 0x80][..], data].concat())
    }

    // a video's picture track comes first, so the audio is not its default track
    #[test]
    fn from_bytes_reads_the_audio_track_of_a_matroska_video() {
        let pcm: Vec<i16> = (0..64).map(|i| (i * 300 - 9600) as i16).collect();
        let pcm_bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        let header = element(
            0x1A45_DFA3,
            &[
                uint(0x4286, 1),
                uint(0x42F7, 1),
                uint(0x42F2, 4),
                uint(0x42F3, 8),
                element(0x4282, b"matroska"),
                uint(0x4287, 4),
                uint(0x4285, 2),
            ]
            .concat(),
        );
        let video = element(
            0xAE,
            &[
                uint(0xD7, 1),
                uint(0x73C5, 1),
                uint(0x83, 1),
                element(0x86, b"V_UNCOMPRESSED"),
                element(0xE0, &[uint(0xB0, 2), uint(0xBA, 2)].concat()),
            ]
            .concat(),
        );
        let audio = element(
            0xAE,
            &[
                uint(0xD7, 2),
                uint(0x73C5, 2),
                uint(0x83, 2),
                element(0x86, b"A_PCM/INT/LIT"),
                element(
                    0xE1,
                    &[
                        element(0xB5, &16000f64.to_be_bytes()),
                        uint(0x9F, 2),
                        uint(0x6264, 16),
                    ]
                    .concat(),
                ),
            ]
            .concat(),
        );
        let segment = element(
            0x1853_8067,
            &[
                element(0x1549_A966, &uint(0x2A_D7B1, 1_000_000)),
                element(0x1654_AE6B, &[video, audio].concat()),
                element(
                    0x1F43_B675,
                    &[uint(0xE7, 0), block(1, &[0; 12]), block(2, &pcm_bytes)].concat(),
                ),
            ]
            .concat(),
        );
        let input = AudioInput::from_bytes(&[header, segment].concat()).unwrap();
        assert_eq!((input.sample_rate, input.channels), (16000, 2));
        let want: Vec<f32> = pcm.iter().map(|&s| f32::from(s) / 32768.0).collect();
        assert_eq!(input.samples, want);
    }

    #[test]
    fn test_normalize() {
        let mut input = AudioInput {
            samples: vec![0.2, -0.5, 0.8, -1.0],
            sample_rate: 16000,
            channels: 1,
        };
        input.normalize();
        let max = input.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!((max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_remove_dc_offset() {
        let mut input = AudioInput {
            samples: vec![1.0, 1.0, 1.0, 1.0],
            sample_rate: 16000,
            channels: 1,
        };
        input.remove_dc_offset();
        for s in input.samples {
            assert!((s - 0.0).abs() < 1e-6);
        }
    }
}
