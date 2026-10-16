//! Uploads in codecs the engine's decoder has none for, decoded to WAV before they reach it: Opus (Ogg, Matroska,
//! WebM) and AC-3 / E-AC-3 (Matroska).

use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, Packet, TimeBase};
use symphonia::core::{
    codecs::{CODEC_TYPE_NULL, CODEC_TYPE_OPUS},
    errors::Error as SymphoniaError,
    formats::{FormatOptions, FormatReader},
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

#[derive(Clone, Copy, PartialEq, Debug)]
enum Codec {
    Opus,
    Ac3,
    Eac3,
}

// each codec names itself near the start: `OpusHead` in Ogg, its codec id in Matroska
const SNIFF_BYTES: usize = 64 * 1024;
const MARKERS: [(&[u8], Codec); 4] = [
    (b"OpusHead", Codec::Opus),
    (b"A_OPUS", Codec::Opus),
    (b"A_EAC3", Codec::Eac3),
    (b"A_AC3", Codec::Ac3),
];
const OPUS_RATE: u32 = 48_000;
// OpusHead (RFC 7845): the channel count at byte 9, then the 48 kHz frames to drop from the start
const HEAD_CHANNELS: usize = 9;
const HEAD_PRE_SKIP: usize = 10;
// the longest Opus packet: 120 ms at 48 kHz
const MAX_PACKET_FRAMES: usize = 5760;
const PCM16_SCALE: f32 = 32768.0;
// the engine's own decoded-sample limit, 30 minutes of 48 kHz stereo, so a small file can't expand past it here
const MAX_DECODED_SAMPLES: usize = 48_000 * 2 * 30 * 60;

struct Pcm {
    samples: Vec<f32>,
    rate: u32,
    channels: usize,
}

impl Pcm {
    fn push(&mut self, samples: impl ExactSizeIterator<Item = f32>) -> anyhow::Result<()> {
        if self.samples.len() + samples.len() > MAX_DECODED_SAMPLES {
            anyhow::bail!("audio exceeds the {MAX_DECODED_SAMPLES} decoded sample limit");
        }
        self.samples.extend(samples);
        Ok(())
    }
}

/// `bytes` as a float WAV when they hold an Opus or AC-3 track; any other upload is returned as it came.
pub(crate) fn as_wav(bytes: Vec<u8>) -> anyhow::Result<Vec<u8>> {
    let head = &bytes[..bytes.len().min(SNIFF_BYTES)];
    let Some(codec) = MARKERS
        .iter()
        .find(|(marker, _)| head.windows(marker.len()).any(|w| w == *marker))
        .map(|&(_, codec)| codec)
    else {
        return Ok(bytes);
    };
    let stream = MediaSourceStream::new(
        Box::new(std::io::Cursor::new(bytes.clone())),
        Default::default(),
    );
    let Ok(probed) = symphonia::default::get_probe().format(
        &Hint::new(),
        stream,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    ) else {
        return Ok(bytes);
    };
    let mut format = probed.format;
    let pcm = match codec {
        Codec::Opus => opus(format.as_mut())?,
        Codec::Ac3 | Codec::Eac3 => dolby(format.as_mut(), codec)?,
    };
    match pcm {
        Some(pcm) => wav(&pcm),
        None => Ok(bytes),
    }
}

// each packet of `track`, until the input ends
fn each_packet(
    format: &mut dyn FormatReader,
    track: u32,
    mut each: impl FnMut(&[u8]) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    loop {
        match format.next_packet() {
            Ok(packet) if packet.track_id() == track => each(packet.buf())?,
            Ok(_) => {}
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        }
    }
}

fn opus(format: &mut dyn FormatReader) -> anyhow::Result<Option<Pcm>> {
    let Some(track) = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec == CODEC_TYPE_OPUS)
    else {
        return Ok(None);
    };
    let track_id = track.id;
    let header = track.codec_params.extra_data.as_deref().unwrap_or_default();
    let channels = header
        .get(HEAD_CHANNELS)
        .map(|&c| usize::from(c))
        .or_else(|| track.codec_params.channels.map(|c| c.count()))
        .unwrap_or(1);
    if channels > 2 {
        anyhow::bail!("Opus with {channels} channels isn't supported; only mono and stereo are");
    }
    let pre_skip = header
        .get(HEAD_PRE_SKIP..HEAD_PRE_SKIP + 2)
        .map_or(0, |b| usize::from(u16::from_le_bytes([b[0], b[1]])));
    let layout = if channels == 2 {
        opus::Channels::Stereo
    } else {
        opus::Channels::Mono
    };
    let mut decoder = opus::Decoder::new(OPUS_RATE, layout)?;
    let mut frame = vec![0f32; MAX_PACKET_FRAMES * channels];
    let mut pcm = Pcm {
        samples: Vec::new(),
        rate: OPUS_RATE,
        channels,
    };
    each_packet(format, track_id, |packet| {
        let frames = decoder.decode_float(packet, &mut frame, false)?;
        pcm.push(frame[..frames * channels].iter().copied())
    })?;
    pcm.samples
        .drain(..(pre_skip * channels).min(pcm.samples.len()));
    Ok(Some(pcm))
}

// symphonia reads the Matroska track but has no AC-3 codec for it, so the track shows as an audio one with none
fn dolby(format: &mut dyn FormatReader, codec: Codec) -> anyhow::Result<Option<Pcm>> {
    let Some((track_id, rate)) = format.tracks().iter().find_map(|t| {
        let rate = t.codec_params.sample_rate?;
        (t.codec_params.codec == CODEC_TYPE_NULL).then_some((t.id, rate))
    }) else {
        return Ok(None);
    };
    let mut registry = CodecRegistry::new();
    oxideav_ac3::register_codecs(&mut registry);
    let id = if codec == Codec::Eac3 {
        oxideav_ac3::CODEC_ID_STR_EAC3
    } else {
        oxideav_ac3::CODEC_ID_STR
    };
    let mut decoder = registry.first_decoder(&CodecParameters::audio(CodecId::new(id)))?;
    let mut pcm = Pcm {
        samples: Vec::new(),
        rate,
        channels: 0,
    };
    each_packet(format, track_id, |packet| {
        decoder.send_packet(&Packet::new(
            0,
            TimeBase::new(1, i64::from(rate)),
            packet.to_vec(),
        ))?;
        // a frame is the decoder's interleaved 16-bit samples, in the source's own channel count
        while let Ok(Frame::Audio(frame)) = decoder.receive_frame() {
            let bytes = &frame.data[0];
            if frame.samples > 0 {
                pcm.channels = bytes.len() / size_of::<i16>() / frame.samples as usize;
            }
            let (samples, _) = bytes.as_chunks::<2>();
            pcm.push(
                samples
                    .iter()
                    .map(|&b| f32::from(i16::from_le_bytes(b)) / PCM16_SCALE),
            )?;
        }
        Ok(())
    })?;
    if pcm.channels == 0 {
        anyhow::bail!("the AC-3 track holds no audio");
    }
    Ok(Some(pcm))
}

fn wav(pcm: &Pcm) -> anyhow::Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: u16::try_from(pcm.channels)?,
        sample_rate: pcm.rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut wav = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut wav, spec)?;
    for &sample in &pcm.samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(wav.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    // synthetic 0.5 s 440 Hz tones made with ffmpeg (`-f lavfi -i sine=440:duration=0.5`): mono Ogg Opus, and
    // stereo AC-3 in Matroska beside a 16x16 video track
    const OPUS_TONE: &[u8] = include_bytes!("../tests/data/tone_440hz.opus");
    const AC3_TONE: &[u8] = include_bytes!("../tests/data/tone_440hz_ac3.mkv");

    // the WAV's rate, channels and one channel's samples
    fn decoded(bytes: &[u8]) -> (u32, u16, Vec<f32>) {
        let wav = as_wav(bytes.to_vec()).unwrap();
        let mut reader = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
        let spec = reader.spec();
        let samples: Vec<f32> = reader.samples::<f32>().map(Result::unwrap).collect();
        let first = samples
            .iter()
            .step_by(usize::from(spec.channels))
            .copied()
            .collect();
        (spec.sample_rate, spec.channels, first)
    }

    fn assert_tone(samples: &[f32], rate: u32) {
        let seconds = samples.len() as f32 / rate as f32;
        assert!((seconds - 0.5).abs() < 0.04, "{seconds} s");
        // a 440 Hz tone crosses zero 880 times a second
        let middle = &samples[samples.len() / 4..samples.len() * 3 / 4];
        let crossings = middle
            .windows(2)
            .filter(|w| (w[0] < 0.) != (w[1] < 0.))
            .count();
        let per_second = crossings as f32 / (middle.len() as f32 / rate as f32);
        assert!(
            (per_second - 880.).abs() < 20.,
            "{per_second} crossings a second"
        );
    }

    #[test]
    fn an_opus_upload_becomes_a_wav_of_its_tone() {
        let (rate, channels, samples) = decoded(OPUS_TONE);
        assert_eq!((rate, channels), (OPUS_RATE, 1));
        assert_tone(&samples, rate);
    }

    #[test]
    fn an_ac3_track_in_a_video_becomes_a_wav_of_its_tone() {
        let (rate, channels, samples) = decoded(AC3_TONE);
        assert_eq!((rate, channels), (48_000, 2));
        assert_tone(&samples, rate);
    }

    #[test]
    fn other_audio_passes_through_untouched() {
        let wav = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        assert_eq!(as_wav(wav.clone()).unwrap(), wav);
    }
}
