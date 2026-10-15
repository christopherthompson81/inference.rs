//! Opus uploads (Ogg, Matroska, WebM), decoded to WAV before they reach the engine, whose decoder has no Opus.

use symphonia::core::{
    codecs::CODEC_TYPE_OPUS, errors::Error as SymphoniaError, formats::FormatOptions,
    io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};

const OPUS_RATE: u32 = 48_000;
// the Opus header names its codec near the start: `OpusHead` in Ogg, the `A_OPUS` codec id in Matroska
const SNIFF_BYTES: usize = 64 * 1024;
const MARKERS: [&[u8]; 2] = [b"OpusHead", b"A_OPUS"];
// OpusHead (RFC 7845): the channel count at byte 9, then the 48 kHz frames to drop from the start
const HEAD_CHANNELS: usize = 9;
const HEAD_PRE_SKIP: usize = 10;
// the longest Opus packet: 120 ms at 48 kHz
const MAX_PACKET_FRAMES: usize = 5760;
// the engine's own decoded-sample limit, 30 minutes of 48 kHz stereo, so a small file can't expand past it here
const MAX_DECODED_SAMPLES: usize = 48_000 * 2 * 30 * 60;

/// `bytes` as a float WAV when they hold an Opus track; any other upload is returned as it came.
pub(crate) fn opus_as_wav(bytes: Vec<u8>) -> anyhow::Result<Vec<u8>> {
    let head = &bytes[..bytes.len().min(SNIFF_BYTES)];
    if !MARKERS
        .iter()
        .any(|marker| head.windows(marker.len()).any(|w| w == *marker))
    {
        return Ok(bytes);
    }
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
    let Some(track) = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec == CODEC_TYPE_OPUS)
    else {
        return Ok(bytes);
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
    let mut samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let frames = decoder.decode_float(packet.buf(), &mut frame, false)?;
        if samples.len() + frames * channels > MAX_DECODED_SAMPLES {
            anyhow::bail!("audio exceeds the {MAX_DECODED_SAMPLES} decoded sample limit");
        }
        samples.extend_from_slice(&frame[..frames * channels]);
    }
    let start = (pre_skip * channels).min(samples.len());

    let spec = hound::WavSpec {
        channels: u16::try_from(channels)?,
        sample_rate: OPUS_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut wav = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut wav, spec)?;
    for &sample in &samples[start..] {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(wav.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    // a synthetic 440 Hz tone, 0.5 s of mono Ogg Opus made with `ffmpeg -f lavfi -i sine=440:duration=0.5 -c:a libopus`
    const TONE: &[u8] = include_bytes!("../tests/data/tone_440hz.opus");

    #[test]
    fn an_opus_upload_becomes_a_wav_of_its_tone() {
        let wav = opus_as_wav(TONE.to_vec()).unwrap();
        let mut reader = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
        assert_eq!(
            (reader.spec().sample_rate, reader.spec().channels),
            (OPUS_RATE, 1)
        );
        let samples: Vec<f32> = reader.samples::<f32>().map(Result::unwrap).collect();
        let seconds = samples.len() as f32 / OPUS_RATE as f32;
        assert!((seconds - 0.5).abs() < 0.03, "{seconds} s");
        // a 440 Hz tone crosses zero 880 times a second
        let middle = &samples[samples.len() / 4..samples.len() * 3 / 4];
        let crossings = middle
            .windows(2)
            .filter(|w| (w[0] < 0.) != (w[1] < 0.))
            .count();
        let rate = crossings as f32 / (middle.len() as f32 / OPUS_RATE as f32);
        assert!((rate - 880.).abs() < 20., "{rate} crossings a second");
    }

    #[test]
    fn other_audio_passes_through_untouched() {
        let wav = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        assert_eq!(opus_as_wav(wav.clone()).unwrap(), wav);
    }
}
