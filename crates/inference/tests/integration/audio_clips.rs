//! Audio the speech tests send: clips encoded as the HTTP routes receive them.

use inference::AudioInput;

const WAV_HEADER_BYTES: usize = 44;

// 16-bit mono PCM in a RIFF/WAVE container
pub fn wav(clip: &AudioInput) -> Vec<u8> {
    let data = clip.samples.len() * 2;
    let mut bytes = Vec::with_capacity(WAV_HEADER_BYTES + data);
    let u32_le = |value: usize| {
        u32::try_from(value)
            .expect("a test clip fits a WAV")
            .to_le_bytes()
    };
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&u32_le(WAV_HEADER_BYTES - 8 + data));
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&clip.sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(clip.sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&u32_le(data));
    for sample in &clip.samples {
        bytes.extend_from_slice(&((sample * f32::from(i16::MAX)) as i16).to_le_bytes());
    }
    bytes
}
