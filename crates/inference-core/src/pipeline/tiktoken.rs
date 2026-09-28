use ahash::AHashMap;

pub(super) fn bytes_to_unicode() -> AHashMap<u8, char> {
    // Create the mapping from bytes to unicode characters.
    // Matches Python's openai/tiktoken bytes_to_unicode().
    let mut bs: Vec<u8> = vec![];

    // Add printable ASCII range
    bs.extend((b'!'..=b'~').collect::<Vec<_>>());
    // Add extended Latin range 1
    bs.extend((0xA1u8..=0xACu8).collect::<Vec<_>>());
    // Add extended Latin range 2
    bs.extend((0xAEu8..=0xFFu8).collect::<Vec<_>>());

    // cs stores the unicode codepoints (may be > 255 for non-printable bytes)
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;

    // Add remaining bytes not in the initial ranges, mapping them to 256+
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    // Create the mapping
    let mut byte_encoder = AHashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        byte_encoder.insert(*b, char::from_u32(*c).unwrap());
    }

    byte_encoder
}

pub(super) fn token_bytes_to_string(bytes: &[u8]) -> String {
    let byte_encoder = bytes_to_unicode();
    bytes.iter().map(|&b| byte_encoder[&b]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_unicode() {
        let byte_encoder = bytes_to_unicode();

        // Test that we have mappings for all 256 bytes
        assert_eq!(byte_encoder.len(), 256);

        // Test specific mappings
        assert_eq!(byte_encoder[&b'h'], 'h');
        assert_eq!(byte_encoder[&b'e'], 'e');
        assert_eq!(byte_encoder[&b'l'], 'l');
        assert_eq!(byte_encoder[&b'o'], 'o');

        // Test that all bytes map to valid chars
        for b in 0u8..=255 {
            assert!(byte_encoder.contains_key(&b));
        }
    }

    #[test]
    fn test_token_bytes_to_string() {
        let test_bytes = b"hello";
        let result = token_bytes_to_string(test_bytes);
        assert_eq!(result, "hello");
    }
}
