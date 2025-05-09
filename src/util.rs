/// Finds the length of the common prefix between a key and a node's key fragment.
///
/// Returns the number of bytes that match starting from the given offset.
pub fn prefix_match(key: &[u8], start_idx: usize, node_key: &[u8]) -> usize {
    let mut i = 0;

    // Find how many bytes match between the key (starting at start_idx)
    // and the node's key fragment
    while i < node_key.len() && start_idx + i < key.len() && key[start_idx + i] == node_key[i] {
        i += 1;
    }

    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    #[test]
    fn test_prefix_match() {
        let key = b"abcdef";
        let node_key = b"abc";

        // They should match completely with the node_key
        assert_eq!(prefix_match(key, 0, node_key), 3);

        // Starting from index 1 (the 2nd byte), they should match 2 bytes
        let matching_key = b"bc";
        assert_eq!(prefix_match(key, 1, matching_key), 2);

        // Different keys
        let different_key = b"xyz";
        assert_eq!(prefix_match(key, 0, different_key), 0);
    }

    #[test]
    fn test_hash_key() {
        let key1 = "hello";
        let key2 = "hello";
        let key3 = "world";

        // Just testing the consistent behavior of Rust's hash implementation
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        let mut hasher3 = DefaultHasher::new();

        key1.hash(&mut hasher1);
        key2.hash(&mut hasher2);
        key3.hash(&mut hasher3);

        // Same keys should have the same hash
        assert_eq!(hasher1.finish(), hasher2.finish());

        // Different keys should have different hashes
        assert_ne!(hasher1.finish(), hasher3.finish());
    }
}
