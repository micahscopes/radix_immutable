//! Utility functions for the radix trie implementation.
//!
//! This module contains helper functions for working with keys, bytes,
//! and other common operations needed throughout the trie implementation.

use std::hash::{Hash, Hasher};

/// Converts a key into a vector of bytes.
///
/// For types that can be represented as byte slices (like strings or byte arrays),
/// this simply returns the byte representation.
pub fn key_to_bytes<K: ?Sized>(key: &K) -> Vec<u8> 
where
    K: AsRef<[u8]>,
{
    key.as_ref().to_vec()
}

/// Finds the length of the common prefix between a key and a node's key fragment.
///
/// Returns the number of bytes that match starting from the given offset.
pub fn prefix_match(key: &[u8], start_idx: usize, node_key: &[u8]) -> usize {
    let mut i = 0;
    
    // Find how many bytes match between the key (starting at start_idx)
    // and the node's key fragment
    while i < node_key.len() && 
          start_idx + i < key.len() && 
          key[start_idx + i] == node_key[i] {
        i += 1;
    }
    
    i
}

/// Splits a byte slice at the given index.
///
/// Returns a tuple of two Vecs, one containing the bytes before the index
/// and one containing the bytes at and after the index.
pub fn split_key(key: &[u8], idx: usize) -> (Vec<u8>, Vec<u8>) {
    let prefix = key[..idx].to_vec();
    let suffix = key[idx..].to_vec();
    
    (prefix, suffix)
}

/// Calculates a deterministic hash for a key.
///
/// This is a utility function to ensure consistent hashing.
pub fn hash_key<K: Hash>(key: &K) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

/// Checks if two references point to the same allocation.
///
/// This is a wrapper around ptr::eq to improve code readability.
pub fn same_instance<T>(a: &T, b: &T) -> bool {
    std::ptr::eq(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_key_to_bytes() {
        let key = "abc";
        let bytes = key_to_bytes(key);
        
        assert_eq!(bytes, b"abc");
        assert_eq!(bytes.len(), 3);
    }
    
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
    fn test_split_key() {
        let key = b"abcdef";
        
        // Split after 'abc'
        let (prefix, suffix) = split_key(key, 3);
        
        assert_eq!(prefix.len(), 3);
        assert_eq!(suffix.len(), 3);
        
        // Verify the prefix is "abc"
        assert_eq!(prefix, b"abc");
        
        // Verify the suffix is "def"
        assert_eq!(suffix, b"def");
    }
    
    #[test]
    fn test_hash_key() {
        let key1 = "hello";
        let key2 = "hello";
        let key3 = "world";
        
        // Same keys should have the same hash
        assert_eq!(hash_key(&key1), hash_key(&key2));
        
        // Different keys should have different hashes
        assert_ne!(hash_key(&key1), hash_key(&key3));
    }
}