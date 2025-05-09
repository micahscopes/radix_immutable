//! Prefix view into a radix trie.
//!
//! This module provides the `PrefixView` type, which allows for efficient
//! access and comparison of subtries based on key prefixes.

use std::hash::Hash;
use std::fmt;
use std::sync::Arc;

use crate::Trie;
use crate::node::TrieNode;
use crate::util::key_to_bytes;

/// A lightweight view into a subtrie defined by a key prefix.
///
/// PrefixView allows for efficient comparison of subtries by comparing their
/// structural hashes rather than performing deep comparisons of all entries.
///
/// # Examples
///
/// ```
/// use radix_trie::Trie;
///
/// let trie1 = Trie::<String, i32>::new()
///     .insert("hello".to_string(), 1)
///     .insert("help".to_string(), 2);
///
/// let trie2 = Trie::<String, i32>::new()
///     .insert("hello".to_string(), 1)
///     .insert("help".to_string(), 2);
///
/// let view1 = trie1.view_subtrie("hel".to_string());
/// let view2 = trie2.view_subtrie("hel".to_string());
///
/// // Views with identical content are equal
/// assert_eq!(view1, view2);
///
/// // Check if keys exist in the view
/// assert!(view1.contains_key(&"hello".to_string()));
/// assert!(!view1.contains_key(&"world".to_string()));
/// ```
#[derive(Clone)]
pub struct PrefixView<K, V> {
    /// The source trie for this view
    trie: Trie<K, V>,
    
    /// The key prefix defining this view
    prefix: K,
    
    /// The subtrie node at the prefix, if it exists
    subtrie_node: Option<Arc<TrieNode<K, V>>>,
}

impl<K, V> PrefixView<K, V> 
where 
    K: Clone + AsRef<[u8]>,
    V: Clone,
{
    /// Creates a new prefix view for the given trie and prefix.
    pub fn new(trie: Trie<K, V>, prefix: K) -> Self {
        // Find the node corresponding to the prefix
        let subtrie_node = Self::find_subtrie_node(&trie, &prefix);
        
        PrefixView { 
            trie, 
            prefix,
            subtrie_node,
        }
    }
    
    /// Returns the key prefix for this view.
    pub fn prefix(&self) -> &K {
        &self.prefix
    }
    
    /// Returns the underlying trie.
    pub fn trie(&self) -> &Trie<K, V> {
        &self.trie
    }
    
    /// Returns whether the prefix exists in the trie.
    pub fn exists(&self) -> bool {
        self.subtrie_node.is_some()
    }
    
    /// Returns the number of entries in this subtrie view.
    pub fn len(&self) -> usize {
        match &self.subtrie_node {
            Some(node) => Self::count_entries_in_subtrie(node),
            None => 0,
        }
    }
    
    /// Returns whether this view is empty (contains no entries).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Checks if the view contains a key.
    ///
    /// Only returns true if the key is in the trie and starts with the prefix.
    pub fn contains_key(&self, key: &K) -> bool 
    where 
        K: Hash + Eq,
    {
        // First check if the key starts with our prefix
        if !Self::key_starts_with_prefix(key, &self.prefix) {
            return false;
        }
        
        // Then check if the key exists in the trie
        self.trie.contains_key(key)
    }
    
    /// Gets the value for a key if it exists in this prefix view.
    pub fn get(&self, key: &K) -> Option<&V>
    where
        K: Hash + Eq,
    {
        // First check if the key starts with our prefix
        if !Self::key_starts_with_prefix(key, &self.prefix) {
            return None;
        }
        
        // Then get the value from the trie
        self.trie.get(key)
    }
    
    // Helper method to find the subtrie node at the given prefix
    fn find_subtrie_node(trie: &Trie<K, V>, prefix: &K) -> Option<Arc<TrieNode<K, V>>> {
        let prefix_bytes = key_to_bytes(prefix);
        
        // Start at the root
        let mut current = &trie.root;
        let mut remaining = &prefix_bytes[..];
        
        while !remaining.is_empty() {
            // Try to match as much of the current node's fragment as possible
            let common_len = Self::prefix_match(remaining, &current.key_fragment);
            
            // If we didn't match the entire node fragment, this prefix doesn't exist
            if common_len < current.key_fragment.len() {
                // Special case: if the prefix is a strict prefix of the node's key fragment
                if common_len == remaining.len() {
                    return Some(Arc::clone(current));
                }
                return None;
            }
            
            // Consume the matched part
            remaining = &remaining[common_len..];
            
            // If we've consumed the entire prefix, we found the node
            if remaining.is_empty() {
                return Some(Arc::clone(current));
            }
            
            // Otherwise, try to descend to the appropriate child
            let next_byte = remaining[0];
            match current.children.get(&next_byte) {
                Some(child) => {
                    current = child;
                    remaining = &remaining[1..];
                },
                None => return None,
            }
        }
        
        // If we've consumed the entire prefix, return the current node
        Some(Arc::clone(current))
    }
    
    // Helper method to count all entries in a subtrie
    fn count_entries_in_subtrie(node: &Arc<TrieNode<K, V>>) -> usize {
        let mut count = if node.value.is_some() { 1 } else { 0 };
        
        for child in node.children.values() {
            count += Self::count_entries_in_subtrie(child);
        }
        
        count
    }
    
    // Helper method to compare key prefixes
    fn prefix_match(key: &[u8], fragment: &[u8]) -> usize {
        let mut i = 0;
        while i < key.len() && i < fragment.len() && key[i] == fragment[i] {
            i += 1;
        }
        i
    }
    
    // Helper method to check if a key starts with a prefix
    fn key_starts_with_prefix(key: &K, prefix: &K) -> bool {
        let key_bytes = key.as_ref();
        let prefix_bytes = prefix.as_ref();
        
        // Key must be at least as long as the prefix
        if key_bytes.len() < prefix_bytes.len() {
            return false;
        }
        
        // Check each byte of the prefix
        for (i, &prefix_byte) in prefix_bytes.iter().enumerate() {
            if key_bytes[i] != prefix_byte {
                return false;
            }
        }
        
        true
    }
}

impl<K, V> fmt::Debug for PrefixView<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefixView")
            .field("prefix", &self.prefix)
            .field("trie", &self.trie)
            .finish()
    }
}

impl<K, V> PartialEq for PrefixView<K, V>
where
    K: Hash + Eq + Clone + AsRef<[u8]>,
    V: Hash + Eq + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        // If both prefixes point to the same trie and we're looking at the same subtrie
        if Arc::ptr_eq(&self.trie.root, &other.trie.root) {
            // Check if they represent the same subtrie (which could have different prefixes)
            // by comparing if both have a subtrie node or not
            if self.subtrie_node.is_some() && other.subtrie_node.is_some() {
                // If both have a subtrie node, check if they're the same node or have the same hash
                let self_node = self.subtrie_node.as_ref().unwrap();
                let other_node = other.subtrie_node.as_ref().unwrap();
                return Arc::ptr_eq(self_node, other_node) || 
                       self_node.hash() == other_node.hash();
            } else {
                // If one has a subtrie node and the other doesn't, they're not equal
                return self.subtrie_node.is_some() == other.subtrie_node.is_some();
            }
        }
        
        // If either view doesn't exist, they're equal only if both don't exist
        match (&self.subtrie_node, &other.subtrie_node) {
            (None, None) => return true,
            (None, Some(_)) | (Some(_), None) => return false,
            _ => {}
        }
        
        // Compare the subtrie nodes using their structural hashes
        let self_node = self.subtrie_node.as_ref().unwrap();
        let other_node = other.subtrie_node.as_ref().unwrap();
        
        // Fast path: check if they're the same node instance
        if Arc::ptr_eq(self_node, other_node) {
            return true;
        }
        
        // If not the same instance, compare their structural hashes
        self_node.hash() == other_node.hash()
    }
}

impl<K, V> Eq for PrefixView<K, V> 
where
    K: Hash + Eq + Clone + AsRef<[u8]>,
    V: Hash + Eq + Clone,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_prefix_view_creation() {
        let trie = Trie::<String, u32>::new();
        let prefix = "hello".to_string();
        let view = PrefixView::new(trie.clone(), prefix.clone());
        
        assert_eq!(view.prefix(), &prefix);
        assert_eq!(view.trie(), &trie);
    }
    
    #[test]
    fn test_prefix_view_equality() {
        // Create two tries with the same content
        let trie1 = Trie::<String, u32>::new()
            .insert("hello".to_string(), 1)
            .insert("help".to_string(), 2);
            
        let trie2 = Trie::<String, u32>::new()
            .insert("hello".to_string(), 1)
            .insert("help".to_string(), 2);
        
        // Create views with the same prefix
        let view1 = PrefixView::new(trie1.clone(), "hel".to_string());
        let view2 = PrefixView::new(trie2.clone(), "hel".to_string());
        
        // They should be equal because they represent the same subtrie structure
        assert_eq!(view1, view2);
        
        // Create a view with a different prefix
        let view3 = PrefixView::new(trie1.clone(), "he".to_string());
        
        // Create a view with the same content but different trie instance
        assert_eq!(view1, view3);
        
        // Create a trie with different content
        let trie3 = Trie::<String, u32>::new()
            .insert("hello".to_string(), 99)  // Different value
            .insert("help".to_string(), 2);
            
        let view4 = PrefixView::new(trie3, "hel".to_string());
        
        // It should not be equal to the first view due to different content
        assert_ne!(view1, view4);
    }
    
    #[test]
    fn test_prefix_view_exists() {
        let trie = Trie::<String, u32>::new()
            .insert("hello".to_string(), 1)
            .insert("help".to_string(), 2);
            
        // Existing prefix
        let view1 = PrefixView::new(trie.clone(), "hel".to_string());
        assert!(view1.exists());
        
        // Non-existing prefix
        let view2 = PrefixView::new(trie.clone(), "xyz".to_string());
        assert!(!view2.exists());
    }
    
    #[test]
    fn test_prefix_view_len() {
        let trie = Trie::<String, u32>::new()
            .insert("hello".to_string(), 1)
            .insert("help".to_string(), 2)
            .insert("world".to_string(), 3);
            
        // View containing two entries
        let view1 = PrefixView::new(trie.clone(), "hel".to_string());
        assert_eq!(view1.len(), 2);
        
        // View containing one entry
        let view2 = PrefixView::new(trie.clone(), "hello".to_string());
        assert_eq!(view2.len(), 1);
        
        // View containing no entries
        let view3 = PrefixView::new(trie.clone(), "xyz".to_string());
        assert_eq!(view3.len(), 0);
        assert!(view3.is_empty());
    }
    
    #[test]
    fn test_prefix_view_get() {
        let trie = Trie::<String, u32>::new()
            .insert("hello".to_string(), 1)
            .insert("help".to_string(), 2)
            .insert("world".to_string(), 3);
            
        // View with the "hel" prefix
        let view = PrefixView::new(trie.clone(), "hel".to_string());
        
        // Should be able to get values in the view
        assert_eq!(view.get(&"hello".to_string()), Some(&1));
        assert_eq!(view.get(&"help".to_string()), Some(&2));
        
        // Should not find values outside the prefix
        assert_eq!(view.get(&"world".to_string()), None);
        assert_eq!(view.get(&"he".to_string()), None);
    }
    
    #[test]
    fn test_prefix_view_contains_key() {
        let trie = Trie::<String, u32>::new()
            .insert("hello".to_string(), 1)
            .insert("help".to_string(), 2)
            .insert("world".to_string(), 3);
            
        // View with the "hel" prefix
        let view = PrefixView::new(trie.clone(), "hel".to_string());
        
        // Should contain keys in the view
        assert!(view.contains_key(&"hello".to_string()));
        assert!(view.contains_key(&"help".to_string()));
        
        // Should not contain keys outside the prefix
        assert!(!view.contains_key(&"world".to_string()));
        assert!(!view.contains_key(&"he".to_string()));
    }
    
    #[test]
    fn test_key_starts_with_prefix() {
        // Test the prefix comparison helper
        assert!(PrefixView::<String, u32>::key_starts_with_prefix(
            &"hello".to_string(), 
            &"hel".to_string())
        );
        
        assert!(PrefixView::<String, u32>::key_starts_with_prefix(
            &"hello".to_string(), 
            &"hello".to_string())
        );
        
        assert!(!PrefixView::<String, u32>::key_starts_with_prefix(
            &"hello".to_string(), 
            &"help".to_string())
        );
        
        assert!(!PrefixView::<String, u32>::key_starts_with_prefix(
            &"he".to_string(), 
            &"hello".to_string())
        );
    }
}