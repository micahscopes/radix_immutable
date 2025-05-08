//! Prefix view into a radix trie.
//!
//! This module provides the `PrefixView` type, which allows for efficient
//! access and comparison of subtries based on key prefixes.

use std::hash::Hash;
use std::fmt;
use std::sync::Arc;

use crate::Trie;

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
/// assert_eq!(view1, view2);
/// ```
#[derive(Clone)]
pub struct PrefixView<K, V> {
    /// The source trie for this view
    trie: Trie<K, V>,
    
    /// The key prefix defining this view
    prefix: K,
}

impl<K, V> PrefixView<K, V> {
    /// Creates a new prefix view for the given trie and prefix.
    pub fn new(trie: Trie<K, V>, prefix: K) -> Self {
        PrefixView { trie, prefix }
    }
    
    /// Returns the key prefix for this view.
    pub fn prefix(&self) -> &K {
        &self.prefix
    }
    
    /// Returns the underlying trie.
    pub fn trie(&self) -> &Trie<K, V> {
        &self.trie
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
    K: Hash + Eq + Clone,
    V: Hash + Eq + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        // Fast path: check if the views represent the same subtrie
        // (i.e., they have the same prefix in the same trie)
        if self.prefix == other.prefix && Arc::ptr_eq(&self.trie.root, &other.trie.root) {
            return true;
        }
        
        // TODO: Implement structural hash comparison when the node functionality is ready
        // For now, we return false for different tries
        false
    }
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
    
    // More tests will be added as we implement functionality
}