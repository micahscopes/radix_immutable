//! The main trie implementation.
//!
//! This module contains the `Trie` type, which provides the primary API for working
//! with the radix trie data structure.

use std::sync::Arc;
use std::borrow::Borrow;
use std::hash::Hash;

use crate::node::TrieNode;
use crate::prefix_view::PrefixView;
use crate::util::{key_to_bytes, prefix_match};

/// An immutable radix trie with structural sharing.
///
/// This Radix Trie (also known as a Patricia Trie) is an ordered tree data structure
/// that efficiently stores and retrieves key-value pairs while preserving
/// the key ordering.
///
/// This implementation is immutable - all operations that would modify the trie
/// return a new trie instance that shares unchanged parts of the structure with
/// the original via `Arc`.
#[derive(Debug)]
pub struct Trie<K, V> {
    /// The root node of the trie
    pub(crate) root: Arc<TrieNode<K, V>>,
    
    /// The number of values stored in the trie
    size: usize,
}

impl<K: Clone, V: Clone> Clone for Trie<K, V> {
    fn clone(&self) -> Self {
        Trie {
            root: Arc::clone(&self.root),
            size: self.size,
        }
    }
}

impl<K, V> Trie<K, V> {
    /// Creates a new, empty trie.
    ///
    /// # Examples
    ///
    /// ```
    /// use radix_trie::Trie;
    ///
    /// let trie = Trie::<String, i32>::new();
    /// assert!(trie.is_empty());
    /// ```
    pub fn new() -> Self {
        Trie {
            root: Arc::new(TrieNode::new(Vec::new())),
            size: 0,
        }
    }
    
    /// Returns the number of values stored in the trie.
    ///
    /// # Examples
    ///
    /// ```
    /// use radix_trie::Trie;
    ///
    /// let trie = Trie::<String, i32>::new();
    /// assert_eq!(trie.len(), 0);
    ///
    /// let trie = trie.insert("hello".to_string(), 42);
    /// assert_eq!(trie.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.size
    }
    
    /// Returns `true` if the trie contains no values.
    ///
    /// # Examples
    ///
    /// ```
    /// use radix_trie::Trie;
    ///
    /// let trie = Trie::<String, i32>::new();
    /// assert!(trie.is_empty());
    ///
    /// let trie = trie.insert("hello".to_string(), 42);
    /// assert!(!trie.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    
    /// Creates a view of the subtrie at the given key prefix.
    ///
    /// This allows for efficient comparison of subtries.
    ///
    /// # Examples
    ///
    /// ```
    /// use radix_trie::Trie;
    ///
    /// let trie = Trie::<String, i32>::new()
    ///     .insert("hello".to_string(), 1)
    ///     .insert("help".to_string(), 2);
    ///
    /// let view = trie.view_subtrie("hel".to_string());
    /// ```
    pub fn view_subtrie<Q>(&self, prefix: K) -> PrefixView<K, V> 
    where 
        K: Clone,
        V: Clone,
    {
        // We'll implement this in a later step
        PrefixView::new(self.clone(), prefix)
    }
}

impl<K, V> Trie<K, V> 
where 
    K: Hash + Eq + Clone + AsRef<[u8]>,
    V: Clone,
{
    /// Retrieves a reference to the value stored for the given key, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use radix_trie::Trie;
    ///
    /// let trie = Trie::<String, i32>::new()
    ///     .insert("hello".to_string(), 42);
    ///
    /// assert_eq!(trie.get(&"hello".to_string()), Some(&42));
    /// assert_eq!(trie.get(&"world".to_string()), None);
    /// ```
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized + AsRef<[u8]>,
    {
        let key_bytes = key_to_bytes(key);
        let mut node = &self.root;
        let mut byte_idx = 0;
        
        loop {
            let match_len = prefix_match(&key_bytes, byte_idx, &node.key_fragment);
            
            // If we didn't match the entire node key fragment, this key isn't in the trie
            if match_len < node.key_fragment.len() {
                return None;
            }
            
            // Update how much of the search key we've matched
            byte_idx += match_len;
            
            // If we've matched the entire search key, return the value at this node (if any)
            if byte_idx == key_bytes.len() {
                return node.value.as_ref().map(|v| v.as_ref());
            }
            
            // Otherwise, we need to go deeper - get the next nibble and find the matching child
            let next_byte = key_bytes[byte_idx];
            match node.children.get(&next_byte) {
                Some(child) => {
                    node = child;
                    byte_idx += 1;
                },
                None => return None,
            }
        }
    }
    
    /// Returns `true` if the trie contains a value for the given key.
    ///
    /// # Examples
    ///
    /// ```
    /// use radix_trie::Trie;
    ///
    /// let trie = Trie::<String, i32>::new()
    ///     .insert("hello".to_string(), 42);
    ///
    /// assert!(trie.contains_key(&"hello".to_string()));
    /// assert!(!trie.contains_key(&"world".to_string()));
    /// ```
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized + AsRef<[u8]>,
    {
        self.get(key).is_some()
    }
    
    /// Inserts a key-value pair into the trie, returning a new trie.
    ///
    /// If the key already exists, the value is replaced and the old value is returned
    /// along with the new trie.
    ///
    /// # Examples
    ///
    /// ```
    /// use radix_trie::Trie;
    ///
    /// let trie1 = Trie::<String, i32>::new();
    /// let trie2 = trie1.insert("hello".to_string(), 42);
    ///
    /// assert!(trie1.is_empty());
    /// assert_eq!(trie2.get(&"hello".to_string()), Some(&42));
    /// ```
    pub fn insert(&self, _key: K, _value: V) -> Self {
        // We'll implement this in a later step
        // For now, just return a clone of self
        self.clone()
    }
    
    /// Removes a key-value pair from the trie, returning a new trie.
    ///
    /// If the key exists, the removed value is returned along with the new trie.
    ///
    /// # Examples
    ///
    /// ```
    /// use radix_trie::Trie;
    ///
    /// let trie1 = Trie::<String, i32>::new()
    ///     .insert("hello".to_string(), 42);
    ///
    /// let (trie2, removed_value) = trie1.remove(&"hello".to_string());
    ///
    /// assert_eq!(removed_value, Some(42));
    /// assert!(trie2.is_empty());
    /// ```
    pub fn remove<Q>(&self, _key: &Q) -> (Self, Option<V>)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized + AsRef<[u8]>,
    {
        // We'll implement this in a later step
        // For now, just return a clone of self and None
        (self.clone(), None)
    }
}

// Default implementation
impl<K, V> Default for Trie<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// Implement PartialEq to enable efficient comparison of tries
impl<K, V> PartialEq for Trie<K, V>
where
    K: Hash + Eq + Clone,
    V: Hash + Eq + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        // Fast path: check if they're the same instance
        if Arc::ptr_eq(&self.root, &other.root) {
            return true;
        }
        
        // If sizes differ, they can't be equal
        if self.size != other.size {
            return false;
        }
        
        // Compare structural hashes of the roots
        // This is much faster than comparing every entry
        self.root.hash() == other.root.hash()
    }
}

// Implement Eq for types where it makes sense
impl<K, V> Eq for Trie<K, V>
where
    K: Hash + Eq + Clone,
    V: Hash + Eq + Clone,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_trie() {
        let trie: Trie<String, u32> = Trie::new();
        assert!(trie.is_empty());
        assert_eq!(trie.len(), 0);
    }
    
    #[test]
    fn test_get_nonexistent() {
        let trie: Trie<String, u32> = Trie::new();
        assert_eq!(trie.get(&"hello".to_string()), None);
    }
    
    // Add more tests as we implement functionality
}