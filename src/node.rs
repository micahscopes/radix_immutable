//! Internal node implementation for the radix trie.
//!
//! This module contains the internal `TrieNode` structure that forms the backbone
//! of the radix trie implementation. `TrieNode` instances are always wrapped in an
//! `Arc` to enable structural sharing.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// Internal node type for the radix trie.
///
/// This type is not exposed directly in the public API but is used internally
/// by the `Trie` type. Each node contains a key fragment, an optional value,
/// and a map of children.
#[derive(Debug)]
pub(crate) struct TrieNode<K, V> {
    /// The key fragment stored at this node (as a sequence of bytes)
    pub key_fragment: Vec<u8>,
    
    /// The value stored at this node, if any
    pub value: Option<Arc<V>>,
    
    /// Child nodes indexed by the first byte of their key fragment
    pub children: HashMap<u8, Arc<TrieNode<K, V>>>,
    
    /// Cached structural hash of this node
    ///
    /// This is used to optimize comparisons between nodes and subtrees.
    /// The hash is calculated lazily and cached for future use.
    pub cached_hash: Mutex<Option<u64>>,
    
    /// Phantom data to carry the key type
    _key_type: std::marker::PhantomData<K>,
}

impl<K, V> TrieNode<K, V> {
    /// Creates a new empty node with the given key fragment
    pub fn new(key_fragment: Vec<u8>) -> Self {
        TrieNode {
            key_fragment,
            value: None,
            children: HashMap::new(),
            cached_hash: Mutex::new(None),
            _key_type: std::marker::PhantomData,
        }
    }
    
    /// Creates a new empty node with the given key fragment and value
    pub fn with_value(key_fragment: Vec<u8>, value: V) -> Self {
        TrieNode {
            key_fragment,
            value: Some(Arc::new(value)),
            children: HashMap::new(),
            cached_hash: Mutex::new(None),
            _key_type: std::marker::PhantomData,
        }
    }
    
    /// Returns the number of values stored in this subtree
    pub fn subtree_size(&self) -> usize {
        let mut count = if self.value.is_some() { 1 } else { 0 };
        
        for child in self.children.values() {
            count += child.subtree_size();
        }
        
        count
    }
    
    /// Clears the cached hash, forcing it to be recalculated next time it's needed
    pub fn invalidate_hash_cache(&self) {
        if let Ok(mut cache) = self.cached_hash.lock() {
            *cache = None;
        }
    }
    
    /// Returns whether this node is a leaf node (has no children)
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
    
    /// Creates a clone of this node, but with a new key fragment
    pub fn with_key_fragment(&self, key_fragment: Vec<u8>) -> Self {
        TrieNode {
            key_fragment,
            value: self.value.clone(),
            children: self.children.clone(),
            cached_hash: Mutex::new(None), // Reset cache for the new node
            _key_type: std::marker::PhantomData,
        }
    }
    
    /// Creates a clone of this node, but with a new value
    pub fn with_value_option(&self, value: Option<Arc<V>>) -> Self {
        TrieNode {
            key_fragment: self.key_fragment.clone(),
            value,
            children: self.children.clone(),
            cached_hash: Mutex::new(None), // Reset cache for the new node
            _key_type: std::marker::PhantomData,
        }
    }
    
    /// Gets the structural hash of this node
    ///
    /// This method computes a hash based on the structure of the node and its children.
    /// The hash is cached for future use, making subsequent calls efficient.
    pub fn hash(&self) -> u64 where K: Hash, V: Hash {
        // First check if we already have a cached hash
        if let Ok(cache) = self.cached_hash.lock() {
            if let Some(hash) = *cache {
                return hash;
            }
        }
        
        // If not, calculate the hash
        let hash = self.calculate_hash();
        
        // Cache the result
        if let Ok(mut cache) = self.cached_hash.lock() {
            *cache = Some(hash);
        }
        
        hash
    }
    
    /// Calculates the structural hash of this node without caching
    fn calculate_hash(&self) -> u64 where K: Hash, V: Hash {
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // Hash the key fragment
        self.key_fragment.hash(&mut hasher);
        
        // Hash the value if present
        if let Some(value) = &self.value {
            // We use a discriminant to differentiate nodes with/without values
            1u8.hash(&mut hasher);
            value.hash(&mut hasher);
        } else {
            0u8.hash(&mut hasher);
        }
        
        // Hash the children in a deterministic order
        let mut children: Vec<(&u8, &Arc<TrieNode<K, V>>)> = self.children.iter().collect();
        children.sort_by_key(|&(k, _)| k);
        
        for (key, child) in children {
            key.hash(&mut hasher);
            // Use the child's hash rather than hashing the entire structure
            child.hash().hash(&mut hasher);
        }
        
        hasher.finish()
    }
}

impl<K, V> Clone for TrieNode<K, V> {
    fn clone(&self) -> Self {
        TrieNode {
            key_fragment: self.key_fragment.clone(),
            value: self.value.clone(),
            children: self.children.clone(),
            cached_hash: Mutex::new(None), // Reset cache for the new node
            _key_type: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_node() {
        let key = Vec::new();
        let node: TrieNode<String, u32> = TrieNode::new(key.clone());
        
        assert_eq!(node.key_fragment, key);
        assert!(node.value.is_none());
        assert!(node.children.is_empty());
        assert!(node.is_leaf());
    }
    
    #[test]
    fn test_with_value() {
        let key = Vec::new();
        let node: TrieNode<String, u32> = TrieNode::with_value(key.clone(), 42);
        
        assert_eq!(node.key_fragment, key);
        assert_eq!(*node.value.unwrap(), 42);
        assert!(node.children.is_empty());
    }
    
    #[test]
    fn test_subtree_size() {
        let root_key = vec![0];
        
        let mut node: TrieNode<String, u32> = TrieNode::with_value(root_key, 42);
        assert_eq!(node.subtree_size(), 1);
        
        // Add a child
        let child_key = vec![1];
        let child = Arc::new(TrieNode::with_value(child_key, 43));
        node.children.insert(1, child);
        
        assert_eq!(node.subtree_size(), 2);
    }
    
    #[test]
    fn test_hash_consistency() {
        let key1 = vec![1];
        
        let node1: TrieNode<String, u32> = TrieNode::with_value(key1.clone(), 42);
        let node2: TrieNode<String, u32> = TrieNode::with_value(key1, 42);
        
        assert_eq!(node1.hash(), node2.hash());
        
        // Different key should have different hash
        let key2 = vec![2];
        let node3: TrieNode<String, u32> = TrieNode::with_value(key2, 42);
        
        assert_ne!(node1.hash(), node3.hash());
    }
    
    #[test]
    fn test_with_key_fragment() {
        let key1 = vec![1];
        
        let node1: TrieNode<String, u32> = TrieNode::with_value(key1, 42);
        
        let key2 = vec![2];
        
        let node2 = node1.with_key_fragment(key2);
        
        // Value should be the same
        assert_eq!(node2.value, node1.value);
        
        // Key should be different
        assert_ne!(node2.key_fragment, node1.key_fragment);
    }
}