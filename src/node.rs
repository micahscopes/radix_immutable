//! Internal node implementation for the radix trie.
//!
//! This module contains the internal `TrieNode` structure that forms the backbone
//! of the radix trie implementation. `TrieNode` instances are always wrapped in an
//! `Arc` to enable structural sharing.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// Stores the original key and its associated value.
#[derive(Debug)]
pub struct KeyValuePair<K, V> {
    pub key: Arc<K>,   // The original, full key
    pub value: Arc<V>, // The value, wrapped in Arc for sharing
}

// Manual implementation of Clone that only requires Arc<K> and Arc<V> to be Clone,
// which they are without requiring K: Clone or V: Clone
impl<K, V> Clone for KeyValuePair<K, V> {
    fn clone(&self) -> Self {
        KeyValuePair {
            key: self.key.clone(),
            value: self.value.clone(),
        }
    }
}

/// Internal node type for the radix trie.
///
/// This type is not exposed directly in the public API but is used internally
/// by the `Trie` type. Each node contains a key fragment, an optional value,
/// and a map of children.
#[derive(Debug)]
pub struct TrieNode<K, V> {
    /// The key fragment stored at this node (as a sequence of bytes)
    pub key_fragment: Vec<u8>,
    
    /// The full key and value stored at this node, if this node represents
    /// the end of a complete key.
    pub data: Option<KeyValuePair<K, V>>,
    
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
            data: None,
            children: HashMap::new(),
            cached_hash: Mutex::new(None),
            _key_type: std::marker::PhantomData,
        }
    }
    
    /// Creates a new node with the given key fragment, original key, and value.
    /// This is used when a node represents the end of a key.
    pub fn with_key_value(key_fragment: Vec<u8>, original_key: K, value: V) -> Self {
        TrieNode {
            key_fragment,
            data: Some(KeyValuePair {
                key: Arc::new(original_key),
                value: Arc::new(value),
            }),
            children: HashMap::new(),
            cached_hash: Mutex::new(None),
            _key_type: std::marker::PhantomData,
        }
    }
    
    /// Creates a new empty node with just a value (and its original key).
    /// This is a convenience for when the key_fragment is empty.
    pub fn new_value_node(original_key: K, value: V) -> Self {
        TrieNode {
            key_fragment: Vec::new(),
            data: Some(KeyValuePair {
                key: Arc::new(original_key),
                value: Arc::new(value),
            }),
            children: HashMap::new(),
            cached_hash: Mutex::new(None),
            _key_type: std::marker::PhantomData,
        }
    }
    
    /// Returns the number of values stored in this subtree
    pub fn subtree_size(&self) -> usize {
        let mut count = if self.data.is_some() { 1 } else { 0 };
        
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
            data: self.data.clone(),
            children: self.children.clone(),
            cached_hash: Mutex::new(None), // Reset cache for the new node
            _key_type: std::marker::PhantomData,
        }
    }
    
    /// Creates a clone of this node, but with new data (key-value pair)
    pub fn with_data_option(&self, data: Option<KeyValuePair<K, V>>) -> Self {
        TrieNode {
            key_fragment: self.key_fragment.clone(),
            data,
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
        
        // Hash the KeyValuePair if present
        if let Some(kvp) = &self.data {
            // We use a discriminant to differentiate nodes with/without values
            1u8.hash(&mut hasher);
            kvp.key.hash(&mut hasher); // Hash the original K
            kvp.value.hash(&mut hasher); // Hash Arc<V> (hashes V by deref)
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
            data: self.data.clone(),
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
        let key_frag = Vec::new();
        let node: TrieNode<String, u32> = TrieNode::new(key_frag.clone());
        
        assert_eq!(node.key_fragment, key_frag);
        assert!(node.data.is_none());
        assert!(node.children.is_empty());
        assert!(node.is_leaf());
    }
    
    #[test]
    fn test_with_key_value() {
        let key_frag = vec![1, 2];
        let original_key_str = "test_key".to_string();
        let val = 42u32;
        let node: TrieNode<String, u32> = TrieNode::with_key_value(key_frag.clone(), original_key_str.clone(), val);
        
        assert_eq!(node.key_fragment, key_frag);
        assert!(node.data.is_some());
        let kvp = node.data.as_ref().unwrap();
        assert_eq!(&*kvp.key, &original_key_str);
        assert_eq!(*kvp.value, val);
        assert!(node.children.is_empty());
    }
    
    #[test]
    fn test_subtree_size_with_data() {
        let key_frag_root = vec![0];
        let original_key_root = "root".to_string();
        
        let mut node: TrieNode<String, u32> = TrieNode::with_key_value(key_frag_root, original_key_root.clone(), 42);
        assert_eq!(node.subtree_size(), 1);
        
        // Add a child
        let child_frag = vec![1];
        let original_key_child = "child".to_string();
        let child_node = Arc::new(TrieNode::with_key_value(child_frag, original_key_child, 43));
        node.children.insert(1, child_node);
        
        assert_eq!(node.subtree_size(), 2); // One for root, one for child
        
        let mut node_no_data: TrieNode<String, u32> = TrieNode::new(vec![5]);
        let child_node2 = Arc::new(TrieNode::with_key_value(vec![6], "child2".to_string(), 44));
        node_no_data.children.insert(1, child_node2);
        assert_eq!(node_no_data.subtree_size(), 1); // Only the child has data
    }
    
    #[test]
    fn test_hash_consistency_with_data() {
        let key_frag = vec![1];
        let k_orig = "key".to_string();
        let v = 42;
        
        let node1: TrieNode<String, u32> = TrieNode::with_key_value(key_frag.clone(), k_orig.clone(), v);
        let node2: TrieNode<String, u32> = TrieNode::with_key_value(key_frag.clone(), k_orig.clone(), v);
        
        assert_eq!(node1.hash(), node2.hash());
        
        // Different original key should have different hash
        let k_orig_diff = "key_diff".to_string();
        let node3: TrieNode<String, u32> = TrieNode::with_key_value(key_frag.clone(), k_orig_diff, v);
        
        assert_ne!(node1.hash(), node3.hash());
        
        // Different value should have different hash
        let v_diff = 43;
        let node4: TrieNode<String, u32> = TrieNode::with_key_value(key_frag.clone(), k_orig.clone(), v_diff);
        
        assert_ne!(node1.hash(), node4.hash());
        
        // Different key_fragment should have different hash
        let key_frag_diff = vec![2];
        let node5: TrieNode<String, u32> = TrieNode::with_key_value(key_frag_diff, k_orig.clone(), v);
        
        assert_ne!(node1.hash(), node5.hash());
    }
    
    #[test]
    fn test_with_key_fragment_keeps_data() {
        let key_frag1 = vec![1];
        let k_orig = "key".to_string();
        
        let node1: TrieNode<String, u32> = TrieNode::with_key_value(key_frag1, k_orig.clone(), 42);
        
        let key_frag2 = vec![2];
        let node2 = node1.with_key_fragment(key_frag2.clone());
        
        assert_eq!(node2.key_fragment, key_frag2);
        assert!(node2.data.is_some());
        assert_eq!(&*node2.data.as_ref().unwrap().key, &k_orig);
        assert_eq!(*node2.data.as_ref().unwrap().value, 42);
    }
    
    #[test]
    fn test_clone_node_with_data() {
        let k_orig = "key_clone".to_string();
        let node1: TrieNode<String, u32> = TrieNode::with_key_value(vec![1], k_orig.clone(), 100);
        let node_clone = node1.clone();
        
        assert_eq!(node_clone.key_fragment, node1.key_fragment);
        assert!(node_clone.data.is_some());
        assert_eq!(&*node_clone.data.as_ref().unwrap().key, &k_orig);
        assert_eq!(*node_clone.data.as_ref().unwrap().value, 100);
        
        // Let's test calculated hash
        let h1 = node1.hash();
        let hc = node_clone.hash();
        assert_eq!(h1, hc);
    }
}