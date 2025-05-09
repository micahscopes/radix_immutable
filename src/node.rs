//! Internal node implementation for the radix trie.
//!
//! This module contains the internal `TrieNode` structure that forms the backbone
//! of the radix trie implementation. `TrieNode` instances are always wrapped in an
//! `Arc` to enable structural sharing.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use once_cell::sync::OnceCell;

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
        /// The hash is calculated lazily and cached for future use using thread-safe
        /// OnceCell, which avoids locking overhead after initialization while
        /// maintaining thread safety.
        pub cached_hash: OnceCell<u64>,
    
        /// Cached subtree size (number of values in this subtree)
        ///
        /// This is computed lazily and cached for future use using thread-safe
        /// OnceCell, which provides efficient read access without locking overhead.
        pub cached_subtree_size: OnceCell<usize>,
    
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
            cached_hash: OnceCell::new(),
            cached_subtree_size: OnceCell::new(),
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
            cached_hash: OnceCell::new(),
            cached_subtree_size: OnceCell::new(),
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
            cached_hash: OnceCell::new(),
            cached_subtree_size: OnceCell::new(),
            _key_type: std::marker::PhantomData,
        }
    }
    
    /// Returns the number of values stored in this subtree
    /// 
    /// This method counts all values in this node and its descendants.
    /// The count is computed once and cached via OnceCell for future use,
    /// providing efficient access with no locking overhead for subsequent calls.
    pub fn subtree_size(&self) -> usize {
        // Return cached value if available
        if let Some(size) = self.cached_subtree_size.get() {
            return *size;
        }
        
        // Calculate the size if not cached
        let mut count = if self.data.is_some() { 1 } else { 0 };
        
        for child in self.children.values() {
            count += child.subtree_size();
        }
        
        // Cache the result (ignore failure, it will be recalculated if needed)
        let _ = self.cached_subtree_size.set(count);
        
        count
    }
    
    #[cfg(test)]
    /// Clears the cached size, forcing it to be recalculated next time it's needed
    fn clear_cached_subtree_size(&mut self) {
        self.cached_subtree_size = OnceCell::new();
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
            cached_hash: OnceCell::new(), // New OnceCell for the modified node
            cached_subtree_size: OnceCell::new(), // New OnceCell for the modified node
            _key_type: std::marker::PhantomData,
        }
    }
    
    /// Creates a clone of this node, but with new data (key-value pair)
    pub fn with_data_option(&self, data: Option<KeyValuePair<K, V>>) -> Self {
        TrieNode {
            key_fragment: self.key_fragment.clone(),
            data,
            children: self.children.clone(),
            cached_hash: OnceCell::new(), // New OnceCell for the modified node
            cached_subtree_size: OnceCell::new(), // New OnceCell for the modified node
            _key_type: std::marker::PhantomData,
        }
    }
    
    /// Gets the structural hash of this node
    ///
    /// This method computes a hash based on the structure of the node and its children.
    /// The hash is computed once and cached via OnceCell for future use, making 
    /// subsequent calls extremely efficient with no locking overhead.
    /// 
    /// Since nodes are immutable after creation, this provides thread-safe caching
    /// without the performance penalties of mutex-based solutions.
    pub fn hash(&self) -> u64 where K: Hash, V: Hash {
        // Return cached value if available
        if let Some(hash) = self.cached_hash.get() {
            return *hash;
        }
        
        // Calculate the hash if not cached
        let hash = self.calculate_hash();
        
        // Cache the result (ignore failure, it will be recalculated if needed)
        let _ = self.cached_hash.set(hash);
        
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
    /// Creates a clone of this node with fresh cache cells
    /// 
    /// The clone operation intentionally creates new, empty OnceCell instances
    /// rather than copying the cached values. This ensures that any modifications
    /// to the cloned structure will have its own cache state.
    fn clone(&self) -> Self {
        TrieNode {
            key_fragment: self.key_fragment.clone(),
            data: self.data.clone(),
            children: self.children.clone(),
            cached_hash: OnceCell::new(), // New OnceCell for the clone
            cached_subtree_size: OnceCell::new(), // New OnceCell for the clone
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
        
        // Clear the cache since we modified the children
        node.clear_cached_subtree_size();
        
        assert_eq!(node.subtree_size(), 2); // One for root, one for child
        
        let mut node_no_data: TrieNode<String, u32> = TrieNode::new(vec![5]);
        let child_node2 = Arc::new(TrieNode::with_key_value(vec![6], "child2".to_string(), 44));
        node_no_data.children.insert(1, child_node2);
        assert_eq!(node_no_data.subtree_size(), 1); // Only the child has data
    }
    
    #[test]
    fn test_subtree_size_caching() {
        let key_frag_root = vec![0];
        let original_key_root = "root".to_string();
        
        // Create a node with a value
        let mut node: TrieNode<String, u32> = TrieNode::with_key_value(key_frag_root, original_key_root.clone(), 42);
        
        // First call computes and caches
        assert_eq!(node.subtree_size(), 1);
        
        // Verify cache is populated after first call
        assert_eq!(node.cached_subtree_size.get(), Some(&1));
        
        // Add a child
        let child_frag = vec![1];
        let original_key_child = "child".to_string();
        let child_node = Arc::new(TrieNode::with_key_value(child_frag, original_key_child, 43));
        node.children.insert(1, child_node);
        
        // Clear the cache since we modified the children
        node.clear_cached_subtree_size();
        
        // Cache should be empty now
        assert_eq!(node.cached_subtree_size.get(), None);
        
        // Calling subtree_size() calculates and caches
        assert_eq!(node.subtree_size(), 2);
        
        // And now cache has updated
        assert_eq!(node.cached_subtree_size.get(), Some(&2));
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
    fn test_hash_caching() {
        let key_frag = vec![1, 2, 3];
        let k_orig = "test_key".to_string();
        let v = 42;
        
        let node: TrieNode<String, u32> = TrieNode::with_key_value(key_frag, k_orig, v);
        
        // Cache should be empty initially
        assert!(node.cached_hash.get().is_none());
        
        // First call computes and caches
        let hash_value = node.hash();
        
        // Verify cache has been populated
        assert_eq!(node.cached_hash.get(), Some(&hash_value));
        
        // Verify cached value is returned on second call
        assert_eq!(node.hash(), hash_value);
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
    fn test_with_key_fragment_fresh_cache() {
        let key_frag1 = vec![1];
        let k_orig = "key".to_string();
        
        let node1: TrieNode<String, u32> = TrieNode::with_key_value(key_frag1, k_orig.clone(), 42);
        
        // Compute hash to populate cache
        let original_hash = node1.hash();
        assert!(node1.cached_hash.get().is_some());
        
        // Create new node with different key fragment
        let key_frag2 = vec![2];
        let node2 = node1.with_key_fragment(key_frag2.clone());
        
        // New node should have empty caches
        assert!(node2.cached_hash.get().is_none());
        assert!(node2.cached_subtree_size.get().is_none());
        
        // Hash should be different because key_fragment is different
        assert_ne!(node2.hash(), original_hash);
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
    
    #[test]
    fn test_clone_resets_caches() {
        let k_orig = "cache_test".to_string();
        let mut node1: TrieNode<String, u32> = TrieNode::with_key_value(vec![1], k_orig, 42);
        
        // Add a child to make subtree size > 1
        let child = Arc::new(TrieNode::<String, u32>::with_key_value(vec![2], "child".to_string(), 43));
        node1.children.insert(2, child);
        
        // Calculate hash and subtree_size to populate caches
        let original_hash = node1.hash();
        let original_size = node1.subtree_size();
        assert_eq!(original_size, 2);
        
        // Verify caches are populated
        assert_eq!(node1.cached_hash.get(), Some(&original_hash));
        assert_eq!(node1.cached_subtree_size.get(), Some(&original_size));
        
        // Clone the node
        let node_clone = node1.clone();
        
        // Cloned node should have empty caches
        assert!(node_clone.cached_hash.get().is_none());
        assert!(node_clone.cached_subtree_size.get().is_none());
        
        // But calculated values should match
        assert_eq!(node_clone.hash(), original_hash);
        assert_eq!(node_clone.subtree_size(), original_size);
        
        // And now caches should be populated
        assert_eq!(node_clone.cached_hash.get(), Some(&original_hash));
        assert_eq!(node_clone.cached_subtree_size.get(), Some(&original_size));
    }
    
    #[test]
    fn test_complex_node_structure() {
        // Create a more complex structure to test both caching mechanisms
        //    root
        //   /   \
        // childA childB
        //        /   \
        //   grandchild1 grandchild2
        
        let root = TrieNode::<String, u32>::with_key_value(
            vec![0], "root".to_string(), 1
        );
        
        let child_a = Arc::new(TrieNode::<String, u32>::with_key_value(
            vec![1], "childA".to_string(), 2
        ));
        
        let grandchild1 = Arc::new(TrieNode::<String, u32>::with_key_value(
            vec![3], "grandchild1".to_string(), 4
        ));
        
        let grandchild2 = Arc::new(TrieNode::<String, u32>::with_key_value(
            vec![4], "grandchild2".to_string(), 5
        ));
        
        let mut child_b_children = HashMap::new();
        child_b_children.insert(3, grandchild1);
        child_b_children.insert(4, grandchild2);
        
        let child_b = Arc::new(TrieNode {
            key_fragment: vec![2],
            data: Some(KeyValuePair {
                key: Arc::new("childB".to_string()),
                value: Arc::new(3u32),
            }),
            children: child_b_children,
            cached_hash: OnceCell::new(),
            cached_subtree_size: OnceCell::new(),
            _key_type: std::marker::PhantomData,
        });
        
        let mut root_with_children = root.clone();
        root_with_children.children.insert(1, child_a);
        root_with_children.children.insert(2, child_b);
        
        // Test subtree size
        assert_eq!(root_with_children.subtree_size(), 5);
        
        // Test that hash works on complex structures
        let hash = root_with_children.hash();
        assert!(hash != 0);
        
        // Test that cache is populated
        assert_eq!(root_with_children.cached_hash.get(), Some(&hash));
        assert_eq!(root_with_children.cached_subtree_size.get(), Some(&5usize));
        
        // Change the structure by creating a copy with different data
        let new_root = root_with_children.with_data_option(Some(KeyValuePair {
            key: Arc::new("new_root_key".to_string()),
            value: Arc::new(100u32),
        }));
        
        // Caches should be empty for the new structure
        assert!(new_root.cached_hash.get().is_none());
        assert!(new_root.cached_subtree_size.get().is_none());
        
        // But subtree size should be the same
        assert_eq!(new_root.subtree_size(), 5);
        
        // And hash should be different
        assert_ne!(new_root.hash(), hash);
    }
}