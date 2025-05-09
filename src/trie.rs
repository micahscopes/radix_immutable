//! The main trie implementation.
//!
//! This module contains the `Trie` type, which provides the primary API for working
//! with the radix trie data structure.

use std::sync::Arc;
use std::borrow::Borrow;
use std::hash::Hash;
use std::collections::HashMap;

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
}



impl<K: AsRef<[u8]>, V> Trie<K, V> {
    /// Creates a view of the subtrie at the given key prefix.
    ///
    /// This method returns a lightweight view into a subtrie defined by a key prefix.
    /// The view supports efficient comparison and lookup operations on the subtrie.
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
    /// // Create a view of the subtrie at "hel" prefix
    /// let view = trie.view_subtrie("hel".to_string());
    ///
    /// // Check if keys exist in the view
    /// assert!(view.contains_key(&"hello".to_string()));
    ///
    /// // Get values from the view
    /// assert_eq!(view.get(&"hello".to_string()), Some(&1));
    /// ```
    pub fn view_subtrie(&self, prefix: K) -> PrefixView<K, V> 
    where 
        K: Clone,
        V: Clone,
    {
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
        // Convert the key to bytes
        let key_bytes = key_to_bytes(key);
        
        // Navigate from the root
        let mut current = &self.root;
        let mut remaining = &key_bytes[..];
        
        while !remaining.is_empty() {
            // Find the common prefix between remaining key and current node's fragment
            let common_len = prefix_match(remaining, 0, &current.key_fragment);
            
            // If current node's fragment isn't completely matched, key doesn't exist
            if common_len < current.key_fragment.len() {
                return None;
            }
            
            // Consume the matched part of the key
            remaining = &remaining[common_len..];
            
            // If there's nothing left, check if this node has a value
            if remaining.is_empty() {
                return current.value.as_ref().map(|v| v.as_ref());
            }
            
            // Otherwise, look for a child matching the next byte
            let next_byte = remaining[0];
            match current.children.get(&next_byte) {
                Some(child) => {
                    current = child;
                    // Move past the branching byte
                    remaining = &remaining[1..];
                }
                None => return None,
            }
        }
        
        // If we've consumed the entire key, return the value at this node (if any)
        current.value.as_ref().map(|v| v.as_ref())
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
    /// If the key already exists, the value is replaced.
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
    pub fn insert(&self, key: K, value: V) -> Self {
        // Convert the key to bytes
        let key_bytes = key_to_bytes(&key);
        
        // Call the recursive helper to perform the insertion with path copying
        let (new_root, value_replaced) = self.insert_recursive(&self.root, &key_bytes, Arc::new(value));
        
        // Calculate the new size
        let new_size = if value_replaced { self.size } else { self.size + 1 };
        
        // Create a new trie with the updated root and size
        Trie {
            root: new_root,
            size: new_size,
        }
    }
    
    // Recursive helper for insert that handles path copying to allow for structural sharing
    fn insert_recursive(&self, node: &Arc<TrieNode<K, V>>, key: &[u8], value: Arc<V>) -> (Arc<TrieNode<K, V>>, bool) {
        if key.is_empty() {
            // We've reached the end of the key, create a new node with the value
            let mut new_node = TrieNode::new(node.key_fragment.clone());
            new_node.children = node.children.clone();
            new_node.value = Some(value);
            
            // Return the new node and whether we replaced a value
            return (Arc::new(new_node), node.value.is_some());
        }
        
        // Find how much of the key matches the current node's key fragment
        let common_len = prefix_match(key, 0, &node.key_fragment);
        
        if common_len < node.key_fragment.len() {
            // The key and node's fragment share a prefix, but they diverge
            // We need to split the current node
            
            // Create a child with the unmatched part of the node's fragment
            let remaining_fragment = node.key_fragment[common_len+1..].to_vec();
            let mut child = TrieNode::new(remaining_fragment);
            child.children = node.children.clone();
            child.value = node.value.clone();
            
            // Create a new node map with the child
            let mut children = HashMap::new();
            let branch_byte = node.key_fragment[common_len];
            children.insert(branch_byte, Arc::new(child));
            
            // If there's more of the key, create a new leaf node
            if common_len < key.len() {
                let new_key_fragment = key[common_len+1..].to_vec();
                let mut leaf = TrieNode::new(new_key_fragment);
                leaf.value = Some(value.clone());
                let new_branch_byte = key[common_len];
                children.insert(new_branch_byte, Arc::new(leaf));
            }
            
            // Create the new split node
            let mut split_node = TrieNode::new(node.key_fragment[..common_len].to_vec());
            
            // If key is fully consumed at the split point, store the value
            if common_len == key.len() {
                split_node.value = Some(value.clone());
            }
            
            split_node.children = children;
            
            // Return the new node (never replacing a value when splitting)
            return (Arc::new(split_node), false);
        }
        
        // The key fragment was fully matched, now we need to handle the rest of the key
        let remaining = &key[common_len..];
        
        if remaining.is_empty() {
            // We've matched the entire key, update the value at this node
            let mut new_node = TrieNode::new(node.key_fragment.clone());
            new_node.children = node.children.clone();
            new_node.value = Some(value);
            
            return (Arc::new(new_node), node.value.is_some());
        }
        
        // We have more key to process, go down the appropriate child
        let next_byte = remaining[0];
        let mut new_children = node.children.clone();
        
        match node.children.get(&next_byte) {
            Some(child) => {
                // Recursively insert into the child
                let (new_child, value_replaced) = self.insert_recursive(child, &remaining[1..], value.clone());
                
                // Update the child in our children map
                new_children.insert(next_byte, new_child);
                
                // Create a new node with the updated children
                let mut new_node = TrieNode::new(node.key_fragment.clone());
                new_node.value = node.value.clone();
                new_node.children = new_children;
                
                return (Arc::new(new_node), value_replaced);
            }
            None => {
                // No matching child, add a new leaf for the remaining key
                let leaf_fragment = remaining[1..].to_vec();
                let new_leaf = TrieNode::with_value(leaf_fragment, (*value).clone());
                
                // Add the new leaf to the children map
                new_children.insert(next_byte, Arc::new(new_leaf));
                
                // Create a new node with the updated children
                let mut new_node = TrieNode::new(node.key_fragment.clone());
                new_node.value = node.value.clone();
                new_node.children = new_children;
                
                return (Arc::new(new_node), false);
            }
        }
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
    pub fn remove<Q>(&self, key: &Q) -> (Self, Option<V>)
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized + AsRef<[u8]>,
    {
        // Convert the key to bytes
        let key_bytes = key_to_bytes(key);
        
        // Use the recursive helper to perform the removal with path copying
        let (new_root, removed_value) = self.remove_recursive(&self.root, &key_bytes);
        
        // Calculate the new size
        let new_size = if removed_value.is_some() { self.size - 1 } else { self.size };
        
        // Create a new trie with the updated root and size
        (Trie {
            root: new_root,
            size: new_size,
        }, removed_value)
    }
    
    // Recursive helper for remove that handles path copying
    fn remove_recursive(&self, node: &Arc<TrieNode<K, V>>, key: &[u8]) -> (Arc<TrieNode<K, V>>, Option<V>) {
        if key.is_empty() {
            // We've reached the end of the key, remove the value if it exists
            if node.value.is_none() {
                // No value to remove
                return (Arc::clone(node), None);
            }
            
            // Create a new node without the value
            let mut new_node = TrieNode::new(node.key_fragment.clone());
            new_node.children = node.children.clone();
            
            // Return the new node and the removed value
            return (Arc::new(new_node), node.value.as_ref().map(|v| (**v).clone()));
        }
        
        // Find how much of the key matches the current node's key fragment
        let common_len = prefix_match(key, 0, &node.key_fragment);
        
        if common_len < node.key_fragment.len() {
            // The key doesn't match the node's fragment completely
            // This means the key doesn't exist in the trie
            return (Arc::clone(node), None);
        }
        
        // The key fragment was fully matched, now handle the remaining key
        let remaining = &key[common_len..];
        
        if remaining.is_empty() {
            // We've reached the target node, remove its value
            if node.value.is_none() {
                // No value to remove
                return (Arc::clone(node), None);
            }
            
            // If this node has no children, it can be removed entirely
            if node.children.is_empty() {
                return (Arc::new(TrieNode::new(Vec::new())), node.value.as_ref().map(|v| (**v).clone()));
            }
            
            // Otherwise, create a new node without the value but with the same children
            let mut new_node = TrieNode::new(node.key_fragment.clone());
            new_node.children = node.children.clone();
            
            return (Arc::new(new_node), node.value.as_ref().map(|v| (**v).clone()));
        }
        
        // We need to go deeper into the trie
        let next_byte = remaining[0];
        
        // Check if there's a child matching the next byte
        if let Some(child) = node.children.get(&next_byte) {
            // Recursively remove from the child
            let (new_child, removed_value) = self.remove_recursive(child, &remaining[1..]);
            
            // If nothing was removed, return the original node
            if removed_value.is_none() {
                return (Arc::clone(node), None);
            }
            
            // Create a new children map for path copying
            let mut new_children = node.children.clone();
            
            // Check if the child should be completely removed or replaced
            if new_child.key_fragment.is_empty() && new_child.children.is_empty() && new_child.value.is_none() {
                // Child is empty, remove it
                new_children.remove(&next_byte);
            } else {
                // Replace the child
                new_children.insert(next_byte, new_child);
            }
            
            // If this node has no value and only one child after removal, we can merge them
            if node.value.is_none() && new_children.len() == 1 {
                let (only_byte, only_child) = new_children.iter().next().unwrap();
                
                // If the child has a value or multiple children, don't merge
                if only_child.value.is_some() || only_child.children.len() > 1 {
                    // Create a new node with the updated children
                    let mut new_node = TrieNode::new(node.key_fragment.clone());
                    new_node.children = new_children;
                    return (Arc::new(new_node), removed_value);
                }
                
                // Merge this node with its only child
                let mut merged_fragment = node.key_fragment.clone();
                merged_fragment.push(*only_byte);
                merged_fragment.extend_from_slice(&only_child.key_fragment);
                
                let mut merged_node = TrieNode::new(merged_fragment);
                merged_node.children = only_child.children.clone();
                merged_node.value = only_child.value.clone();
                
                return (Arc::new(merged_node), removed_value);
            }
            
            // Create a new node with the updated children
            let mut new_node = TrieNode::new(node.key_fragment.clone());
            new_node.value = node.value.clone();
            new_node.children = new_children;
            
            return (Arc::new(new_node), removed_value);
        }
        
        // No matching child, key doesn't exist in the trie
        return (Arc::clone(node), None);
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
    use std::sync::Arc;
    
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
    
    #[test]
    fn test_insert_and_get() {
        let trie: Trie<String, u32> = Trie::new();
        let trie = trie.insert("hello".to_string(), 42);
        
        assert_eq!(trie.len(), 1);
        assert_eq!(trie.get(&"hello".to_string()), Some(&42));
        assert_eq!(trie.get(&"world".to_string()), None);
    }
    
    #[test]
    fn test_insert_replace() {
        let trie: Trie<String, u32> = Trie::new();
        let trie1 = trie.insert("hello".to_string(), 42);
        let trie2 = trie1.insert("hello".to_string(), 100);
        
        assert_eq!(trie1.len(), 1);
        assert_eq!(trie2.len(), 1);
        assert_eq!(trie1.get(&"hello".to_string()), Some(&42));
        assert_eq!(trie2.get(&"hello".to_string()), Some(&100));
    }
    
    #[test]
    fn test_insert_multiple() {
        let trie: Trie<String, u32> = Trie::new();
        let trie1 = trie.insert("hello".to_string(), 42);
        let trie2 = trie1.insert("world".to_string(), 100);
        let trie3 = trie2.insert("hello world".to_string(), 200);
        
        assert_eq!(trie1.len(), 1);
        assert_eq!(trie2.len(), 2);
        assert_eq!(trie3.len(), 3);
        
        assert_eq!(trie3.get(&"hello".to_string()), Some(&42));
        assert_eq!(trie3.get(&"world".to_string()), Some(&100));
        assert_eq!(trie3.get(&"hello world".to_string()), Some(&200));
    }
    
    #[test]
    fn test_structural_sharing() {
        let trie: Trie<String, u32> = Trie::new();
        let trie1 = trie.insert("hello".to_string(), 42);
        let trie2 = trie1.insert("help".to_string(), 100);
        
        // The root nodes should be different
        assert!(!Arc::ptr_eq(&trie1.root, &trie2.root));
        
        // But they should share structure for the "hel" prefix
        // We need to navigate to the child nodes to check this
        let first_child1 = &trie1.root.children.get(&b'h').unwrap();
        let first_child2 = &trie2.root.children.get(&b'h').unwrap();
        
        // The 'h' nodes should be different since we modified this path
        assert!(!Arc::ptr_eq(first_child1, first_child2));
        
        // Now insert a completely different prefix
        let trie3 = trie2.insert("world".to_string(), 200);
        
        // The "hel" subtree should be shared between trie2 and trie3
        let prefix_node2 = &trie2.root.children.get(&b'h').unwrap();
        let prefix_node3 = &trie3.root.children.get(&b'h').unwrap();
        
        // Verify same pointer (same memory location) - structural sharing!
        assert!(Arc::ptr_eq(prefix_node2, prefix_node3));
    }
    
    #[test]
    fn test_node_splitting() {
        let trie: Trie<String, u32> = Trie::new();
        
        // Insert a key
        let trie1 = trie.insert("alphabet".to_string(), 1);
        
        // Insert another with common prefix - should cause splitting
        let trie2 = trie1.insert("alpha".to_string(), 2);
        
        assert_eq!(trie2.get(&"alphabet".to_string()), Some(&1));
        assert_eq!(trie2.get(&"alpha".to_string()), Some(&2));
    }
    
    #[test]
    fn test_larger_key_first() {
        let trie: Trie<String, u32> = Trie::new();
        
        // First insert the longer key
        let trie1 = trie.insert("alphabet".to_string(), 1);
        
        // Then insert the shorter one
        let trie2 = trie1.insert("alpha".to_string(), 2);
        
        assert_eq!(trie2.get(&"alphabet".to_string()), Some(&1));
        assert_eq!(trie2.get(&"alpha".to_string()), Some(&2));
    }
    
    #[test]
    fn test_shorter_key_first() {
        let trie: Trie<String, u32> = Trie::new();
        
        // First insert the shorter key
        let trie1 = trie.insert("alpha".to_string(), 1);
        
        // Then insert the longer one
        let trie2 = trie1.insert("alphabet".to_string(), 2);
        
        assert_eq!(trie2.get(&"alpha".to_string()), Some(&1));
        assert_eq!(trie2.get(&"alphabet".to_string()), Some(&2));
    }
    
    #[test]
    fn test_remove_existing() {
        let trie: Trie<String, u32> = Trie::new()
            .insert("hello".to_string(), 42)
            .insert("world".to_string(), 100);
        
        assert_eq!(trie.len(), 2);
        
        // Remove an existing key
        let (trie2, removed) = trie.remove(&"hello".to_string());
        
        assert_eq!(removed, Some(42));
        assert_eq!(trie2.len(), 1);
        assert_eq!(trie2.get(&"hello".to_string()), None);
        assert_eq!(trie2.get(&"world".to_string()), Some(&100));
        
        // The original trie should be unchanged
        assert_eq!(trie.len(), 2);
        assert_eq!(trie.get(&"hello".to_string()), Some(&42));
    }
    
    #[test]
    fn test_remove_nonexistent() {
        let trie: Trie<String, u32> = Trie::new()
            .insert("hello".to_string(), 42);
        
        // Remove a non-existent key
        let (trie2, removed) = trie.remove(&"world".to_string());
        
        assert_eq!(removed, None);
        assert_eq!(trie2.len(), 1);
        assert_eq!(trie2.get(&"hello".to_string()), Some(&42));
        
        // Removing from an empty trie
        let empty: Trie<String, u32> = Trie::new();
        let (empty2, removed) = empty.remove(&"anything".to_string());
        
        assert_eq!(removed, None);
        assert_eq!(empty2.len(), 0);
    }
    
    #[test]
    fn test_remove_with_compression() {
        // Create a trie with keys that will cause path compression when removed
        let trie: Trie<String, u32> = Trie::new()
            .insert("abc".to_string(), 1)
            .insert("abcde".to_string(), 2);
            
        // Remove the middle key, which should cause compression
        let (trie2, removed) = trie.remove(&"abc".to_string());
        
        assert_eq!(removed, Some(1));
        assert_eq!(trie2.len(), 1);
        assert_eq!(trie2.get(&"abc".to_string()), None);
        assert_eq!(trie2.get(&"abcde".to_string()), Some(&2));
    }
    
    #[test]
    fn test_remove_structural_sharing() {
        let trie: Trie<String, u32> = Trie::new()
            .insert("hello".to_string(), 1)
            .insert("help".to_string(), 2)
            .insert("world".to_string(), 3);
            
        // Remove a key from one branch
        let (trie2, _) = trie.remove(&"world".to_string());
        
        // The "hel" branch should be shared between the two tries
        let h_node1 = trie.root.children.get(&b'h').unwrap();
        let h_node2 = trie2.root.children.get(&b'h').unwrap();
        
        // The 'h' nodes should be the same (structural sharing)
        assert!(Arc::ptr_eq(h_node1, h_node2));
        
        // But removing from a branch should create new nodes along that path
        let (trie3, _) = trie.remove(&"hello".to_string());
        
        let h_node3 = trie3.root.children.get(&b'h').unwrap();
        
        // The 'h' nodes should be different now
        assert!(!Arc::ptr_eq(h_node1, h_node3));
    }
}