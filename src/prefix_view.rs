//! Prefix view into a radix trie.
//!
//! This module provides the `PrefixView` type, which allows for efficient
//! access and comparison of subtries based on key prefixes.

use std::collections::VecDeque;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::key_converter::KeyToBytes;
use crate::node::TrieNode;
use crate::util::prefix_match;
use crate::Trie;

/// A lightweight view into a subtrie defined by a key prefix.
///
/// PrefixView allows for efficient comparison of subtries by comparing their
/// structural hashes rather than performing deep comparisons of all entries.
///
/// # Examples
///
/// ```
/// use radix_immutable::{Trie, StrKeyConverter};
///
/// let trie1 = Trie::<String, i32, StrKeyConverter<String>>::new_str_key()
///     .insert("hello".to_string(), 1)
///     .insert("help".to_string(), 2);
///
/// let trie2 = Trie::<String, i32, StrKeyConverter<String>>::new_str_key()
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
pub struct PrefixView<K: Clone + Hash + Eq, V, KC: KeyToBytes<K>> {
    /// The source trie for this view
    trie: Trie<K, V, KC>,

    /// The key prefix defining this view
    prefix: K,

    /// The subtrie node at the prefix, if it exists
    subtrie_node: Option<Arc<TrieNode<K, V>>>,

    /// Phantom data for the key converter
    _phantom_kc: PhantomData<KC>,
}

/// An iterator over the entries of a PrefixView.
///
/// This iterator performs a depth-first traversal of the trie and yields
/// cloned keys and values that match the prefix.
pub struct PrefixViewIter<K: Clone + Hash + Eq, V: Clone, KC: KeyToBytes<K>> {
    /// Stack of nodes to visit, along with their parents (for backtracking)
    stack: VecDeque<(Arc<TrieNode<K, V>>, Vec<u8>)>,

    /// The prefix view we're iterating over - owned to avoid lifetime issues
    view: PrefixView<K, V, KC>,
}

/// Arc-based iterator over the key-value pairs in a prefix view.
///
/// This iterator performs a depth-first traversal of the trie and yields
/// Arc references to the keys and values that match the prefix.
pub struct PrefixViewArcIter<K: Clone + Hash + Eq, V, KC: KeyToBytes<K> + Clone> {
    /// Stack of nodes to visit, along with their parents (for backtracking)
    stack: VecDeque<(Arc<TrieNode<K, V>>, Vec<u8>)>,

    /// The prefix view we're iterating over - owned to avoid lifetime issues
    view: PrefixView<K, V, KC>,
}

impl<K: Clone + Hash + Eq, V, KC: KeyToBytes<K>> PrefixView<K, V, KC>
where
    V: Clone,  // Trie::clone() requires V: Clone
    KC: Clone, // Trie::clone() requires KC: Clone (KeyToBytes already requires Clone)
{

    /// Creates a new prefix view for the given trie and prefix.
    pub fn new(trie: Trie<K, V, KC>, prefix: K) -> Self {
        // Find the node corresponding to the prefix
        // Ensure K and KC bounds are met by the caller of new() due to Trie's bounds
        let subtrie_node = Self::find_subtrie_node(&trie, &prefix);

        PrefixView {
            trie,
            prefix,
            subtrie_node,
            _phantom_kc: PhantomData,
        }
    }

    /// Returns the key prefix for this view.
    pub fn prefix(&self) -> &K {
        &self.prefix
    }

    /// Returns the underlying trie.
    pub fn trie(&self) -> &Trie<K, V, KC> {
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

    /// Returns an iterator over the key-value pairs in the prefix view.
    ///
    /// The iterator yields pairs of `(K, V)` (cloned values) in depth-first order.
    pub fn iter(&self) -> PrefixViewIter<K, V, KC> 
    where V: Clone {
        let mut iter = PrefixViewIter::<K, V, KC> {
            stack: VecDeque::new(),
            view: self.clone(),
        };

        // If the view has a subtrie node, add it to the stack
        if let Some(node) = &self.subtrie_node {
            iter.stack.push_back((Arc::clone(node), Vec::new()));
        }

        iter
    }
    
    /// Returns an iterator over Arc references to the key-value pairs in the prefix view.
    ///
    /// The iterator yields pairs of `(Arc<K>, Arc<V>)` in depth-first order.
    /// This avoids cloning the actual values but changes the return type.
    pub fn iter_arc(&self) -> PrefixViewArcIter<K, V, KC> {
        let mut iter = PrefixViewArcIter::<K, V, KC> {
            stack: VecDeque::new(),
            view: self.clone(),
        };

        // If the view has a subtrie node, add it to the stack
        if let Some(node) = &self.subtrie_node {
            iter.stack.push_back((Arc::clone(node), Vec::new()));
        }

        iter
    }

    // Helper method to find the subtrie node at the given prefix
    fn find_subtrie_node(trie: &Trie<K, V, KC>, prefix: &K) -> Option<Arc<TrieNode<K, V>>> {
        let prefix_bytes = KC::convert(prefix);

        // Start at the root
        let mut current = &trie.root;
        let mut remaining = &prefix_bytes[..];

        while !remaining.is_empty() {
            // Try to match as much of the current node's fragment as possible
            let common_len = prefix_match(remaining, 0, &current.key_fragment);

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
                }
                None => return None,
            }
        }

        // If we've consumed the entire prefix, return the current node
        Some(Arc::clone(current))
    }

    // Helper method to count all entries in a subtrie
    fn count_entries_in_subtrie(node: &Arc<TrieNode<K, V>>) -> usize {
        let mut count = if node.data.is_some() { 1 } else { 0 };

        for child in node.children.values() {
            count += Self::count_entries_in_subtrie(child);
        }

        count
    }

    // // Helper method to compare key prefixes
    // fn prefix_match(key: &[u8], fragment: &[u8]) -> usize {
    //     let mut i = 0;
    //     while i < key.len() && i < fragment.len() && key[i] == fragment[i] {
    //         i += 1;
    //     }
    //     i
    // }

    // Helper method to check if a key starts with a prefix
    fn key_starts_with_prefix(key: &K, prefix: &K) -> bool {
        let key_bytes = KC::convert(key);
        let prefix_bytes = KC::convert(prefix);

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

impl<K: Clone + Hash + Eq + fmt::Debug, V: fmt::Debug, KC: KeyToBytes<K> + fmt::Debug> fmt::Debug
    for PrefixView<K, V, KC>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefixView")
            .field("prefix", &self.prefix)
            // Avoid printing the whole trie; root pointer is enough for debugging structure.
            .field("trie_root_ptr", &Arc::as_ptr(&self.trie.root))
            .field(
                "subtrie_node_ptr",
                &self.subtrie_node.as_ref().map(Arc::as_ptr),
            )
            .finish()
    }
}

// KC doesn't necessarily need to be PartialEq itself if its type identity is sufficient.
// The core comparison logic relies on node hashes.
impl<K: Clone + Hash + Eq, V: Hash + Eq + Clone, KC: KeyToBytes<K>> PartialEq
    for PrefixView<K, V, KC>
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
                return Arc::ptr_eq(self_node, other_node) || self_node.hash() == other_node.hash();
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

impl<K: Clone + Hash + Eq, V: Hash + Eq + Clone, KC: KeyToBytes<K>> Eq for PrefixView<K, V, KC> {}

impl<'a, K: Clone + Hash + Eq, V: Clone, KC: KeyToBytes<K> + Clone> IntoIterator for &'a PrefixView<K, V, KC> {
    type Item = (K, V);
    type IntoIter = PrefixViewIter<K, V, KC>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K: Clone + Hash + Eq, V: Clone, KC: KeyToBytes<K> + Clone> Iterator for PrefixViewIter<K, V, KC> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, path)) = self.stack.pop_front() {
            // Add children to the stack (in reverse order for depth-first traversal)
            // Children are explored relative to the subtrie_node, path helps reconstruct full path if needed.
            let mut children_to_visit: Vec<_> = node.children.iter().collect();
            children_to_visit.sort_by_key(|&(k, _)| k); // Ensure deterministic order

            for (branch_byte, child_node) in children_to_visit.into_iter().rev() {
                // Store path info for potential future uses
                let mut child_path = path.clone();
                child_path.extend_from_slice(&node.key_fragment);
                child_path.push(*branch_byte);
                self.stack.push_front((Arc::clone(child_node), child_path));
            }

            // If this node has a key-value pair, yield it (cloned)
            if let Some(kvp) = &node.data {
                // Clone the key and value
                let key = (*kvp.key).clone();
                let value = (*kvp.value).clone();

                // Make sure the key starts with our view's prefix
                let key_bytes = KC::convert(&key);
                let prefix_bytes = KC::convert(&self.view.prefix);
                
                if key_bytes.starts_with(&prefix_bytes) {
                    return Some((key, value));
                }
                // If the key doesn't match our prefix, continue checking other nodes
                // This might happen if subtrie_node itself is for a prefix that is shorter than some keys it contains.
            }
        }

        None
    }
}

// Helper has been consolidated in the PrefixView struct

impl<K: Clone + Hash + Eq, V, KC: KeyToBytes<K>> Iterator for PrefixViewArcIter<K, V, KC> {
    type Item = (Arc<K>, Arc<V>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, path)) = self.stack.pop_front() {
            // Add children to the stack (in reverse order for depth-first traversal)
            // Children are explored relative to the subtrie_node, path helps reconstruct full path if needed.
            let mut children_to_visit: Vec<_> = node.children.iter().collect();
            children_to_visit.sort_by_key(|&(k, _)| k); // Ensure deterministic order

            for (branch_byte, child_node) in children_to_visit.into_iter().rev() {
                // Store path info for potential future uses
                let mut child_path = path.clone();
                child_path.extend_from_slice(&node.key_fragment);
                child_path.push(*branch_byte);
                self.stack.push_front((Arc::clone(child_node), child_path));
            }

            // If this node has a key-value pair, yield it as Arc references
            if let Some(kvp) = &node.data {
                // Make sure the key starts with our view's prefix
                let key_bytes = KC::convert(&*kvp.key);
                let prefix_bytes = KC::convert(&self.view.prefix);
                
                // Simple prefix check
                if key_bytes.starts_with(&prefix_bytes) {
                    return Some((Arc::clone(&kvp.key), Arc::clone(&kvp.value)));
                }
            }
        }

        None
    }
}

// PrefixViewIter now uses the key_starts_with_prefix method from PrefixView directly

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_converter::StrKeyConverter;
    use std::collections::HashSet;
    // It seems util::prefix_match is used internally but not needed in tests directly for PrefixView
    // use crate::util::prefix_match;

    #[test]
    fn test_prefix_view_creation() {
        let trie = Trie::<String, i32, StrKeyConverter<String>>::new_str_key()
            .insert("hello".to_string(), 1)
            .insert("help".to_string(), 2)
            .insert("world".to_string(), 3);

        // Create a view with a prefix that exists
        let view = trie.view_subtrie("hel".to_string());

        // Basic properties
        assert!(view.exists());
        assert_eq!(view.prefix(), &"hel".to_string());
        // Test trie equality by comparing root pointers if clone is used,
        // or if Trie itself implements PartialEq based on content.
        // For now, let's assume view.trie() returns a clone or we check specific properties.
        // assert_eq!(view.trie(), &trie); // This requires Trie to be PartialEq
        assert_eq!(view.len(), 2);
        assert!(!view.is_empty());
    }

    #[test]
    fn test_prefix_view_equality() {
        // Create two tries with the same content
        let trie1 = Trie::<String, u32, StrKeyConverter<String>>::new_str_key()
            .insert("hello".to_string(), 1)
            .insert("help".to_string(), 2);

        let trie2 = Trie::<String, u32, StrKeyConverter<String>>::new_str_key()
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
        let trie3 = Trie::<String, u32, StrKeyConverter<String>>::new_str_key()
            .insert("hello".to_string(), 99) // Different value
            .insert("help".to_string(), 2);

        let view4 = PrefixView::new(trie3, "hel".to_string());

        // It should not be equal to the first view due to different content
        assert_ne!(view1, view4);
    }

    #[test]
    fn test_prefix_view_exists() {
        let trie = Trie::<String, u32, StrKeyConverter<String>>::new_str_key()
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
        let trie = Trie::<String, u32, StrKeyConverter<String>>::new_str_key()
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
        let trie = Trie::<String, u32, StrKeyConverter<String>>::new_str_key()
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
        // "he" is a prefix of "hel" but not a key itself in this view's scope
        assert_eq!(view.get(&"he".to_string()), None);
    }

    #[test]
    fn test_prefix_view_contains_key() {
        let trie = Trie::<String, u32, StrKeyConverter<String>>::new_str_key()
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
    fn test_key_starts_with_prefix_helper() {
        let view_prefix_k_hel = "hel".to_string();
        let view_prefix_k_hello = "hello".to_string();
        let view_prefix_k_help = "help".to_string();

        // Test the PrefixView::key_starts_with_prefix
        let hello_string = "hello".to_string();
        let he_string = "he".to_string();

        assert!(
            PrefixView::<String, u32, StrKeyConverter<String>>::key_starts_with_prefix(
                &hello_string,
                &view_prefix_k_hel
            )
        );
        assert!(
            PrefixView::<String, u32, StrKeyConverter<String>>::key_starts_with_prefix(
                &hello_string,
                &view_prefix_k_hello
            )
        );
        assert!(
            !PrefixView::<String, u32, StrKeyConverter<String>>::key_starts_with_prefix(
                &hello_string,
                &view_prefix_k_help
            )
        );
        assert!(
            !PrefixView::<String, u32, StrKeyConverter<String>>::key_starts_with_prefix(
                &he_string,
                &view_prefix_k_hel
            )
        );

        // Test the key prefix matching functionality
        let hello_string = "hello".to_string();
        let hello_key_str = "hello".to_string();
        let he_string = "he".to_string();
        let help_string = "help".to_string();

        assert!(PrefixView::<String, u32, StrKeyConverter<String>>::key_starts_with_prefix(
            &hello_string, &view_prefix_k_hel)
        );
        assert!(
            PrefixView::<String, u32, StrKeyConverter<String>>::key_starts_with_prefix(
                &hello_string, &hello_key_str)
        );
        assert!(
            !PrefixView::<String, u32, StrKeyConverter<String>>::key_starts_with_prefix(
                &hello_string, &help_string)
        );
        assert!(
            !PrefixView::<String, u32, StrKeyConverter<String>>::key_starts_with_prefix(
                &he_string, &view_prefix_k_hel)
        );
    }

    #[test]
    fn test_prefix_view_iter() {
        let trie = Trie::<String, u32, StrKeyConverter<String>>::new_str_key()
            .insert("hello".to_string(), 1)
            .insert("help".to_string(), 2)
            .insert("world".to_string(), 3);

        // View with the "hel" prefix
        let view = PrefixView::new(trie.clone(), "hel".to_string());

        // Collect results into a set for easier comparison
        let results: HashSet<(String, u32)> = view.iter().collect();

        // Expected results - these will be cloned values
        let expected: HashSet<(String, u32)> = vec![
            ("hello".to_string(), 1), 
            ("help".to_string(), 2)
        ].into_iter().collect();

        assert_eq!(results, expected);

        // Also test the Arc-based iterator
        let arc_results: Vec<(Arc<String>, Arc<u32>)> = view.iter_arc().collect();
        assert_eq!(arc_results.len(), 2);
        assert!(arc_results.iter().any(|(k, v)| **k == "hello" && **v == 1));
        assert!(arc_results.iter().any(|(k, v)| **k == "help" && **v == 2));

        // View with a more specific prefix
        let view2 = PrefixView::new(trie.clone(), "hello".to_string());
        let results2: Vec<(String, u32)> = view2.iter().collect();

        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0], ("hello".to_string(), 1));

        // View with a prefix that doesn't match any keys
        let view3 = PrefixView::new(trie.clone(), "xyz".to_string());
        let results3: Vec<(String, u32)> = view3.iter().collect();

        assert!(results3.is_empty());
    }

    #[test]
    fn test_prefix_view_iter_order() {
        // Test that iteration happens in a deterministic order
        let trie = Trie::<String, u32, StrKeyConverter<String>>::new_str_key()
            .insert("aa".to_string(), 1)
            .insert("ab".to_string(), 2)
            .insert("ac".to_string(), 3)
            .insert("ba".to_string(), 4);

        // View with the "a" prefix
        let view = PrefixView::new(trie.clone(), "a".to_string());

        // Get results in order
        let results: Vec<(String, u32)> = view.iter().collect();

        // Should have three items
        assert_eq!(results.len(), 3);

        // Just verify all expected elements are present.
        // Note: TrieIter sorts children by first byte before pushing to stack,
        // so it should be lexicographical.
        
        assert!(results.contains(&("aa".to_string(), 1)));
        assert!(results.contains(&("ab".to_string(), 2)));
        assert!(results.contains(&("ac".to_string(), 3)));
        assert!(!results.contains(&("ba".to_string(), 4)));

        // Explicitly check order for the "a" prefix view
        let expected_a_results = vec![
            ("aa".to_string(), 1), 
            ("ab".to_string(), 2), 
            ("ac".to_string(), 3)
        ];
        assert_eq!(results, expected_a_results);
        
        // Test the Arc-based iterator also maintains order
        let arc_results: Vec<(Arc<String>, Arc<u32>)> = view.iter_arc().collect();
        assert_eq!(arc_results.len(), 3);
        assert_eq!(*arc_results[0].0, "aa".to_string());
        assert_eq!(*arc_results[1].0, "ab".to_string());
        assert_eq!(*arc_results[2].0, "ac".to_string());
    }
}
