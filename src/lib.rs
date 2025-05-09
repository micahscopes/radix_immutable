//! # Radix Trie
//!
//! A radix trie implementation with structural sharing and fast prefix comparison.
//!
//! This crate provides an immutable radix trie (also known as a patricia trie) that uses structural
//! sharing via `Arc` to efficiently create new versions of the trie while sharing unchanged parts.
//!
//! ## Features
//!
//! - **Immutable API**: All modifying operations return a new trie instance
//! - **Structural Sharing**: Efficient memory usage through shared unchanged subtrees
//! - **Fast Prefix Comparison**: Compare subtrees efficiently using structural hashing
//! - **Prefix Views**: Create lightweight views into trie prefixes with fast comparison and lookup support
//!
//! ## Example
//!
//! ```rust
//! use radix_trie::Trie;
//!
//! // Create a new trie
//! let trie = Trie::<String, u32>::new();
//!
//! // Insert some values (each operation returns a new trie)
//! let trie = trie.insert("hello".to_string(), 1);
//! let trie = trie.insert("world".to_string(), 2);
//!
//! // Lookup values
//! assert_eq!(trie.get(&"hello".to_string()), Some(&1));
//! ```

pub mod node;
mod prefix_view;
mod trie;
mod util;

// Re-export public types
pub use crate::prefix_view::PrefixView;
pub use crate::trie::Trie;
pub use crate::node::TrieNode;

/// PrefixView provides efficient views of subtries based on key prefixes.
/// This enables fast comparison between subtries with the same prefix, as well
/// as key lookups and existence checks.
///
/// ```rust
/// use radix_trie::Trie;
///
/// let trie = Trie::<String, u32>::new()
///     .insert("hello".to_string(), 1)
///     .insert("help".to_string(), 2);
///
/// // Create a view of the "hel" prefix
/// let view = trie.view_subtrie("hel".to_string());
///
/// // Check if a key exists in the view
/// assert!(view.contains_key(&"hello".to_string()));
/// 
/// // Get a value from the view
/// assert_eq!(view.get(&"hello".to_string()), Some(&1));
/// ```
///
/// Errors that can occur in trie operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Key is invalid for the operation
    InvalidKey,
    /// Other error with description
    Other(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidKey => write!(f, "Invalid key for this operation"),
            Error::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for Error {}