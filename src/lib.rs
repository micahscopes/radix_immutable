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
//! - **Prefix Views**: Create lightweight views into trie prefixes
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

mod node;
mod prefix_view;
mod trie;
mod util;

// Re-export public types
pub use crate::prefix_view::PrefixView;
pub use crate::trie::Trie;

/// Errors that can occur in the trie operations
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