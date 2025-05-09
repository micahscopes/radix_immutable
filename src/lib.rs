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
//! use radix_immutable::StringTrie;
//!
//! // Create a new trie
//! let trie = StringTrie::<String, i32>::new();
//!
//! // Insert some values (each operation returns a new trie)
//! let trie = trie.insert("hello".to_string(), 1);
//! let trie = trie.insert("world".to_string(), 2);
//!
//! // Lookup values
//! assert_eq!(trie.get(&"hello".to_string()), Some(&1));
//! ```
//!
//! You can also use custom types with a specialized key converter:
//!
//! ```rust
//! use radix_immutable::{Trie, KeyToBytes};
//! use std::path::{Path, PathBuf};
//! use std::borrow::Cow;
//! use std::marker::PhantomData;
//!
//! // Create a custom key converter for PathBuf
//! #[derive(Clone, Debug)]
//! struct PathKeyConverter<K>(PhantomData<K>);
//!
//! impl<K> Default for PathKeyConverter<K> {
//!     fn default() -> Self {
//!         Self(PhantomData)
//!     }
//! }
//!
//! impl<K: Clone + std::hash::Hash + Eq + AsRef<Path>> KeyToBytes<K> for PathKeyConverter<K> {
//!     fn convert<'a>(key: &'a K) -> Cow<'a, [u8]> {
//!         // Convert path to string and then to bytes
//!         let path_str = key.as_ref().to_string_lossy();
//!         Cow::Owned(path_str.as_bytes().to_vec())
//!     }
//! }
//!
//! // Create a trie that uses PathBuf as keys with our custom converter
//! let trie = Trie::<PathBuf, i32, PathKeyConverter<PathBuf>>::new();
//! ```

pub mod key_converter;
pub mod node;
mod prefix_view;
mod trie;
mod util;

// Re-export public types
pub use crate::key_converter::{BytesKeyConverter, KeyToBytes, StrKeyConverter};
pub use crate::node::TrieNode;
pub use crate::prefix_view::{PrefixView, PrefixViewIter, PrefixViewArcIter};
pub use crate::trie::Trie;

/// Type alias for a Trie that uses string-based keys (anything implementing `AsRef<str>`)
///
/// This provides a convenient type for working with any keys that can be converted to strings.
/// Examples of compatible types include: String, &str, PathBuf, Url, etc.
pub type StringTrie<K, V> = Trie<K, V, StrKeyConverter<K>>;

/// Type alias for a Trie that uses byte-based keys (anything implementing AsRef<[u8]>)
///
/// This provides a convenient type for working with any keys that can be converted to byte slices.
/// Examples of compatible types include: `Vec<u8>`, `&[u8]`, etc.
pub type BytesTrie<K, V> = Trie<K, V, BytesKeyConverter<K>>;

/// Type alias for a PrefixView of a StringTrie
///
/// This provides a convenient type for working with prefix views over string-based keys.
pub type StringPrefixView<'a, K, V> = PrefixView<K, V, StrKeyConverter<K>>;

/// Type alias for a PrefixView of a BytesTrie
///
/// This provides a convenient type for working with prefix views over byte-based keys.
pub type BytesPrefixView<'a, K, V> = PrefixView<K, V, BytesKeyConverter<K>>;

/// PrefixView provides efficient views of subtries based on key prefixes.
/// This enables fast comparison between subtries with the same prefix, as well
/// as key lookups and existence checks.
///
/// ```rust
/// use radix_immutable::StringTrie;
///
/// let trie = StringTrie::<String, u32>::new()
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
/// PrefixView also works with various key types:
///
/// ```rust
/// use radix_immutable::{StringTrie, StringPrefixView};
///
/// // Create a trie with string keys
/// let trie = StringTrie::<String, i32>::new()
///     .insert("hello".to_string(), 1)
///     .insert("help".to_string(), 2);
///
/// // Create a prefix view of the trie
/// let view = trie.view_subtrie("hel".to_string());
/// assert_eq!(view.len(), 2);
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
