//! Defines traits and structs for converting trie keys into byte sequences.
use std::borrow::Cow;
use std::hash::Hash;
use std::marker::PhantomData;

/// A trait for types that can convert a key of type `K` into a byte slice.
/// The key `K` itself must satisfy `Clone + Hash + Eq` for use in the trie.
pub trait KeyToBytes<K: Clone + Hash + Eq>: Clone {
    /// Converts the given key into a `Cow<[u8]>`.
    /// `Cow` allows for borrowing if the key can provide a direct slice,
    /// or owning (e.g., via `Vec<u8>`) if a conversion is necessary.
    fn convert<'a>(key: &'a K) -> Cow<'a, [u8]>;
}

/// A key converter for keys that implement `AsRef<str>`.
/// Examples: `String`, `&'static str`.
#[derive(Debug, Clone)]
pub struct StrKeyConverter<K>(PhantomData<K>);

impl<K> StrKeyConverter<K> {
    /// Creates a new string key converter
    pub fn new() -> Self {
        StrKeyConverter(PhantomData)
    }
}

impl<K> Default for StrKeyConverter<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Clone + Hash + Eq + AsRef<str>> KeyToBytes<K> for StrKeyConverter<K> {
    fn convert<'a>(key: &'a K) -> Cow<'a, [u8]> {
        Cow::Borrowed(key.as_ref().as_bytes())
    }
}

/// A key converter for keys that implement `AsRef<[u8]>`.
/// Examples: `Vec<u8>`, `&'static [u8]`, `String`.
#[derive(Debug, Clone)]
pub struct BytesKeyConverter<K>(PhantomData<K>);

impl<K> BytesKeyConverter<K> {
    /// Creates a new bytes key converter
    pub fn new() -> Self {
        BytesKeyConverter(PhantomData)
    }
}

impl<K> Default for BytesKeyConverter<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Clone + Hash + Eq + AsRef<[u8]>> KeyToBytes<K> for BytesKeyConverter<K> {
    fn convert<'a>(key: &'a K) -> Cow<'a, [u8]> {
        Cow::Borrowed(key.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_str_key_converter_string() {
        let key = "hello".to_string();
        let bytes = StrKeyConverter::<String>::convert(&key);
        assert_eq!(bytes.as_ref(), b"hello");
        assert!(matches!(bytes, Cow::Borrowed(_)));
    }

    #[test]
    fn test_str_key_converter_static_str() {
        let key: &'static str = "world";
        let bytes = StrKeyConverter::<&'static str>::convert(&key);
        assert_eq!(bytes.as_ref(), b"world");
        assert!(matches!(bytes, Cow::Borrowed(_)));
    }

    #[test]
    fn test_bytes_key_converter_vec_u8() {
        let key = vec![1, 2, 3];
        let bytes = BytesKeyConverter::<Vec<u8>>::convert(&key);
        assert_eq!(bytes.as_ref(), &[1, 2, 3]);
        assert!(matches!(bytes, Cow::Borrowed(_)));
    }

    #[test]
    fn test_bytes_key_converter_static_slice_u8() {
        let key: &'static [u8] = b"data";
        let bytes = BytesKeyConverter::<&'static [u8]>::convert(&key);
        assert_eq!(bytes.as_ref(), b"data");
        assert!(matches!(bytes, Cow::Borrowed(_)));
    }

    #[test]
    fn test_bytes_key_converter_string() {
        // String also implements AsRef<[u8]>
        let key = "hello_bytes".to_string();
        let bytes = BytesKeyConverter::<String>::convert(&key);
        assert_eq!(bytes.as_ref(), b"hello_bytes");
        assert!(matches!(bytes, Cow::Borrowed(_)));
    }

    // Custom type that implements AsRef<str> but not AsRef<[u8]>
    #[derive(Clone, Debug, Hash, Eq, PartialEq)]
    struct MyCustomStrKey(String);
    impl AsRef<str> for MyCustomStrKey {
        fn as_ref(&self) -> &str {
            self.0.as_ref()
        }
    }

    #[test]
    fn test_str_key_converter_custom_str() {
        let key = MyCustomStrKey("custom_str".to_string());
        let bytes = StrKeyConverter::<MyCustomStrKey>::convert(&key);
        assert_eq!(bytes.as_ref(), b"custom_str");
    }

    // Custom type that implements AsRef<[u8]> but not AsRef<str>
    #[derive(Clone, Debug, Hash, Eq, PartialEq)]
    struct MyCustomBytesKey(Vec<u8>);
    impl AsRef<[u8]> for MyCustomBytesKey {
        fn as_ref(&self) -> &[u8] {
            self.0.as_ref()
        }
    }

    #[test]
    fn test_bytes_key_converter_custom_bytes() {
        let key = MyCustomBytesKey(vec![10, 20, 30]);
        let bytes = BytesKeyConverter::<MyCustomBytesKey>::convert(&key);
        assert_eq!(bytes.as_ref(), &[10, 20, 30]);
    }
}