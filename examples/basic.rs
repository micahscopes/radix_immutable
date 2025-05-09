//! Examples of using the radix trie
#[allow(unused_imports)]
use radix_immutable::{StrKeyConverter, StringTrie, Trie};

fn main() {
    // Create a new trie with string keys
    let trie = StringTrie::<String, i32>::new();

    // Insert some values
    let trie = trie.insert("hello".to_string(), 1);
    let trie = trie.insert("world".to_string(), 2);

    // Check values
    assert_eq!(trie.get(&"hello".to_string()), Some(&1));
    assert_eq!(trie.get(&"world".to_string()), Some(&2));
    assert_eq!(trie.get(&"missing".to_string()), None);

    // Alternatively, use the explicit type parameters
    let trie2 = Trie::<String, u32, StrKeyConverter<String>>::new();
    let trie2 = trie2.insert("test".to_string(), 42);

    assert_eq!(trie2.get(&"test".to_string()), Some(&42));

    // Or use the generic StringTrie that accepts any AsRef<str> key
    let trie3 = StringTrie::<String, u32>::new();
    let trie3 = trie3.insert("example".to_string(), 100);

    assert_eq!(trie3.get(&"example".to_string()), Some(&100));
}

#[test]
fn test_prefix_view() {
    let trie = StringTrie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2)
        .insert("world".to_string(), 3);

    // Create a view of the "hel" prefix
    let view = trie.view_subtrie("hel".to_string());

    // Check prefix view properties
    assert!(view.exists());
    assert_eq!(view.len(), 2);
    assert!(!view.is_empty());

    // Check key existence in the view
    assert!(view.contains_key(&"hello".to_string()));
    assert!(view.contains_key(&"help".to_string()));
    assert!(!view.contains_key(&"world".to_string()));

    // Get values from the view
    assert_eq!(view.get(&"hello".to_string()), Some(&1));
    assert_eq!(view.get(&"help".to_string()), Some(&2));
    assert_eq!(view.get(&"world".to_string()), None);
}
