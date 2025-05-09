use radix_immutable::StringTrie;
use std::collections::HashSet;

#[test]
fn test_prefix_view_creation() {
    let trie = StringTrie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2)
        .insert("world".to_string(), 3);

    // Create a view with a prefix that exists
    let view = trie.view_subtrie("hel".to_string());

    // Basic properties
    assert!(view.exists());
    assert_eq!(view.prefix(), &"hel".to_string());
    assert_eq!(view.trie(), &trie);
    assert_eq!(view.len(), 2);
    assert!(!view.is_empty());
}

#[test]
fn test_prefix_view_lexicographic_iteration() {
    // Create a trie with keys deliberately not in lexicographic order
    let trie = StringTrie::<String, u32>::new()
        .insert("zebra".to_string(), 5)
        .insert("apple".to_string(), 1)
        .insert("banana".to_string(), 2)
        .insert("cherry".to_string(), 3)
        .insert("date".to_string(), 4)
        .insert("apricot".to_string(), 6)
        .insert("blueberry".to_string(), 7)
        .insert("blackberry".to_string(), 8);

    // View for 'a' prefix
    let view_a = trie.view_subtrie("a".to_string());

    // Collect results
    let mut a_keys = Vec::new();
    for (key, _) in &view_a {
        a_keys.push(key.clone());
    }

    // Expected keys should be in lexicographic order
    let expected_a_keys = vec!["apple".to_string(), "apricot".to_string()];

    assert_eq!(a_keys, expected_a_keys);

    // View for 'b' prefix
    let view_b = trie.view_subtrie("b".to_string());

    // Collect results
    let mut b_keys = Vec::new();
    for (key, _) in &view_b {
        b_keys.push(key.clone());
    }

    // Expected keys should be in lexicographic order
    let expected_b_keys = vec![
        "banana".to_string(),
        "blackberry".to_string(),
        "blueberry".to_string(),
    ];

    assert_eq!(b_keys, expected_b_keys);

    // Test with a multi-key prefix that includes part of some keys
    let complex_trie = StringTrie::<String, u32>::new()
        .insert("abcd".to_string(), 1)
        .insert("abce".to_string(), 2)
        .insert("abcf".to_string(), 3)
        .insert("abcg".to_string(), 4)
        .insert("abd".to_string(), 5);

    let view_abc = complex_trie.view_subtrie("abc".to_string());

    let mut abc_keys = Vec::new();
    for (key, _) in &view_abc {
        abc_keys.push(key.clone());
    }

    let expected_abc_keys = vec![
        "abcd".to_string(),
        "abce".to_string(),
        "abcf".to_string(),
        "abcg".to_string(),
    ];

    assert_eq!(abc_keys, expected_abc_keys);
}

#[test]
fn test_prefix_view_nonexistent() {
    let trie = StringTrie::<String, u32>::new()
        .insert("hello".to_string(), 1)
        .insert("world".to_string(), 2);

    let view = trie.view_subtrie("xyz".to_string());

    assert!(!view.exists());
    assert_eq!(view.len(), 0);
    assert!(view.is_empty());
    assert!(!view.contains_key(&"hello".to_string()));
}

#[test]
fn test_prefix_view_subtree_equality() {
    // Create a trie with items at paths a, b, c
    let trie1 = StringTrie::<String, u32>::new()
        .insert("a".to_string(), 1)
        .insert("b".to_string(), 2)
        .insert("c".to_string(), 3);

    // Access the same subtree in different ways
    let view1 = trie1.view_subtrie("a".to_string());
    let view2 = trie1.view_subtrie("a".to_string());

    // Views to the same subtree should be equal
    assert_eq!(view1, view2);

    // But views to different subtrees should not be equal
    let view3 = trie1.view_subtrie("b".to_string());
    assert_ne!(view1, view3);

    // Create a trie with the same 'a' entry
    let trie2 = StringTrie::<String, u32>::new()
        .insert("a".to_string(), 1)
        .insert("x".to_string(), 9)
        .insert("y".to_string(), 10);

    // Create a view to the 'a' subtree
    let view4 = trie2.view_subtrie("a".to_string());

    // Views to structurally equivalent subtrees should be equal
    // even if they come from different tries
    assert_eq!(view1, view4);

    // But if the values are different, the views should not be equal
    let trie3 = StringTrie::<String, u32>::new().insert("a".to_string(), 99); // Different value

    let view5 = trie3.view_subtrie("a".to_string());
    assert_ne!(view1, view5);

    // Test with a prefix that doesn't exist in either trie
    let view6 = trie1.view_subtrie("z".to_string());
    let view7 = trie2.view_subtrie("z".to_string());

    // Views to non-existent subtrees should be equal
    assert_eq!(view6, view7);
}

#[test]
fn test_prefix_view_equality() {
    // Create two tries with the same content
    let trie1 = StringTrie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2);

    let trie2 = StringTrie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2);

    // Create views with the same prefix
    let view1 = trie1.view_subtrie("hel".to_string());
    let view2 = trie2.view_subtrie("hel".to_string());

    // They should be equal
    assert_eq!(view1, view2);

    // Create a trie with different content
    let trie3 = StringTrie::<String, i32>::new()
        .insert("hello".to_string(), 99) // Different value
        .insert("help".to_string(), 2);

    let view3 = trie3.view_subtrie("hel".to_string());

    // Should not be equal due to different values
    assert_ne!(view1, view3);
}

#[test]
fn test_prefix_view_contains_key() {
    let trie = StringTrie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2)
        .insert("world".to_string(), 3);

    // View with the "hel" prefix
    let view = trie.view_subtrie("hel".to_string());

    // Should contain keys in the view
    assert!(view.contains_key(&"hello".to_string()));
    assert!(view.contains_key(&"help".to_string()));

    // Should not contain keys outside the prefix
    assert!(!view.contains_key(&"world".to_string()));
    assert!(!view.contains_key(&"he".to_string()));
}

#[test]
fn test_prefix_view_get() {
    let trie = StringTrie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2)
        .insert("world".to_string(), 3);

    // View with the "hel" prefix
    let view = trie.view_subtrie("hel".to_string());

    // Should be able to get values in the view
    assert_eq!(view.get(&"hello".to_string()), Some(&1));
    assert_eq!(view.get(&"help".to_string()), Some(&2));

    // Should not find values outside the prefix
    assert_eq!(view.get(&"world".to_string()), None);
    assert_eq!(view.get(&"he".to_string()), None);
}

#[test]
fn test_prefix_view_iter() {
    let trie = StringTrie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2)
        .insert("world".to_string(), 3);

    // View with the "hel" prefix
    let view = trie.view_subtrie("hel".to_string());

    // Collect results into a set for easier comparison
    let results: HashSet<(String, i32)> = view.iter().collect();

    // Create keys with proper ownership
    let hello_key = "hello".to_string();
    let help_key = "help".to_string();

    // Expected results - convert to a HashSet for easier comparison
    let expected: HashSet<(String, i32)> =
        vec![(hello_key, 1), (help_key, 2)].into_iter().collect();

    assert_eq!(results, expected);
}

#[test]
fn test_prefix_view_non_existent_prefix() {
    let trie = StringTrie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2);

    // Prefix is a partial match of "hello"
    let view = trie.view_subtrie("he".to_string());

    // The prefix exists, but is not a key itself
    assert!(view.exists());
    assert_eq!(view.len(), 2);
    assert!(!view.contains_key(&"he".to_string()));

    // But it contains both keys that have the prefix
    assert!(view.contains_key(&"hello".to_string()));
    assert!(view.contains_key(&"help".to_string()));
}

#[test]
fn test_prefix_view_nested() {
    let trie = StringTrie::<String, i32>::new()
        .insert("a".to_string(), 1)
        .insert("ab".to_string(), 2)
        .insert("abc".to_string(), 3)
        .insert("abcd".to_string(), 4);

    // Create nested views
    let view_a = trie.view_subtrie("a".to_string());
    let view_ab = trie.view_subtrie("ab".to_string());
    let view_abc = trie.view_subtrie("abc".to_string());

    // Check lengths
    assert_eq!(view_a.len(), 4);
    assert_eq!(view_ab.len(), 3);
    assert_eq!(view_abc.len(), 2);

    // Check specific keys
    assert!(view_a.contains_key(&"a".to_string()));
    assert!(view_a.contains_key(&"abc".to_string()));

    assert!(!view_ab.contains_key(&"a".to_string()));
    assert!(view_ab.contains_key(&"ab".to_string()));
    assert!(view_ab.contains_key(&"abc".to_string()));

    assert!(!view_abc.contains_key(&"ab".to_string()));
    assert!(view_abc.contains_key(&"abc".to_string()));
    assert!(view_abc.contains_key(&"abcd".to_string()));
}

#[test]
fn test_prefix_view_empty_prefix() {
    let trie = StringTrie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("world".to_string(), 2);

    // Empty prefix should match the entire trie
    let view = trie.view_subtrie("".to_string());

    assert!(view.exists());
    assert_eq!(view.len(), 2);
    assert!(view.contains_key(&"hello".to_string()));
    assert!(view.contains_key(&"world".to_string()));
}

#[test]
fn test_prefix_view_cloning() {
    let trie = StringTrie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2);

    let view = trie.view_subtrie("hel".to_string());
    let view_clone = view.clone();

    // The clone should be equivalent to the original
    assert_eq!(view, view_clone);
    assert_eq!(view.len(), view_clone.len());
    assert!(view_clone.contains_key(&"hello".to_string()));

    // Both views should get the same values
    assert_eq!(
        view.get(&"hello".to_string()),
        view_clone.get(&"hello".to_string())
    );
    assert_eq!(
        view.get(&"help".to_string()),
        view_clone.get(&"help".to_string())
    );
}
