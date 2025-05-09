use radix_trie::Trie;

#[test]
fn test_prefix_view_creation() {
    let trie = Trie::<String, i32>::new()
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
    let trie = Trie::<String, u32>::new()
        .insert("zebra".to_string(), 5)
        .insert("apple".to_string(), 1)
        .insert("banana".to_string(), 2)
        .insert("cherry".to_string(), 3)
        .insert("date".to_string(), 4);
    
    // Create a view of the entire trie with empty prefix
    let view = trie.view_subtrie(String::new());
    
    // Collect the keys in iteration order
    let keys: Vec<String> = view.iter()
        .map(|(k, _)| k.clone())
        .collect();
    
    // Expected keys in lexicographic order
    let expected = vec![
        "apple".to_string(),
        "banana".to_string(),
        "cherry".to_string(),
        "date".to_string(),
        "zebra".to_string(),
    ];
    
    // Verify that keys are returned in lexicographic order
    assert_eq!(keys, expected);
    
    // Now test with a specific prefix
    let b_view = trie.insert("berry".to_string(), 6)
                     .view_subtrie("b".to_string());
    
    let b_keys: Vec<String> = b_view.iter()
        .map(|(k, _)| k.clone())
        .collect();
    
    let expected_b_keys = vec![
        "banana".to_string(),
        "berry".to_string(),
    ];
    
    assert_eq!(b_keys, expected_b_keys);
    
    // Test with a multi-key prefix that includes part of some keys
    let complex_trie = Trie::<String, u32>::new()
        .insert("abcd".to_string(), 1)
        .insert("abce".to_string(), 2)
        .insert("abcf".to_string(), 3)
        .insert("abdc".to_string(), 4)
        .insert("abde".to_string(), 5);
    
    let complex_view = complex_trie.view_subtrie("abc".to_string());
    
    let complex_keys: Vec<String> = complex_view.iter()
        .map(|(k, _)| k.clone())
        .collect();
    
    let expected_complex_keys = vec![
        "abcd".to_string(),
        "abce".to_string(),
        "abcf".to_string(),
    ];
    
    assert_eq!(complex_keys, expected_complex_keys);
}

#[test]
fn test_prefix_view_nonexistent() {
    let trie = Trie::<String, u32>::new()
        .insert("hello".to_string(), 1)
        .insert("world".to_string(), 2);
    
    let view = trie.view_subtrie("xyz".to_string());
    
    assert!(!view.exists());
    assert_eq!(view.len(), 0);
    assert!(view.is_empty());
}

#[test]
fn test_prefix_view_subtree_equality() {
    // Create a trie with items at paths a, b, c
    let trie1 = Trie::<String, u32>::new()
        .insert("a".to_string(), 1)
        .insert("b".to_string(), 2)
        .insert("c".to_string(), 3);
    
    // Get the view of subtrie at path "b"
    let view_b1 = trie1.view_subtrie("b".to_string());
    
    // Check that view_b1 only contains one item
    assert_eq!(view_b1.len(), 1);
    assert!(view_b1.contains_key(&"b".to_string()));
    assert!(!view_b1.contains_key(&"a".to_string()));
    assert!(!view_b1.contains_key(&"c".to_string()));
    
    // Create a new trie with an added item at path "cc"
    let trie2 = trie1.insert("cc".to_string(), 4);
    
    // Get the view of subtrie at path "b" from the new trie
    let view_b2 = trie2.view_subtrie("b".to_string());
    
    // Check that view_b2 is equal to view_b1
    // since adding "cc" doesn't affect the "b" subtree
    assert_eq!(view_b1, view_b2);
    
    // Create a new trie with an added item at path "bb"
    // which should affect the "b" subtree
    let trie3 = trie1.insert("bb".to_string(), 5);
    
    // Get the view of subtrie at path "b" from this newest trie
    let view_b3 = trie3.view_subtrie("b".to_string());
    
    // Check that view_b3 is different from view_b1 and view_b2
    // since adding "bb" does affect the "b" subtree
    assert_ne!(view_b1, view_b3);
    assert_ne!(view_b2, view_b3);
    
    // Verify the content of view_b3
    assert_eq!(view_b3.len(), 2); // Should contain "b" and "bb"
    assert!(view_b3.contains_key(&"b".to_string()));
    assert!(view_b3.contains_key(&"bb".to_string()));
    assert!(!view_b3.contains_key(&"a".to_string()));
    assert!(!view_b3.contains_key(&"c".to_string()));
    assert!(!view_b3.contains_key(&"cc".to_string()));
    
    // Create a trie4 with all items
    let trie4 = trie1
        .insert("bb".to_string(), 5)
        .insert("cc".to_string(), 4);
    
    // Views from different paths should not be equal
    let view_a = trie4.view_subtrie("a".to_string());
    let view_b = trie4.view_subtrie("b".to_string());
    let view_c = trie4.view_subtrie("c".to_string());
    
    assert_ne!(view_a, view_b);
    assert_ne!(view_a, view_c);
    assert_ne!(view_b, view_c);
    
    // But views of the same subtree from different tries with 
    // the same content in that subtree should be equal
    let view_b_again = trie3.view_subtrie("b".to_string());
    assert_eq!(view_b, view_b_again);
}

#[test]
fn test_prefix_view_equality() {
    // Create two tries with the same content
    let trie1 = Trie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2);
        
    let trie2 = Trie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2);
    
    // Create views with the same prefix
    let view1 = trie1.view_subtrie("hel".to_string());
    let view2 = trie2.view_subtrie("hel".to_string());
    
    // They should be equal because they represent the same subtrie structure
    assert_eq!(view1, view2);
    
    // Create a trie with different content
    let trie3 = Trie::<String, i32>::new()
        .insert("hello".to_string(), 99)  // Different value
        .insert("help".to_string(), 2);
        
    let view3 = trie3.view_subtrie("hel".to_string());
    
    // It should not be equal to the first view due to different content
    assert_ne!(view1, view3);
    
    // Create a view with a different prefix but same content
    let view4 = trie1.view_subtrie("he".to_string());
    
    // They should be equal because they point to the same subtrie content
    assert_eq!(view1, view4);
}

#[test]
fn test_prefix_view_contains_key() {
    let trie = Trie::<String, i32>::new()
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
    let trie = Trie::<String, i32>::new()
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
fn test_prefix_view_partial_match() {
    let trie = Trie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("world".to_string(), 2);
        
    // Prefix is a partial match of "hello"
    let view = trie.view_subtrie("he".to_string());
    
    assert!(view.exists());
    assert_eq!(view.len(), 1);
    assert!(view.contains_key(&"hello".to_string()));
    assert_eq!(view.get(&"hello".to_string()), Some(&1));
    
    // Should not contain the unrelated key
    assert!(!view.contains_key(&"world".to_string()));
}

#[test]
fn test_prefix_view_nested() {
    let trie = Trie::<String, i32>::new()
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
    let trie = Trie::<String, i32>::new()
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
    let trie = Trie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2);
        
    let view = trie.view_subtrie("hel".to_string());
    let view_clone = view.clone();
    
    // The clone should be equivalent to the original
    assert_eq!(view, view_clone);
    assert_eq!(view.len(), view_clone.len());
    assert!(view_clone.contains_key(&"hello".to_string()));
    
    // Both views should get the same values
    assert_eq!(view.get(&"hello".to_string()), view_clone.get(&"hello".to_string()));
    assert_eq!(view.get(&"help".to_string()), view_clone.get(&"help".to_string()));
}