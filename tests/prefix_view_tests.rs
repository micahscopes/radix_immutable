use radix_trie::{Trie, PrefixView};

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
fn test_prefix_view_nonexistent() {
    let trie = Trie::<String, i32>::new()
        .insert("hello".to_string(), 1)
        .insert("help".to_string(), 2);
        
    // Create a view with a prefix that doesn't exist
    let view = trie.view_subtrie("xyz".to_string());
    
    // Should not exist
    assert!(!view.exists());
    assert_eq!(view.len(), 0);
    assert!(view.is_empty());
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