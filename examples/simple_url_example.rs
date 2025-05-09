//! Simple example using url::Url as keys in a radix trie
use radix_immutable::StringTrie;
use url::Url;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new trie for URL keys and string values
    let mut url_trie = StringTrie::<Url, String>::new();

    // Create some example URLs
    let home = Url::parse("https://example.com/")?;
    let about = Url::parse("https://example.com/about")?;
    let contact = Url::parse("https://example.com/contact")?;
    let blog = Url::parse("https://example.com/blog")?;
    let blog_post = Url::parse("https://example.com/blog/first-post")?;

    // Insert values associated with each URL
    url_trie = url_trie.insert(home.clone(), "Home page".to_string());
    url_trie = url_trie.insert(about.clone(), "About us".to_string());
    url_trie = url_trie.insert(contact.clone(), "Contact info".to_string());
    url_trie = url_trie.insert(blog.clone(), "Blog index".to_string());
    url_trie = url_trie.insert(blog_post.clone(), "First blog post".to_string());

    // Lookup values by URL
    println!("Looking up URLs:");
    println!("  {} → {:?}", home, url_trie.get(&home));
    println!("  {} → {:?}", blog, url_trie.get(&blog));
    println!("  {} → {:?}", blog_post, url_trie.get(&blog_post));

    // Get a prefix view for the blog section
    let blog_view = url_trie.view_subtrie(blog.clone());

    println!("\nBlog section pages:");
    for (url, content) in blog_view.iter() {
        println!("  {} → {}", url, content);
    }

    // Check if a URL exists
    let unknown = Url::parse("https://example.com/unknown")?;
    println!("\nURL existence check:");
    println!("  {} exists: {}", home, url_trie.contains_key(&home));
    println!("  {} exists: {}", unknown, url_trie.contains_key(&unknown));

    Ok(())
}
