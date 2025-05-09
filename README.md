Radix Immutable
====

This is a [Radix Trie][radix-wiki] implementation in Rust that focuses on immutability and efficient prefix operations. This started as a fork of the original `radix_trie` crate and was rewritten with the assistance of LLM tools.

# Features

* Immutable API with structural sharing for efficient versioning
* Fast prefix comparison between trees
* Compressed nodes with common key prefixes stored only once
* Structural hashing for efficient subtree comparison
* Key generic - supports any type that can be converted to bytes
* Prefix views for efficiently querying and comparing subtries
* Safe - no unsafe code

# Documentation

https://docs.rs/radix-immutable/

# Usage

Available on [Crates.io][] as [`radix-immutable`][radix-immutable].

Add `radix-immutable` to the dependencies section of your `Cargo.toml`:

```toml
# Basic usage
radix-immutable = "0.1"
```

## Original `radix_trie` Contributors

* Allan Simon ([@allan-simon](https://github.com/allan-simon))
* Andrew Smith ([@andrewcsmith](https://github.com/andrewcsmith))
* Arthur Carcano ([@NougatRillettes](https://github.com/NougatRillettes))
* Devin Ragotzy ([@DevinR528](https://github.com/DevinR528))
* [@hanabi1224](https://github.com/hanabi1224)
* Jakob Dalsgaard ([@jakobdalsgaard](https://github.com/jakobdalsgaard))
* Michael Sproul ([@michaelsproul](https://github.com/michaelsproul))
* Robin Lambertz ([@roblabla](https://github.com/roblabla))
* Sergey ([@Albibek](https://github.com/Albibek))
* Stuart Hinson ([@stuarth](https://github.com/stuarth))
* Vinzent Steinberg ([@vks](https://github.com/vks))

# License

MIT License. Copyright © Micah Fitch, Michael Sproul and contributors 2015-present.

[radix-wiki]: http://en.wikipedia.org/wiki/Radix_tree
[seq-trie]: https://github.com/michaelsproul/rust_sequence_trie
[radix-paper]: https://michaelsproul.github.io/rust_radix_paper/
[crates.io]: https://crates.io/
[radix-crate]: https://crates.io/crates/radix-immutable
