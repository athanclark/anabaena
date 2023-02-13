Anabaena
==========

L-System (Lindenmayer system) for Rust.

## Features

- Generic L-System alphabet
- Support for [stochastic grammars](https://en.wikipedia.org/wiki/L-system#Stochastic_grammars)
- Support for [context sensitive grammars](https://en.wikipedia.org/wiki/L-system#Context_sensitive_grammars)
- Support for [parametric grammars](https://en.wikipedia.org/wiki/L-system#Parametric_grammars)
- Mutable context support during individual production rules, and summarized after a round is complete
- Streaming iterator implementation

## Examples

### Algae

Taken from the [Wikipedia article on L-Systems](https://en.wikipedia.org/wiki/L-system#Example_1:_Algae),
the following is an implementation of the algae example:

```rust
use anabaena::{LSystem, LRulesHash, LRulesSet, Total};
use std::collections::HashMap;
use streaming_iterator::StreamingIterator;

let rules: LRulesHash<(), char, LRulesSet<char>> = |_| HashMap::from([
    (
        'A',
        LRulesSet::new(vec![
            (1, vec!['A', 'B']),
        ]),
    ),
    (
        'B',
        LRulesSet::new(vec![
            (1, vec!['A']),
        ]),
    )
]);

let axiom: Vec<char> = vec!['A'];

let mut lsystem: LSystem<_,_,_, Total> = LSystem::new(
    axiom,
    rules,
);

assert_eq!(lsystem.next(), Some(&"AB".chars().collect()));
assert_eq!(lsystem.next(), Some(&"ABA".chars().collect()));
assert_eq!(lsystem.next(), Some(&"ABAAB".chars().collect()));
assert_eq!(lsystem.next(), Some(&"ABAABABA".chars().collect()));
assert_eq!(lsystem.next(), Some(&"ABAABABAABAAB".chars().collect()));
assert_eq!(lsystem.next(), Some(&"ABAABABAABAABABAABABA".chars().collect()));
assert_eq!(lsystem.next(), Some(&"ABAABABAABAABABAABABAABAABABAABAAB".chars().collect()));
```

### Examples with Turtle.rs

Most of the other examples in the Wikipedia article are implemented in the `examples/` folder, which
use the [turtle.rs library](https://turtle.rs/) to render the results. You can run the B-Tree example,
for instance, with the following command:

```bash
cargo run --example btree
```
