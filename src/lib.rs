use std::{collections::HashMap, hash::Hash};
use rand::{prelude::*, distributions::WeightedIndex};
#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;


/// A set of rules to apply when a token is matched, with a weight (`u8`) - the higher the weight compared
/// to the rest of the options, the more likely it will be selected (i.e., this set of production rules
/// satisfies a [stochastic grammar](https://en.wikipedia.org/wiki/L-system#Stochastic_grammars)).
/// `P` is the context used when generating a replacement string (`Vec<T>`).
#[derive(Default)]
pub struct LRulesSet<P, T> {
    options: Vec<(u8, Box<dyn Fn(&P) -> Vec<T>>)>,
}

impl<P, T> LRulesSet<P, T> {
    /// Create a new rules set.
    ///
    /// As an example taken from [Wikipedia](https://en.wikipedia.org/wiki/L-system#Example_1:_Algae):
    ///
    /// ```
    /// use anabaena::LRulesSet;
    ///
    /// // rule for the "A" character
    /// let algae_rules_a: LRulesSet<(), char> = LRulesSet::new(vec![
    ///     (1, Box::new(|_| vec!['A', 'B']))
    /// ]);
    ///
    /// // rule for the "B" character
    /// let algae_rules_b: LRulesSet<(), char> = LRulesSet::new(vec![
    ///     (1, Box::new(|_| vec!['A']))
    /// ]);
    /// ```
    ///
    /// Note that both of these rule sets have a unitary (`()`) context (no context is computed on each L-System
    /// iteration), and the alphabet type they correspond to is `char`.
    pub fn new(options: Vec<(u8, Box<dyn Fn(&P) -> Vec<T>>)>) -> Self {
        LRulesSet {
            options,
        }
    }

    /// Selects a result generator randomly, then applies the context.
    fn select(&self, context: &P, rng: &mut ThreadRng) -> Vec<T> {
        let mut weights: Vec<u8> = vec![];
        for (w,_) in self.options.iter() {
            weights.push(*w);
        }
        let dist = WeightedIndex::new(&weights).unwrap();
        (self.options[dist.sample(rng)].1)(&context)
    }
}

/// Qualifier for selecting a [LRulesSet](LRulesSet), where the neighbors of the token are known
/// as constants, with some direct adjacency from the character being matched. This enables "[Context-sensitive
/// grammars](https://en.wikipedia.org/wiki/L-system#Context_sensitive_grammars)"; where the context in question
/// simply matches the exact characters neighboring the selected character. The `usize` key in the
/// `HashMap` represents the "distance from the selected character".
///
/// In the Wikipedia article linked, the production rule `b < a > c -> aa` would have `b` and `c` matched with
/// this `ExactNeighbors`:
///
/// ```
/// use anabaena::ExactNeighbors;
///
/// let mut neighbors: ExactNeighbors<char> = ExactNeighbors::new();
/// neighbors.before.insert(0, 'b');
/// neighbors.after.insert(0, 'c');
/// ```
#[derive(Clone)]
pub struct ExactNeighbors<T> {
    /// Tokens to be matched before the selected character, with `usize` representing the number of
    /// characters before.
    pub before: HashMap<usize, T>,
    /// Tokens to be matched after the selected character, with `usize` representing the number of
    /// characters after.
    pub after: HashMap<usize, T>,
}

impl<T> Default for ExactNeighbors<T> {
    fn default() -> Self {
        Self {
            before: HashMap::new(),
            after: HashMap::new(),
        }
    }
}

impl<T> ExactNeighbors<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T> ExactNeighbors<T> where T: PartialEq {
    fn is_valid(&self, prefix: &[T], suffix: &[T]) -> bool {
        let prefix_last: usize = if prefix.is_empty() {
            0
        } else {
            prefix.len() - 1
        };
        self.before.iter().all(|(position, value)| prefix.get(prefix_last - *position) == Some(value))
            && self.after.iter().all(|(position, value)| suffix.get(*position) == Some(value))
    }
}

/// Qualifier for selecting a `LRulesSet`, where neighbors of the token must pass a Boolean predicate. Note
/// that there is no modification of the input string in the predicate - the prefix and suffix are applied
/// to the predicates _as-is_:
///
/// ```
/// use anabaena::PredicateNeighbors;
///
/// let mut foo_preds: PredicateNeighbors<char> = PredicateNeighbors::new();
/// foo_preds.before = Box::new(|prefix| {
///     println!("before: {:?}", prefix);
///     true
/// });
/// foo_preds.after = Box::new(|suffix| {
///     println!("after: {:?}", suffix);
///     true
/// });
/// ```
///
/// Applied:
///
/// ```ignore
/// input: ['a','b','c','d','e']
/// selected token: 'c'
/// stdout:
///     before: ['a','b']
///     after: ['d','e']
/// ```
pub struct PredicateNeighbors<T> {
    pub before: Box<dyn Fn(&[T]) -> bool>,
    pub after: Box<dyn Fn(&[T]) -> bool>,
}

impl<T> Default for PredicateNeighbors<T> {
    fn default() -> Self {
        Self {
            before: Box::new(|_| true),
            after: Box::new(|_| true),
        }
    }
}

impl<T> PredicateNeighbors<T> {
    pub fn new() -> Self {
        Self::default()
    }

    fn is_valid(&self, prefix: &[T], suffix: &[T]) -> bool {
        (self.before)(prefix) && (self.after)(suffix)
    }
}

/// The complete set of production rules for a selected character, with gradually relaxing constraints.
/// Note that the _first_ stochastic production rule set is returned when qualifying a selected token - as
/// an example, take the following set of rules around a selected token `'s'`:
///
/// ```
/// use anabaena::{ExactNeighbors, LRulesSet, LRulesQualified};
///
/// let mut rules: LRulesQualified<(), char> = LRulesQualified::new();
/// rules.exact_matches = vec![
///     {
///         let mut neighbors = ExactNeighbors::new();
///         neighbors.before.insert(0, 'a'); // an 'a' character directly preceeds the selector
///         (neighbors, LRulesSet::new(vec![(1, Box::new(|_| vec!['a', 'a', 's']))]))
///     },
///     {
///         let mut neighbors = ExactNeighbors::new();
///         neighbors.after.insert(0, 'b'); // a 'b' character directly follows the selector
///         (neighbors, LRulesSet::new(vec![(1, Box::new(|_| vec!['s', 'b', 'b']))]))
///     }
/// ];
/// ```
///
/// This set of production rules then gets applied to the string `['a','s','b']`. The result is
/// `['a','a','s','b']`, because only the first qualifier in `exact_matches` was picked. This further
/// expands to the other qualifiers `with_predicate` and `no_context` - the intent is to apply production
/// rules with the order of "most specific" to "least specific", as [the Wikipedia article on context
/// sensitive grammars details](https://en.wikipedia.org/wiki/L-system#Context_sensitive_grammars).
pub struct LRulesQualified<P, T> {
    /// Matched first, as exact token matches with position relative to a selected token are most specific
    pub exact_matches: Vec<(ExactNeighbors<T>, LRulesSet<P, T>)>,
    /// Matched second
    pub with_predicate: Vec<(PredicateNeighbors<T>, LRulesSet<P, T>)>,
    /// Matched last
    pub no_context: Option<LRulesSet<P, T>>,
}

impl<P, T> Default for LRulesQualified<P, T> {
    fn default() -> Self {
        Self {
            exact_matches: vec![],
            with_predicate: vec![],
            no_context: None,
        }
    }
}

impl<P, T> LRulesQualified<P, T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<P, T> LRulesQualified<P, T> where T: PartialEq {
    /// Gets the first production rule set given the context around a selected token
    fn get_rules_set(&self, prefix: &[T], suffix: &[T]) -> Option<&LRulesSet<P, T>> {
        for (e, r) in self.exact_matches.iter() {
            if e.is_valid(prefix, suffix) {
                return Some(&r)
            }
        }

        for (p, r) in self.with_predicate.iter() {
            if p.is_valid(prefix, suffix) {
                return Some(&r)
            }
        }

        self.no_context.as_ref()
    }
}


/// Trait that allows different implementations of matching a specific token to their production rules
pub trait ProductionRules<P, T> {
    fn apply(&self, context: &P, token: T, index: usize, string: &[T], rng: &mut ThreadRng) -> Option<Vec<T>>;
}


/// Exact match of tokens to a set of production rules via a HashMap. As an example taken from the
/// [Wikipedia article section demonstrating algae](https://en.wikipedia.org/wiki/L-system#Example_1:_Algae):
///
/// ```
/// use anabaena::{LRulesHash, LRulesQualified, LRulesSet};
///
/// let rules: LRulesHash<(), char> = LRulesHash::new(vec![
///     (
///         'A',
///         LRulesQualified {
///            no_context: Some(LRulesSet::new(vec![
///                (1, Box::new(|_| vec!['A', 'B'])),
///            ])),
///            ..LRulesQualified::default()
///         }
///     ),
///     (
///         'B',
///         LRulesQualified {
///            no_context: Some(LRulesSet::new(vec![
///                (1, Box::new(|_| vec!['A'])),
///            ])),
///            ..LRulesQualified::default()
///         }
///     )
/// ]);
/// ```
// FIXME make FromIterator?
pub struct LRulesHash<P, T>(HashMap<T, LRulesQualified<P, T>>);

impl<P, T> LRulesHash<P, T> where T: Eq + Hash {
    pub fn new(xs: Vec<(T, LRulesQualified<P, T>)>) -> Self {
        Self(xs.into_iter().collect())
    }
}

impl<P, T> ProductionRules<P, T> for LRulesHash<P, T> where T: Hash + Eq + Clone {
    fn apply(&self, context: &P, c: T, i: usize, string: &[T], rng: &mut ThreadRng) -> Option<Vec<T>> {
        self.0.get(&c).and_then(|rules| {
            let (prefix, suffix) = string.split_at(i);
            // drop token that matched `c`
            let suffix = if suffix.is_empty() {
                suffix
            } else {
                let (_, suffix) = suffix.split_at(1);
                suffix
            };
            rules.get_rules_set(prefix, suffix).map(|rules| rules.select(&context, rng))
        })
    }
}

/// Consists of the current string of tokens, and the rules applied to it.
/// Represents a running state, stepped as an Iterator. Taking the example from
/// [Wikipedia on algae](https://en.wikipedia.org/wiki/L-system#Example_1:_Algae):
///
/// ```
/// use anabaena::{LSystem, LRulesHash, LRulesQualified, LRulesSet};
///
/// let rules: LRulesHash<(), char> = LRulesHash::new(vec![
///     (
///         'A',
///         LRulesQualified {
///            no_context: Some(LRulesSet::new(vec![
///                (1, Box::new(|_| vec!['A', 'B'])),
///            ])),
///            ..LRulesQualified::default()
///         }
///     ),
///     (
///         'B',
///         LRulesQualified {
///            no_context: Some(LRulesSet::new(vec![
///                (1, Box::new(|_| vec!['A'])),
///            ])),
///            ..LRulesQualified::default()
///         }
///     )
/// ]);
///
/// let axiom: Vec<char> = vec!['A'];
///
/// let mut lsystem = LSystem {
///     string: axiom,
///     rules,
///     context: (),
///     mk_context: Box::new(|_, _| ()),
/// };
///
/// assert_eq!(lsystem.next(), Some("AB".chars().collect()));
/// assert_eq!(lsystem.next(), Some("ABA".chars().collect()));
/// assert_eq!(lsystem.next(), Some("ABAAB".chars().collect()));
/// assert_eq!(lsystem.next(), Some("ABAABABA".chars().collect()));
/// assert_eq!(lsystem.next(), Some("ABAABABAABAAB".chars().collect()));
/// assert_eq!(lsystem.next(), Some("ABAABABAABAABABAABABA".chars().collect()));
/// assert_eq!(lsystem.next(), Some("ABAABABAABAABABAABABAABAABABAABAAB".chars().collect()));
/// ```
pub struct LSystem<R, P, T> {
    pub string: Vec<T>,
    pub rules: R,
    pub context: P,
    pub mk_context: Box<dyn Fn(&P, &[T]) -> P>,
}


impl<R, P, T> Iterator for LSystem<R, P, T> where T: PartialEq + Clone, R: ProductionRules<P, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut rng = thread_rng();
        let mut applied = false;

        let new_context: P = (self.mk_context)(&self.context, &self.string);
        let new_string: Vec<T> = self.string.clone().into_iter().enumerate().map(|(i,c)| {
            match self.rules.apply(&new_context, c.clone(), i, &self.string, &mut rng) {
                None => vec![c],
                Some(replacement) => {
                    applied = true;
                    replacement
                }
            }
        }).flatten().collect();

        if applied {
            self.string = new_string.clone();
            self.context = new_context;
            Some(new_string)
        } else {
            None
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_set_selects() {
        let rules: LRulesSet<(), char> = LRulesSet::new(vec![
            (1, Box::new(|_| vec!['a'])),
            (1, Box::new(|_| vec!['b'])),
            (1, Box::new(|_| vec!['c']))
        ]);
        let mut rng = thread_rng();
        for _ in 0..10 {
            let result = rules.select(&(), &mut rng);
            assert!(
                result == ['a'] || result == ['b'] || result == ['c'],
                "Result isn't a, b, or c: {:?}",
                result
            );
        }
    }

    #[quickcheck]
    fn all_neighbors_are_valid(xs: Vec<char>) -> bool {
        if xs.len() == 0 {
            return true;
        }
        let mut rng = thread_rng();
        let i = rng.gen::<usize>() % xs.len();
        let prefix: &[char] = &xs[0..i];
        let suffix: &[char] = if i == xs.len() - 1 {
            &[]
        } else {
            &xs[i+1..xs.len()]
        };
        let exact_neighbors: ExactNeighbors<char> = ExactNeighbors {
            before: {
                let mut ys = HashMap::new();
                for (j,y) in prefix.iter().enumerate() {
                    ys.insert((prefix.len() - 1) - j, *y);
                }
                ys
            },
            after: {
                let mut ys = HashMap::new();
                for (j,y) in suffix.iter().enumerate() {
                    ys.insert(j, *y);
                }
                ys
            },
        };
        exact_neighbors.is_valid(prefix, suffix)
    }
}
