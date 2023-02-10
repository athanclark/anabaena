use rand::{distributions::WeightedIndex, prelude::*};
use std::{
    collections::{BTreeSet, HashMap},
    hash::Hash,
};
use streaming_iterator::StreamingIterator;
#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

/// A set of rules to apply when a token is matched, with a weight (`u8`) - the higher the weight compared
/// to the rest of the options, the more likely it will be selected (i.e., this set of production rules
/// satisfies a [stochastic grammar](https://en.wikipedia.org/wiki/L-system#Stochastic_grammars)).
///
/// As an example taken from [Wikipedia](https://en.wikipedia.org/wiki/L-system#Example_1:_Algae):
///
/// ```
/// use anabaena::LRulesSet;
///
/// // rule for the "A" character
/// let algae_rules_a: LRulesSet<char> = LRulesSet::new(vec![
///     (1, vec!['A', 'B'])
/// ]);
///
/// // rule for the "B" character
/// let algae_rules_b: LRulesSet<char> = LRulesSet::new(vec![
///     (1, vec!['A'])
/// ]);
/// ```
pub struct LRulesSet<T>(Vec<(u8, Vec<T>)>);

impl<T> LRulesSet<T> {
    pub fn new(options: Vec<(u8, Vec<T>)>) -> Self {
        Self(options)
    }

    /// Selects a result generator randomly, then applies the context.
    fn select<'a, 'b>(&'a self, rng: &'b mut ThreadRng) -> &'a [T] {
        let mut weights: Vec<u8> = vec![];
        for (w, _) in self.0.iter() {
            weights.push(*w);
        }
        let dist = WeightedIndex::new(&weights).unwrap();
        &self.0[dist.sample(rng)].1
    }
}

/// Qualifier for selecting a [LRulesSet](LRulesSet), where the neighbors of the token are known
/// as constants, with some direct adjacency from the character being matched. This enables "[Context-sensitive
/// grammars](https://en.wikipedia.org/wiki/L-system#Context_sensitive_grammars)"; where the context in question
/// simply matches the exact characters neighboring the selected character.
///
/// The `usize` key in the `HashMap` represents the "distance from the selected character".
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

impl<T> ExactNeighbors<T>
where
    T: PartialEq,
{
    fn is_valid(&self, prefix: &[T], suffix: &[T]) -> bool {
        let prefix_last: usize = if prefix.is_empty() {
            0
        } else {
            prefix.len() - 1
        };
        self.before
            .iter()
            .all(|(position, value)| prefix.get(prefix_last - *position) == Some(value))
            && self
                .after
                .iter()
                .all(|(position, value)| suffix.get(*position) == Some(value))
    }
}

/// Qualifier for selecting a `LRulesSet`, where neighbors of the token must pass a Boolean predicate.
/// This enables "[Context-sensitive
/// grammars](https://en.wikipedia.org/wiki/L-system#Context_sensitive_grammars)"; where the context in question
/// is more flexible than the [ExactNeighbors](ExactNeighbors) version, and simply has to pass a predicate.
///
/// Note that there is no modification of the input string in the predicate - the prefix and suffix are applied
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
    pub before: Box<Predicate<T>>,
    pub after: Box<Predicate<T>>,
}

/// Simple type that qualifies either a prefix or suffix of tokens `T`.
pub type Predicate<T> = fn(&[T]) -> bool;

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
///
/// Note that the _first_ stochastic production rule set is returned when qualifying a selected token - as
/// an example, take the following set of rules around a selected token `'s'`:
///
/// ```
/// use anabaena::{ExactNeighbors, LRulesSet, LRulesQualified};
///
/// let mut rules: LRulesQualified<char> = LRulesQualified::new();
/// rules.exact_matches = vec![
///     {
///         let mut neighbors = ExactNeighbors::new();
///         neighbors.before.insert(0, 'a'); // an 'a' character directly preceeds the selector
///         (neighbors, LRulesSet::new(vec![(1, vec!['a', 'a', 's'])]))
///     },
///     {
///         let mut neighbors = ExactNeighbors::new();
///         neighbors.after.insert(0, 'b'); // a 'b' character directly follows the selector
///         (neighbors, LRulesSet::new(vec![(1, vec!['s', 'b', 'b'])]))
///     }
/// ];
/// ```
///
/// This set of production rules then gets applied to the string `['a','s','b']`. The result is
/// `['a','a','s','b']`, because only the first qualifier in `exact_matches` was picked. This further
/// expands to the other qualifiers `with_predicate` and `no_context` - the intent is to apply production
/// rules with the order of "most specific" to "least specific", as [the Wikipedia article on context
/// sensitive grammars details](https://en.wikipedia.org/wiki/L-system#Context_sensitive_grammars).
pub struct LRulesQualified<T> {
    /// Matched first, as exact token matches with position relative to a selected token are most specific
    pub exact_matches: Vec<(ExactNeighbors<T>, LRulesSet<T>)>,
    /// Matched second
    pub with_predicate: Vec<(PredicateNeighbors<T>, LRulesSet<T>)>,
    /// Matched last
    pub no_context: Option<LRulesSet<T>>,
}

impl<T> Default for LRulesQualified<T> {
    fn default() -> Self {
        Self {
            exact_matches: vec![],
            with_predicate: vec![],
            no_context: None,
        }
    }
}

impl<T> LRulesQualified<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T> LRulesQualified<T>
where
    T: PartialEq,
{
    /// Gets the first production rule set given the context around a selected token
    fn get_rules_set(&self, prefix: &[T], suffix: &[T]) -> Option<&LRulesSet<T>> {
        for (e, r) in self.exact_matches.iter() {
            if e.is_valid(prefix, suffix) {
                return Some(r);
            }
        }

        for (p, r) in self.with_predicate.iter() {
            if p.is_valid(prefix, suffix) {
                return Some(r);
            }
        }

        self.no_context.as_ref()
    }
}

pub trait ProductionRules<P, T> {
    fn apply(
        &mut self,
        context: &mut P,
        token: &T,
        index: usize,
        rng: &mut ThreadRng,
    ) -> Option<Vec<T>>;
}

/// Trait that allows different implementations of matching a specific token to their production rules
pub trait QualifiedProductionRules<P, T> {
    fn apply_qualified(
        &mut self,
        context: &mut P,
        token: &T,
        index: usize,
        string: &[T],
        rng: &mut ThreadRng,
    ) -> Option<Vec<T>>;
}

/// Exact match of tokens to a set of production rules via a HashMap.
///
/// As an example taken from the
/// [Wikipedia article section demonstrating algae](https://en.wikipedia.org/wiki/L-system#Example_1:_Algae):
///
/// ```
/// use anabaena::{LRulesHash, LRulesQualified, LRulesSet};
/// use std::collections::HashMap;
///
/// let rules: LRulesHash<(), char, LRulesQualified<char>> = |_| HashMap::from([
///     (
///         'A',
///         LRulesQualified {
///            no_context: Some(LRulesSet::new(vec![
///                (1, vec!['A', 'B']),
///            ])),
///            ..LRulesQualified::default()
///         }
///     ),
///     (
///         'B',
///         LRulesQualified {
///            no_context: Some(LRulesSet::new(vec![
///                (1, vec!['A']),
///            ])),
///            ..LRulesQualified::default()
///         }
///     )
/// ]);
/// ```
pub type LRulesHash<P, T, Q> = fn(&mut P) -> HashMap<T, Q>;

impl<P, T> ProductionRules<P, T> for LRulesHash<P, T, LRulesSet<T>>
where
    T: Hash + Eq + Clone,
{
    fn apply(&mut self, context: &mut P, c: &T, _i: usize, rng: &mut ThreadRng) -> Option<Vec<T>> {
        self(context).get(c).map(|rules| rules.select(rng).to_vec())
    }
}

impl<P, T> QualifiedProductionRules<P, T> for LRulesHash<P, T, LRulesQualified<T>>
where
    T: Hash + Eq + Clone,
{
    fn apply_qualified<'a>(
        &mut self,
        context: &mut P,
        c: &T,
        i: usize,
        string: &[T],
        rng: &mut ThreadRng,
    ) -> Option<Vec<T>> {
        self(context).get(c).and_then(|rules| {
            let (prefix, suffix) = string.split_at(i);
            // drop token that matched `c`
            let suffix = if suffix.is_empty() {
                suffix
            } else {
                let (_, suffix) = suffix.split_at(1);
                suffix
            };
            rules
                .get_rules_set(prefix, suffix)
                .map(|rules| rules.select(rng).to_vec())
        })
    }
}

/// Exact match of tokens to a set of production rules via a function.
///
/// As an example taken from the
/// [Wikipedia article section demonstrating algae](https://en.wikipedia.org/wiki/L-system#Example_1:_Algae):
///
/// ```
/// use anabaena::{LRulesFunction, LRulesQualified, LRulesSet};
///
/// let rules: LRulesFunction<(), char, LRulesQualified<char>> = |_, c| {
///     match c {
///         'A' =>
///             Some(LRulesQualified {
///                no_context: Some(LRulesSet::new(vec![
///                    (1, vec!['A', 'B']),
///                ])),
///                ..LRulesQualified::default()
///             }),
///         'B' =>
///             Some(LRulesQualified {
///                no_context: Some(LRulesSet::new(vec![
///                    (1, vec!['A']),
///                ])),
///                ..LRulesQualified::default()
///             }),
///         _ => None,
///     }
/// };
/// ```
pub type LRulesFunction<P, T, Q> = fn(&mut P, &T) -> Option<Q>;

impl<P, T> ProductionRules<P, T> for LRulesFunction<P, T, LRulesSet<T>>
where
    T: Clone,
{
    fn apply(&mut self, context: &mut P, c: &T, _i: usize, rng: &mut ThreadRng) -> Option<Vec<T>> {
        self(context, c).map(|rules| rules.select(rng).to_vec())
    }
}

impl<P, T> QualifiedProductionRules<P, T> for LRulesFunction<P, T, LRulesQualified<T>>
where
    T: PartialEq + Clone,
{
    fn apply_qualified(
        &mut self,
        context: &mut P,
        c: &T,
        i: usize,
        string: &[T],
        rng: &mut ThreadRng,
    ) -> Option<Vec<T>> {
        self(context, c).and_then(|rules| {
            let (prefix, suffix) = string.split_at(i);
            // drop token that matched `c`
            let suffix = if suffix.is_empty() {
                suffix
            } else {
                let (_, suffix) = suffix.split_at(1);
                suffix
            };
            rules
                .get_rules_set(prefix, suffix)
                .map(|rules| rules.select(rng).to_vec())
        })
    }
}

/// Exact match of tokens to a set of production rules via a function, also including the token's index
/// in the string being processed. Note that this is the index of the token during the current iteration
/// of production rules, and the indicies won't update until the next iteration of the L-System.
pub type LRulesFunctionWithIndex<P, T, Q> = fn(&mut P, &T, usize) -> Option<Q>;

impl<P, T> ProductionRules<P, T> for LRulesFunctionWithIndex<P, T, LRulesSet<T>>
where
    T: Clone,
{
    fn apply(&mut self, context: &mut P, c: &T, i: usize, rng: &mut ThreadRng) -> Option<Vec<T>> {
        self(context, c, i).map(|rules| rules.select(rng).to_vec())
    }
}

impl<P, T> QualifiedProductionRules<P, T> for LRulesFunctionWithIndex<P, T, LRulesQualified<T>>
where
    T: PartialEq + Clone,
{
    fn apply_qualified(
        &mut self,
        context: &mut P,
        c: &T,
        i: usize,
        string: &[T],
        rng: &mut ThreadRng,
    ) -> Option<Vec<T>> {
        self(context, c, i).and_then(|rules| {
            let (prefix, suffix) = string.split_at(i);
            // drop token that matched `c`
            let suffix = if suffix.is_empty() {
                suffix
            } else {
                let (_, suffix) = suffix.split_at(1);
                suffix
            };
            rules
                .get_rules_set(prefix, suffix)
                .map(|rules| rules.select(rng).to_vec())
        })
    }
}

/// Consists of the current string of tokens, and the rules applied to it.
/// Represents a running state, stepped as an Iterator.
///
/// Note that the L-System is generic in the alphabet type `T` - this means we can define any
/// Enum to treat as tokens, and even enrich them with parametric information as described in
/// [the Wikipedia article on Parametric grammars](https://en.wikipedia.org/wiki/L-system#Parametric_grammars).
///
/// Furthermore, each iteration of the L-System can develop and share a "contextual state" `P`, to be used
/// however you'd like when generating more tokens. It's a stateful value, and can be used to track
/// concepts like age of the L-System, or pre-process the previous iteration's tokens to make production
/// rules more efficient.
///
/// Lastly, the L-System is also parametric in its production rule set implementation `R` - as long as the
/// production rule system suits the [ProductionRules](ProductionRules) trait, then the L-System can iterate.
/// We provide two simple implementations: [LRulesHash](LRulesHash), which indexes the production rule sets
/// by each matching selector token `T` (implying `T` must suit `Hash` and `Eq`), and
/// [LRulesFunction](LRulesFunction), which is a simple function where the author is expected to provide their
/// own matching functionality.
///
/// Taking the example from
/// [Wikipedia on algae](https://en.wikipedia.org/wiki/L-system#Example_1:_Algae):
///
/// ```
/// use anabaena::{LSystem, LRulesHash, LRulesQualified, LRulesSet};
/// use std::collections::HashMap;
///
/// let rules: LRulesHash<(), char, LRulesQualified<char>> = |_| HashMap::from([
///     (
///         'A',
///         LRulesQualified {
///            no_context: Some(LRulesSet::new(vec![
///                (1, vec!['A', 'B']),
///            ])),
///            ..LRulesQualified::default()
///         }
///     ),
///     (
///         'B',
///         LRulesQualified {
///            no_context: Some(LRulesSet::new(vec![
///                (1, vec!['A']),
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
///     mut_context: |_, _| {},
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
    /// The current token string - set this to an initial value (i.e., the _axiom_) when creating your L-System
    pub string: Vec<T>,
    /// The production rules applied to the string
    pub rules: R,
    /// The mutable context used throughout the production rules, and lastly in-batch with `mut_context`
    pub context: P,
    /// Performed _after_ all production rules have been applied
    pub mut_context: MutContext<P, T>,
}

/// Used to generate the next contextual type `P` from the previous context, and previous string
/// of tokens `&[T]`.
pub type MutContext<P, T> = fn(&mut P, &[T]);

// FIXME increase efficiency of this design, and implement streaming iterator rather than iterator!

impl<R, P, T> Iterator for LSystem<R, P, T>
where
    T: PartialEq + Clone,
    R: QualifiedProductionRules<P, T>,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut rng = thread_rng();
        let mut applied = false;

        let new_string: Vec<T> = self
            .string
            .clone()
            .into_iter()
            .enumerate()
            .flat_map(|(i, c)| {
                match self
                    .rules
                    .apply_qualified(&mut self.context, &c, i, &self.string, &mut rng)
                {
                    None => vec![c],
                    Some(replacement) => {
                        applied = true;
                        replacement.to_vec()
                    }
                }
            })
            .collect();

        if applied {
            self.string = new_string.clone();
            (self.mut_context)(&mut self.context, &self.string);
            Some(new_string)
        } else {
            None
        }
    }
}

/// Similarly to the [LSystem](LSystem), this operates as an iterator, however, the exact indicies in which
/// the production rules are to be performed are derived from the context.
pub struct LSystemSelective<R, P, T> {
    /// The current token string - set this to an initial value (i.e., the _axiom_) when creating your L-System
    pub string: Vec<T>,
    /// The production rules applied to the string
    pub rules: R,
    /// The mutable context used throughout the production rules, and lastly in-batch with `mut_context`
    pub context: P,
    /// Performed _after_ all production rules have been applied
    pub mut_context: MutContext<P, T>,
    /// Fetch the indicies to perform the production rules
    pub get_indicies: GetIndicies<P>,
    /// Update the indicies stored in `P` to a new offset (a previous index was replaced by either a
    /// production greater than length 1, or 0). This is run on the result of every production rule.
    pub update_indicies: UpdateIndicies<P>,
    /// Has been applied
    pub applied: bool,
}

/// Used to generate the next contextual type `P` from the previous context, and previous string
/// of tokens `&[T]`.
pub type GetIndicies<P> = fn(&P) -> BTreeSet<usize>;

/// Update `P` to reflect the change in indicies by some constant offset. The order of arguments is:
///
/// 1. Mutable context `P`
/// 2. Indicies greater than or equal to this index are affected
/// 3. Offset to apply
///
/// Furthermore, an offset value of `0` indicates
/// that the indicies should be reduced by `1`. All other values indicate the offset should be increased by
/// that very value - i.e. a offset of `1` indicates the indicies should be increased by `1`.
/// It is also expected that any modifications to the context during the production rule respect the
/// new indicies without having to be compensated for - this function should only affect tokens
/// whose indicies are _after_ the last element of the vector produced by the production rule.
pub type UpdateIndicies<P> = fn(&mut P, usize, usize);

impl<R, P, T> LSystemSelective<R, P, T> {
    fn get_(&self) -> Option<&Vec<T>> {
        if self.applied {
            Some(&self.string)
        } else {
            None
        }
    }
}

impl<R, P, T> LSystemSelective<R, P, T>
    where
    R: QualifiedProductionRules<P, T>,
    T: Clone,
{
    fn advance_qualified(&mut self) {
        let mut rng = thread_rng();

        let indicies: BTreeSet<usize> = (self.get_indicies)(&self.context);

        self.applied = false;

        let mut cumulative_offset: i64 = 0;

        let old_string = self.string.clone();

        for i in indicies {
            let mut orig = self
                .string
                .split_off((i as i64 + cumulative_offset) as usize);
            let mut tail = orig.split_off(1);
            if let Some(mut replacement) =
                self.rules
                    .apply_qualified(&mut self.context, &orig[0], i, &old_string, &mut rng)
            {
                self.applied = true;
                let l = replacement.len();
                cumulative_offset += l as i64 - 1;
                replacement.append(&mut tail);
                self.string.append(&mut replacement);
                (self.update_indicies)(&mut self.context, i + 1, l);
            } else {
                orig.append(&mut tail);
                self.string.append(&mut orig);
            }
        }

        if self.applied {
            (self.mut_context)(&mut self.context, &self.string);
        }
    }
}

impl<R, P, T> LSystemSelective<R, P, T>
    where
    R: ProductionRules<P, T>,
    T: Clone,
{
    fn advance_context_free(&mut self) {
        let mut rng = thread_rng();

        let indicies: BTreeSet<usize> = (self.get_indicies)(&self.context);

        self.applied = false;

        let mut cumulative_offset: i64 = 0;

        for i in indicies {
            let mut orig = self
                .string
                .split_off((i as i64 + cumulative_offset) as usize);
            let mut tail = orig.split_off(1);
            if let Some(mut replacement) =
                self.rules
                    .apply(&mut self.context, &orig[0], i, &mut rng)
            {
                self.applied = true;
                let l = replacement.len();
                cumulative_offset += l as i64 - 1;
                replacement.append(&mut tail);
                self.string.append(&mut replacement);
                (self.update_indicies)(&mut self.context, i + 1, l);
            } else {
                orig.append(&mut tail);
                self.string.append(&mut orig);
            }
        }

        if self.applied {
            (self.mut_context)(&mut self.context, &self.string);
        }
    }
}

impl<P, T> StreamingIterator for LSystemSelective<LRulesHash<P, T, LRulesQualified<T>>, P, T>
where
    T: Hash + Eq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_qualified()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystemSelective<LRulesFunction<P, T, LRulesQualified<T>>, P, T>
where
    T: PartialEq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_qualified()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystemSelective<LRulesFunctionWithIndex<P, T, LRulesQualified<T>>, P, T>
where
    T: PartialEq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_qualified()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystemSelective<LRulesHash<P, T, LRulesSet<T>>, P, T>
where
    T: Hash + Eq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystemSelective<LRulesFunction<P, T, LRulesSet<T>>, P, T>
where
    T: Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystemSelective<LRulesFunctionWithIndex<P, T, LRulesSet<T>>, P, T>
where
    T: Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

/// Version of an L-System in which unchanged tokens aren't kept - only newly generated tokens
/// are iterated. This implies a few things:
///
/// - order of tokens doesn't matter in your L-System (they should have some kind of encoded identifier if
///   being post-processed)
/// - inactive indicies are thrown away - all indicies are assumed to be active, because the last iteration
///   caused a successful production rule
/// - any tokens that don't generate a production are thrown away
/// - these L-Systems only make sense in a context-free scenario
///
/// This is useful in a situation where the tokens being generated can be processed and interpreted in a
/// similarly context-free fashion in order to be useful, while also avoiding a combinatorial explosion
/// s.t. all memory is consumed while generating a token string - in this circumstance, you'd process
/// and interpret the newly generated tokens on each iteration of the StreamingIterator.
pub struct LSystemDelta<R, P, T> {
    /// The current token string - set this to an initial value (i.e., the _axiom_) when creating your L-System
    pub string: Vec<T>,
    /// The production rules applied to the string
    pub rules: R,
    /// The mutable context used throughout the production rules, and lastly in-batch with `mut_context`
    pub context: P,
    /// Performed _after_ all production rules have been applied
    pub mut_context: MutContext<P, T>,
    /// Has been applied
    pub applied: bool,
}

impl<R, P, T> LSystemDelta<R, P, T> {
    fn get_(&self) -> Option<&Vec<T>> {
        if self.applied {
            Some(&self.string)
        } else {
            None
        }
    }
}

impl<R, P, T> LSystemDelta<R, P, T>
    where
    R: ProductionRules<P, T>,
    T: Clone,
{
    fn advance_context_free(&mut self) {
        let mut rng = thread_rng();

        self.applied = false;

        let mut new_string = vec![];

        for (i, c) in self.string.iter().enumerate() {
            if let Some(mut replacement) =
                self.rules.apply(&mut self.context, c, i, &mut rng)
            {
                self.applied = true;
                new_string.append(&mut replacement);
            }
        }

        self.string = new_string;

        if self.applied {
            (self.mut_context)(&mut self.context, &self.string);
        }
    }
}

impl<P, T> StreamingIterator for LSystemDelta<LRulesHash<P, T, LRulesSet<T>>, P, T>
where
    T: Hash + Eq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystemDelta<LRulesFunction<P, T, LRulesSet<T>>, P, T>
where
    T: Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystemDelta<LRulesFunctionWithIndex<P, T, LRulesSet<T>>, P, T>
where
    T: Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use streaming_iterator::StreamingIterator;

    #[test]
    fn rules_set_selects() {
        let rules: LRulesSet<char> = LRulesSet::new(vec![(1, vec!['a']), (1, vec!['b']), (1, vec!['c'])]);
        let mut rng = thread_rng();
        for _ in 0..10 {
            let result = rules.select(&mut rng).clone();
            assert!(
                result == vec!['a'] || result == vec!['b'] || result == vec!['c'],
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
            &xs[i + 1..xs.len()]
        };
        let exact_neighbors: ExactNeighbors<char> = ExactNeighbors {
            before: {
                let mut ys = HashMap::new();
                for (j, y) in prefix.iter().enumerate() {
                    ys.insert((prefix.len() - 1) - j, *y);
                }
                ys
            },
            after: {
                let mut ys = HashMap::new();
                for (j, y) in suffix.iter().enumerate() {
                    ys.insert(j, *y);
                }
                ys
            },
        };
        exact_neighbors.is_valid(prefix, suffix)
    }

    #[test]
    fn selective_has_same_results_as_normal_forall_indicies() {
        let axiom: Vec<char> = vec!['A'];

        let mut lsystem: LSystem<LRulesHash<(), _, _>, (), char> = LSystem {
            string: axiom.clone(),
            rules: |_| {
                HashMap::from([
                    (
                        'A',
                        LRulesQualified {
                            no_context: Some(LRulesSet::new(vec![(1, vec!['A', 'B'])])),
                            ..LRulesQualified::default()
                        },
                    ),
                    (
                        'B',
                        LRulesQualified {
                            no_context: Some(LRulesSet::new(vec![(1, vec!['A'])])),
                            ..LRulesQualified::default()
                        },
                    ),
                ])
            },
            context: (),
            mut_context: |_, _| {},
        };

        let mut lsystem_selective: LSystemSelective<LRulesHash<usize, _, _>, usize, char> =
            LSystemSelective {
                string: axiom.clone(),
                rules: |_| {
                    HashMap::from([
                        (
                            'A',
                            LRulesQualified {
                                no_context: Some(LRulesSet::new(vec![(1, vec!['A', 'B'])])),
                                ..LRulesQualified::default()
                            },
                        ),
                        (
                            'B',
                            LRulesQualified {
                                no_context: Some(LRulesSet::new(vec![(1, vec!['A'])])),
                                ..LRulesQualified::default()
                            },
                        ),
                    ])
                },
                context: axiom.len(),
                mut_context: |ctx: &mut usize, ts: &[char]| {
                    *ctx = ts.len();
                },
                get_indicies: |ctx: &usize| (0..*ctx).collect(),
                update_indicies: |ctx: &mut usize, _i: usize, offset: usize| {
                    if offset == 0 {
                        *ctx -= 1;
                    } else {
                        *ctx += offset - 1;
                    }
                },
                applied: true,
            };

        for i in 1..20 {
            let x: Option<Vec<char>> = lsystem.next();
            let y: Option<Vec<char>> = lsystem_selective.next().cloned();
            assert_eq!(
                x,
                y,
                "LSystems aren't equal on iteration {i:}"
            );
            assert_eq!(
                lsystem_selective.context,
                lsystem_selective.string.len(),
                "context and string length aren't equal."
            );
        }

        assert!(true);
    }
}
