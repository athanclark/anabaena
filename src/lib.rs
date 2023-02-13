use rand::{distributions::WeightedIndex, prelude::*};
use std::{collections::HashMap, hash::Hash, marker::PhantomData};
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

/// Production rules that depend on surrounding characters when selecting
/// for a specific character, ordered by gradually relaxing constraints.
///
/// Note that the _first_ production rule set is returned when qualifying a selected token - as
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
/// Let's say this set of production rules then gets applied to the string `['a','s','b']`. The result is
/// `['a','a','s','b']`, because only the _first_ qualifier in `exact_matches` was picked.
///
/// This further
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

/// Trait for applying production rules bound by [LRulesSet](LRulesSet), generic in the mechanism
/// by which a selected character is matched.
pub trait ProductionRules<P, T> {
    fn apply(
        &mut self,
        context: &mut P,
        token: &T,
        index: usize,
        rng: &mut ThreadRng,
    ) -> Option<Vec<T>>;
}

/// Trait for applying production rules bound by [LRulesQualified](LRulesQualified), generic in the mechanism
/// by which a selected character is matched.
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

/// Implement this trait if you want to share an alternative mutable context throughout your production
/// rules.
pub trait ProductionRuleContext<T> {
    /// This function runs after all the production rules have been applied and integrated into the
    /// L-System's string state - the argument `tokens` represents the newly generated string.
    fn on_complete(&mut self, tokens: &[T]);
}

impl<T> ProductionRuleContext<T> for () {
    fn on_complete(&mut self, _: &[T]) {}
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
/// let rules: LRulesHash<(), char, LRulesSet<char>> = |_| HashMap::from([
///     (
///         'A',
///         LRulesSet::new(vec![
///             (1, vec!['A', 'B']),
///         ]),
///     ),
///     (
///         'B',
///         LRulesSet::new(vec![
///             (1, vec!['A']),
///         ]),
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
/// let rules: LRulesFunction<(), char, LRulesSet<char>> = |_, c| {
///     match c {
///         'A' =>
///             Some(LRulesSet::new(vec![
///                 (1, vec!['A', 'B']),
///             ])),
///         'B' =>
///             Some(LRulesSet::new(vec![
///                 (1, vec!['A']),
///             ])),
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
/// in the string being processed.
///
/// Note that this is the index of the token during the current iteration
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

/// Trait for qualifying phantom types for the L-System.
pub trait LSystemMode {}

/// Runs the production rules on every character in the string, every iteration.
pub struct Total;

/// Runs the production rules on only the indicies of the string returned by
/// [get_indicies()](IndexedContext::get_indicies).
pub struct Indexed;

/// Runs the production rules on only the characters which had values generated last iteration.
///
/// This has some strong implications about your L-System:
///
/// - order of tokens doesn't matter in your L-System (the alphabet
///   should have some kind of encoded identifier if being post-processed)
/// - inactive indicies are thrown away - all indicies are assumed to be active, because the last iteration
///   caused a successful production rule
/// - any tokens that don't generate a production are thrown away
/// - these L-Systems only make sense in a context-free scenario
///
/// This is useful in a situation where the tokens being generated can be processed and interpreted in a
/// similarly context-free fashion, while also avoiding a combinatorial explosion
/// s.t. all memory is consumed while generating a token string - in this circumstance, you'd process
/// and interpret the newly generated tokens on each iteration of the [StreamingIterator](StreamingIterator).
pub struct Unordered;

impl LSystemMode for Total {}
impl LSystemMode for Indexed {}
impl LSystemMode for Unordered {}

/// Trait that represents contexts which can independently track the indicies of the tokens valid for
/// generation in a particular L-System string.
pub trait IndexedContext<T>: ProductionRuleContext<T> {
    /// Apply production rules only to these indicies.
    ///
    /// Note - it's expected that the indicies are
    /// **ordered from low to high** - without that guarantee,
    /// [offset_indicies()](IndexedContext::offset_indicies)
    /// couldn't provide a reasonable value to your context.
    fn get_indicies(&self) -> Vec<usize>;

    /// Update `P` to reflect the change in indicies by some constant offset. The order of arguments is:
    ///
    /// Indicies greater than or equal to `starting_index` should only be affected. Furthermore, the `offset`
    /// has unique semantics: an offset value of `0` indicates
    /// that all indicies greater-than or equal to `starting_index` should be reduced by `1`.
    /// All other values indicate the offset should be increased by
    /// that very value - i.e. a offset of `1` indicates the indicies should be increased by `1`.
    fn offset_indicies(&mut self, starting_index: usize, offset: usize);
}

/// An L-System consists of the current string of tokens, and the rules applied to it.
///
/// Represents a running state, stepped as a [StreamingIterator](StreamingIterator).
///
/// Note that the L-System is generic in the alphabet type `T` - this means we can define any
/// Enum to treat as an alphabet for tokens, and even enrich them with parametric information as described in
/// [the Wikipedia article on Parametric grammars](https://en.wikipedia.org/wiki/L-system#Parametric_grammars).
///
/// Furthermore, each iteration of the L-System can develop and share a "contextual state" `P` to be used
/// however you'd like when generating more tokens. It's a mutable value, and can be used to track
/// concepts like age of the L-System, or pre-process the previous iteration's tokens to make production
/// rules more efficient.
///
/// The L-System is also parametric in its production rule set implementation `R` - as long as the
/// production rule system suits the [ProductionRules](ProductionRules) or
/// [QualifiedProductionRules](QualifiedProductionRules) traits, then the L-System can iterate.
///
/// We provide three simple implementations:
///
/// - [LRulesHash](LRulesHash), which indexes the production rule sets
///   by each matching selector token `T` (implying `T` must suit `Hash` and `Eq`)
/// - [LRulesFunction](LRulesFunction), which is a simple function where the author is expected to provide their
///   own matching functionality.
/// - [LRulesFunctionWithIndex](LRulesFunctionWithIndex), which is identical to [LRulesFunction](LRulesFunction),
///   but also includes the token's index in the string.
///
/// Lastly, [LSystem](LSystem) is enriched with a phantom type parameter `M`, which could be either
/// [Total](Total), [Indexed](Indexed), or [Unordered](Unordered).
///
/// - [Total](Total) represents L-Systems where _every token_ is applied to the production rules on every
///   iteration. This is nice for a proof-of-concept, but can become unweildy and inefficient very quickly.
/// - [Indexed](Indexed) represents L-Systems where the context `P` dictates which tokens are considered
///   active, by implementing [IndexedContext](IndexedContext). This can increase efficiency by skipping
///   inactive / "terminal" tokens, but will still store the entire string produced by the L-System.
/// - [Unordered](Unordered) represents L-Systems which have context-free grammars, and don't need the
///   string they generate to be stored in entireity. This saves on memory substantially, but requires the
///   alphabet to provide features like unique identifiers or other mechanisms in order to be useful.
///
/// Taking the example from
/// [Wikipedia on algae](https://en.wikipedia.org/wiki/L-system#Example_1:_Algae):
///
/// ```
/// use anabaena::{LSystem, LRulesHash, LRulesSet, Total};
/// use std::collections::HashMap;
/// use streaming_iterator::StreamingIterator;
///
/// let rules: LRulesHash<(), char, LRulesSet<char>> = |_| HashMap::from([
///     (
///         'A',
///         LRulesSet::new(vec![
///             (1, vec!['A', 'B']),
///         ]),
///     ),
///     (
///         'B',
///         LRulesSet::new(vec![
///             (1, vec!['A']),
///         ]),
///     )
/// ]);
///
/// let axiom: Vec<char> = vec!['A'];
///
/// let mut lsystem: LSystem<_,_,_, Total> = LSystem::new(
///     axiom,
///     rules,
/// );
///
/// assert_eq!(lsystem.next(), Some(&"AB".chars().collect()));
/// assert_eq!(lsystem.next(), Some(&"ABA".chars().collect()));
/// assert_eq!(lsystem.next(), Some(&"ABAAB".chars().collect()));
/// assert_eq!(lsystem.next(), Some(&"ABAABABA".chars().collect()));
/// assert_eq!(lsystem.next(), Some(&"ABAABABAABAAB".chars().collect()));
/// assert_eq!(lsystem.next(), Some(&"ABAABABAABAABABAABABA".chars().collect()));
/// assert_eq!(lsystem.next(), Some(&"ABAABABAABAABABAABABAABAABABAABAAB".chars().collect()));
/// ```
pub struct LSystem<R, P, T, M: LSystemMode = Total> {
    /// The current token string - set this to an initial value (i.e., the _axiom_) when creating your L-System
    pub string: Vec<T>,
    /// The production rules applied to the string
    pub rules: R,
    /// The mutable context used throughout the production rules, and lastly in-batch with `mut_context`
    pub context: P,
    /// Has been applied
    applied: bool,
    /// phantom data
    phantom: PhantomData<M>,
}

impl<R, P, T, M> LSystem<R, P, T, M>
where
    P: Default,
    M: LSystemMode,
{
    pub fn new(axiom: Vec<T>, rules: R) -> Self {
        Self {
            string: axiom,
            rules,
            context: Default::default(),
            applied: true,
            phantom: PhantomData,
        }
    }
}

impl<R, P, T, M> LSystem<R, P, T, M>
where
    M: LSystemMode,
{
    pub fn new_with_context(axiom: Vec<T>, rules: R, context: P) -> Self {
        Self {
            string: axiom,
            rules,
            context,
            applied: true,
            phantom: PhantomData,
        }
    }

    fn get_(&self) -> Option<&Vec<T>> {
        if self.applied {
            Some(&self.string)
        } else {
            None
        }
    }
}

impl<R, P, T> LSystem<R, P, T, Total>
where
    R: QualifiedProductionRules<P, T>,
    P: ProductionRuleContext<T>,
    T: PartialEq + Clone,
{
    fn advance_qualified_total(&mut self) {
        let mut rng = thread_rng();

        self.applied = false;

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
                        self.applied = true;
                        replacement.to_vec()
                    }
                }
            })
            .collect();

        if self.applied {
            self.string = new_string;
            self.context.on_complete(&self.string);
        }
    }
}

impl<R, P, T> LSystem<R, P, T, Total>
where
    R: ProductionRules<P, T>,
    P: ProductionRuleContext<T>,
    T: Clone,
{
    fn advance_context_free_total(&mut self) {
        let mut rng = thread_rng();

        self.applied = false;

        let new_string: Vec<T> = self
            .string
            .clone()
            .into_iter()
            .enumerate()
            .flat_map(
                |(i, c)| match self.rules.apply(&mut self.context, &c, i, &mut rng) {
                    None => vec![c],
                    Some(replacement) => {
                        self.applied = true;
                        replacement.to_vec()
                    }
                },
            )
            .collect();

        if self.applied {
            self.string = new_string;
            self.context.on_complete(&self.string);
        }
    }
}

impl<R, P, T> LSystem<R, P, T, Indexed>
where
    R: QualifiedProductionRules<P, T>,
    P: IndexedContext<T>,
    T: Clone,
{
    fn advance_qualified_indexed(&mut self) {
        let mut rng = thread_rng();

        let indicies = self.context.get_indicies();

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
                self.context.offset_indicies(i + 1, l);
            } else {
                orig.append(&mut tail);
                self.string.append(&mut orig);
            }
        }

        if self.applied {
            self.context.on_complete(&self.string);
        }
    }
}

impl<R, P, T> LSystem<R, P, T, Indexed>
where
    R: ProductionRules<P, T>,
    P: IndexedContext<T>,
    T: Clone,
{
    fn advance_context_free_indexed(&mut self) {
        let mut rng = thread_rng();

        let indicies = self.context.get_indicies();

        self.applied = false;

        let mut cumulative_offset: i64 = 0;

        for i in indicies {
            let mut orig = self
                .string
                .split_off((i as i64 + cumulative_offset) as usize);
            let mut tail = orig.split_off(1);
            if let Some(mut replacement) =
                self.rules.apply(&mut self.context, &orig[0], i, &mut rng)
            {
                self.applied = true;
                let l = replacement.len();
                cumulative_offset += l as i64 - 1;
                replacement.append(&mut tail);
                self.string.append(&mut replacement);
                self.context.offset_indicies(i + 1, l);
            } else {
                orig.append(&mut tail);
                self.string.append(&mut orig);
            }
        }

        if self.applied {
            self.context.on_complete(&self.string);
        }
    }
}

impl<R, P, T> LSystem<R, P, T, Unordered>
where
    R: ProductionRules<P, T>,
    P: ProductionRuleContext<T>,
    T: Clone,
{
    fn advance_context_free_unordered(&mut self) {
        let mut rng = thread_rng();

        self.applied = false;

        let mut new_string = vec![];

        for (i, c) in self.string.iter().enumerate() {
            if let Some(mut replacement) = self.rules.apply(&mut self.context, c, i, &mut rng) {
                self.applied = true;
                new_string.append(&mut replacement);
            }
        }

        self.string = new_string;

        if self.applied {
            self.context.on_complete(&self.string);
        }
    }
}

impl<P, T> StreamingIterator for LSystem<LRulesHash<P, T, LRulesQualified<T>>, P, T, Total>
where
    P: ProductionRuleContext<T>,
    T: Hash + Eq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_qualified_total()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystem<LRulesFunction<P, T, LRulesQualified<T>>, P, T, Total>
where
    P: ProductionRuleContext<T>,
    T: PartialEq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_qualified_total()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator
    for LSystem<LRulesFunctionWithIndex<P, T, LRulesQualified<T>>, P, T, Total>
where
    P: ProductionRuleContext<T>,
    T: PartialEq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_qualified_total()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystem<LRulesHash<P, T, LRulesSet<T>>, P, T, Total>
where
    P: ProductionRuleContext<T>,
    T: Hash + Eq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free_total()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystem<LRulesFunction<P, T, LRulesSet<T>>, P, T, Total>
where
    P: ProductionRuleContext<T>,
    T: Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free_total()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystem<LRulesFunctionWithIndex<P, T, LRulesSet<T>>, P, T, Total>
where
    P: ProductionRuleContext<T>,
    T: Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free_total()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystem<LRulesHash<P, T, LRulesQualified<T>>, P, T, Indexed>
where
    P: IndexedContext<T>,
    T: Hash + Eq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_qualified_indexed()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystem<LRulesFunction<P, T, LRulesQualified<T>>, P, T, Indexed>
where
    P: IndexedContext<T>,
    T: PartialEq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_qualified_indexed()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator
    for LSystem<LRulesFunctionWithIndex<P, T, LRulesQualified<T>>, P, T, Indexed>
where
    P: IndexedContext<T>,
    T: PartialEq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_qualified_indexed()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystem<LRulesHash<P, T, LRulesSet<T>>, P, T, Indexed>
where
    P: IndexedContext<T>,
    T: Hash + Eq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free_indexed()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystem<LRulesFunction<P, T, LRulesSet<T>>, P, T, Indexed>
where
    P: IndexedContext<T>,
    T: Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free_indexed()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystem<LRulesFunctionWithIndex<P, T, LRulesSet<T>>, P, T, Indexed>
where
    P: IndexedContext<T>,
    T: Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free_indexed()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystem<LRulesHash<P, T, LRulesSet<T>>, P, T, Unordered>
where
    P: ProductionRuleContext<T>,
    T: Hash + Eq + Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free_unordered()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator for LSystem<LRulesFunction<P, T, LRulesSet<T>>, P, T, Unordered>
where
    P: ProductionRuleContext<T>,
    T: Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free_unordered()
    }

    fn get(&self) -> Option<&Self::Item> {
        self.get_()
    }
}

impl<P, T> StreamingIterator
    for LSystem<LRulesFunctionWithIndex<P, T, LRulesSet<T>>, P, T, Unordered>
where
    P: ProductionRuleContext<T>,
    T: Clone,
{
    type Item = Vec<T>;

    fn advance(&mut self) {
        self.advance_context_free_unordered()
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
        let rules: LRulesSet<char> =
            LRulesSet::new(vec![(1, vec!['a']), (1, vec!['b']), (1, vec!['c'])]);
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

        let mut lsystem: LSystem<LRulesHash<(), _, _>, (), char> =
            LSystem::new(axiom.clone(), |_| {
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
            });

        #[derive(PartialEq, Eq, Debug)]
        struct IdxContext(usize);

        impl<T> ProductionRuleContext<T> for IdxContext {
            fn on_complete(&mut self, tokens: &[T]) {
                self.0 = tokens.len();
            }
        }

        impl<T> IndexedContext<T> for IdxContext {
            fn get_indicies(&self) -> Vec<usize> {
                (0..self.0).collect()
            }

            fn offset_indicies(&mut self, _i: usize, offset: usize) {
                if offset == 0 {
                    self.0 -= 1;
                } else {
                    self.0 += offset - 1;
                }
            }
        }

        let mut lsystem_selective: LSystem<
            LRulesHash<IdxContext, _, _>,
            IdxContext,
            char,
            Indexed,
        > = LSystem::new_with_context(
            axiom.clone(),
            |_| {
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
            IdxContext(axiom.len()),
        );

        for i in 1..20 {
            let x: Option<&Vec<char>> = lsystem.next();
            let y: Option<&Vec<char>> = lsystem_selective.next();
            assert_eq!(x, y, "LSystems aren't equal on iteration {i:}");
            assert_eq!(
                lsystem_selective.context.0,
                lsystem_selective.string.len(),
                "context and string length aren't equal."
            );
        }

        assert!(true);
    }
}
