use anabaena::*;
use std::collections::HashMap;
use streaming_iterator::StreamingIterator;
use turtle::{Distance, Turtle};

#[derive(PartialEq, Eq, Debug, Hash, Clone)]
enum SierpinskiAlphabet {
    F,
    G,
    Left,
    Right,
}

fn main() {
    use SierpinskiAlphabet::*;

    let rules: LRulesHash<(), _, _> = |_| {
        HashMap::from([
            (
                F,
                LRulesQualified {
                    no_context: Some(LRulesSet::new(vec![(
                        1,
                        vec![F, Right, G, Left, F, Left, G, Right, F],
                    )])),
                    ..LRulesQualified::default()
                },
            ),
            (
                G,
                LRulesQualified {
                    no_context: Some(LRulesSet::new(vec![(1, vec![G, G])])),
                    ..LRulesQualified::default()
                },
            ),
        ])
    };

    let mut lsystem = LSystem::new(vec![F, Right, G, Right, G], rules);

    let set = lsystem.nth(6).unwrap();

    let mut turtle = Turtle::new();
    turtle.use_degrees();
    turtle.set_speed(25);
    turtle.pen_up();
    turtle.go_to([-200.0, -200.0]);
    turtle.pen_down();

    const UNIT_DISTANCE: Distance = 3.0;

    for x in set {
        match x {
            F => {
                turtle.forward(UNIT_DISTANCE);
            }
            G => {
                turtle.forward(UNIT_DISTANCE);
            }
            Left => {
                turtle.left(120.0);
            }
            Right => {
                turtle.right(120.0);
            }
        }
    }
}
