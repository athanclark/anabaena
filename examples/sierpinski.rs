use anabaena::*;
use std::collections::HashMap;
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

    let rules: LRulesHash<(), SierpinskiAlphabet> = |_| {
        HashMap::from([
            (
                F,
                LRulesQualified {
                    no_context: Some(vec![(1, vec![F, Right, G, Left, F, Left, G, Right, F])]),
                    ..LRulesQualified::default()
                },
            ),
            (
                G,
                LRulesQualified {
                    no_context: Some(vec![(1, vec![G, G])]),
                    ..LRulesQualified::default()
                },
            ),
        ])
    };

    let mut lsystem = LSystem {
        string: vec![F, Right, G, Right, G],
        rules,
        context: (),
        mk_context: Box::new(|_, _| ()),
    };

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
