use anabaena::*;
use std::collections::HashMap;
use turtle::{Distance, Turtle};

#[derive(PartialEq, Eq, Debug, Hash, Clone)]
enum SierpinskiAlphabet {
    A,
    B,
    Left,
    Right,
}

fn main() {
    use SierpinskiAlphabet::*;

    let rules: LRulesHash<(), _, _> = |_| {
        HashMap::from([
            (
                A,
                LRulesQualified {
                    no_context: Some(LRulesSet::new(vec![(1, vec![B, Right, A, Right, B])])),
                    ..LRulesQualified::default()
                },
            ),
            (
                B,
                LRulesQualified {
                    no_context: Some(LRulesSet::new(vec![(1, vec![A, Left, B, Left, A])])),
                    ..LRulesQualified::default()
                },
            ),
        ])
    };

    let mut lsystem = LSystem {
        string: vec![A],
        rules,
        context: (),
        mut_context: |_, _| {},
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
            A => {
                turtle.forward(UNIT_DISTANCE);
            }
            B => {
                turtle.forward(UNIT_DISTANCE);
            }
            Left => {
                turtle.left(60.0);
            }
            Right => {
                turtle.right(60.0);
            }
        }
    }
}
