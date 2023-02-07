use anabaena::*;
use std::collections::HashMap;
use turtle::{Distance, Turtle};

#[derive(PartialEq, Eq, Debug, Hash, Clone)]
enum KochAlphabet {
    Forward,
    Left,
    Right,
}

fn main() {
    let rules: LRulesHash<(), _, _> = Box::new(|_| {
        HashMap::from([(
            KochAlphabet::Forward,
            LRulesQualified {
                no_context: Some(LRulesSet::new(vec![(
                    1,
                    vec![
                        KochAlphabet::Forward,
                        KochAlphabet::Left,
                        KochAlphabet::Forward,
                        KochAlphabet::Right,
                        KochAlphabet::Forward,
                        KochAlphabet::Right,
                        KochAlphabet::Forward,
                        KochAlphabet::Left,
                        KochAlphabet::Forward,
                    ],
                )])),
                ..LRulesQualified::default()
            },
        )])
    });

    let mut lsystem = LSystem {
        string: vec![KochAlphabet::Forward],
        rules,
        context: (),
        mut_context: Box::new(|_, _| {}),
    };

    let set = lsystem.nth(2).unwrap();

    let mut turtle = Turtle::new();
    turtle.use_degrees();
    turtle.pen_down();
    turtle.right(90.0);

    const UNIT_DISTANCE: Distance = 10.0;

    for x in set {
        match x {
            KochAlphabet::Forward => {
                turtle.forward(UNIT_DISTANCE);
            }
            KochAlphabet::Left => {
                turtle.left(90.0);
            }
            KochAlphabet::Right => {
                turtle.right(90.0);
            }
        }
    }
}
