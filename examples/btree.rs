use anabaena::*;
use std::collections::HashMap;
use turtle::{Angle, Distance, Point, Turtle};

#[derive(PartialEq, Eq, Debug, Hash, Clone)]
enum BTreeAlphabet {
    LineEndingInLeaf,
    Line,
    LeftPush,
    RightPop,
}

fn main() {
    let rules: LRulesHash<(), BTreeAlphabet> = |_| {
        HashMap::from([
            (
                BTreeAlphabet::Line,
                LRulesQualified {
                    no_context: Some(vec![(1, vec![BTreeAlphabet::Line, BTreeAlphabet::Line])]),
                    ..LRulesQualified::default()
                },
            ),
            (
                BTreeAlphabet::LineEndingInLeaf,
                LRulesQualified {
                    no_context: Some(vec![(
                        1,
                        vec![
                            BTreeAlphabet::Line,
                            BTreeAlphabet::LeftPush,
                            BTreeAlphabet::LineEndingInLeaf,
                            BTreeAlphabet::RightPop,
                            BTreeAlphabet::LineEndingInLeaf,
                        ],
                    )]),
                    ..LRulesQualified::default()
                },
            ),
        ])
    };

    let mut lsystem = LSystem {
        string: vec![BTreeAlphabet::LineEndingInLeaf],
        rules,
        context: (),
        mut_context: Box::new(|_, _| {}),
    };

    let set = lsystem.nth(6).unwrap();

    let mut turtle = Turtle::new();
    turtle.use_degrees();

    const UNIT_DISTANCE: Distance = 2.0;
    let mut lifo: Vec<(Point, Angle)> = vec![];
    let mut current_angle: Angle = 0.0;

    for x in set {
        match x {
            BTreeAlphabet::Line => {
                turtle.pen_down();
                turtle.forward(UNIT_DISTANCE);
                turtle.pen_up();
            }
            BTreeAlphabet::LineEndingInLeaf => {
                turtle.pen_down();
                turtle.forward(UNIT_DISTANCE);
                turtle.pen_up();
            }
            BTreeAlphabet::LeftPush => {
                // save the angle and position before turning
                lifo.push((turtle.position(), current_angle));
                // turn
                current_angle -= 45.0;
                turtle.left(45.0);
            }
            BTreeAlphabet::RightPop => {
                let (pos, new_angle) = lifo.pop().unwrap();
                let diff = new_angle - current_angle;
                current_angle = new_angle;
                // restore the saved angle and position
                turtle.right(diff);
                turtle.go_to(pos);
                // turn
                current_angle += 45.0;
                turtle.right(45.0);
            }
        }
    }
}
